# Copyright 2026 The CToM Authors
"""sft_v7 — decide / draft-predict-revise / ask-with-assumption: pure helpers.

Design: the paper's method section. Imported by the agent loop (rollout), the
reward manager (telemetry), the eval harness (mirror) and the tests, so
everything here is pure Python with no verl/torch dependency.

The opt-in categorical_v1 protocol uses A/B/C decisions and plain ASK without
prediction. See the paper's method section. The flow below
documents the preserved legacy protocol.

One round (one real user message -> one visible assistant message):

    assistant: [THINK_START] <plan> [THINK_END]  [DECISION] DRAFT | ASK | COMMIT
    DRAFT:     [DRAFT_START] draft_j [DRAFT_END]
               user-slot:  [ANTICIPATED]\\n<prediction_j>\\n[PREDICTION_END]   (policy-written)
               verdict_j = parse_verdict(prediction_j); UNSATISFIED -> a new assistant
               turn writes draft_{j+1}; SATISFIED or cap -> draft_j is shown.
    ASK:       [ASK_START] question [ASK_END] + the same anticipated user slot;
               the visible message is the question plus the prediction as a
               confirm-or-correct assumption.
    COMMIT:    the last shown draft is re-sent (submit marker on spec tasks); ends.

The verdict parsers were mined and calibrated on real simulator replies on
2026-09-04 (eval_outputs/verdict_markers_src/); tests/test_controller.py pins
the measured gaps.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from ctom.utils import PREDICTION_END_TAG, SPEC_TASK_TYPES
from ctom.markers import ASK_END_TAG, ASK_START_TAG, DRAFT_END_TAG, DRAFT_START_TAG  # noqa: F401
from ctom.prediction_loss import is_terminate_only_target, strip_termination_signal

# ── Controller-owned tags ────────────────────────────────────────────────────
# "PLAN", not "THINK": on smoke 967233 the raw Qwen3 treated [THINK_START] as a full
# reasoning phase and clipped the 256-token cap on ~90% of rounds.
THINK_START_TAG = "[PLAN_START]"
THINK_END_TAG = "[PLAN_END]"
DECISION_TAG = "[DECISION]"
# First line of the policy-written user turn. The chat-template header and this
# marker are controller text; the prediction after it is sampled.
ANTICIPATED_MARKER = "[ANTICIPATED]"
DECISION_MAX_TOKENS = 4

# ── Decisions ────────────────────────────────────────────────────────────────
DRAFT = "DRAFT"
ASK = "ASK"
COMMIT = "COMMIT"
DECISIONS = (DRAFT, ASK, COMMIT)

# Opt-in protocol; historical checkpoints keep their original controller.
ACTION_PROTOCOLS = ("legacy", "categorical_v1")
ACTION_LABELS = ("A", "B", "C")  # DRAFT, ASK, COMMIT respectively
CATEGORICAL_INSTRUCTION = """Controller protocol categorical_v1 (supersedes earlier action/ASK instructions):
After [DECISION], output exactly one character: A = DRAFT, B = ASK, C = COMMIT.
COMMIT is available only after a draft has been shown. ASK sends only a focused
question to the user. Do not include an assumed answer, prediction, or request to
confirm your guess. No anticipated-user slot follows ASK. Only DRAFT is followed
by an anticipated-user slot. Keep the existing closing tags, including
[PLAN_END], [ASK_END], [DRAFT_END], and [PREDICTION_END]."""


def validate_action_protocol(protocol: str) -> str:
    if protocol not in ACTION_PROTOCOLS:
        raise ValueError(f"Unknown v7 action protocol: {protocol!r}")
    return protocol


def action_token_ids(encode) -> list[int]:
    """Resolve and validate the tokenizer-specific single-token action alphabet."""
    ids = [list(encode(label)) for label in ACTION_LABELS]
    if any(len(row) != 1 for row in ids) or len({row[0] for row in ids}) != 3:
        raise ValueError(f"categorical_v1 requires three distinct single tokens: {ids}")
    return [row[0] for row in ids]


def action_prompt_messages(messages: list[dict], protocol: str) -> list[dict]:
    """Apply identical protocol instructions in train/eval without mutating data."""
    validate_action_protocol(protocol)
    if protocol == "legacy":
        return messages
    messages = [dict(message) for message in messages]
    for message in messages:
        if message.get("role") == "system":
            if CATEGORICAL_INSTRUCTION not in message["content"]:
                message["content"] += "\n\n" + CATEGORICAL_INSTRUCTION
            break
    else:
        messages.insert(0, {"role": "system", "content": CATEGORICAL_INSTRUCTION})
    return messages


def categorical_sampling_params(params: dict, allowed_ids: list[int]) -> dict:
    """No truncation/penalties may distort the categorical policy."""
    return {
        **params, "allowed_token_ids": allowed_ids, "top_p": 1.0, "top_k": -1,
        "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
        "frequency_penalty": 0.0, "min_tokens": 0, "stop": [],
        "stop_token_ids": [], "ignore_eos": True,
    }


def parse_action_token(ids: list[int], alphabet: list[int], has_shown: bool) -> str:
    legal = alphabet if has_shown else alphabet[:2]
    if len(ids) != 1 or ids[0] not in legal:
        raise ValueError(f"Constrained action sampling returned invalid token(s): {ids}")
    return DECISIONS[alphabet.index(ids[0])]

# ── Verdicts ─────────────────────────────────────────────────────────────────
SATISFIED = "SATISFIED"
UNSATISFIED = "UNSATISFIED"
VERDICT_MODES = ("parse",)  # "token" / "oracle" are spec ablations, not implemented
CONTROLLER_VERSION = "v7-20260907-format2-drop1"

TERMINATE_SIGNAL = "[[TERMINATE CHAT]]"

# Canonical satisfied utterance substituted for a terminate-only counterfactual
# target (design §2.1): 64.4% of expert replies after a correct math answer are
# the bare terminate signal, which pred_sft_skip_terminate would drop, leaving
# the aux loss 77/23 pushback-heavy.
TERMINATE_SUBSTITUTE = {
    "math": "That makes sense, thanks.",
    "writing": "ok",
    "bigcodebench": "ok",
}

# ── Decision parsing ─────────────────────────────────────────────────────────
_DECISION_RE = re.compile(r"\s*\[?\s*([A-Za-z]+)")


def parse_decision(text: str) -> Optional[str]:
    """First word of the decision phase, upper-cased, if it is a known decision."""
    if not text:
        return None
    m = _DECISION_RE.match(text)
    if not m:
        return None
    word = m.group(1).upper()
    return word if word in DECISIONS else None


def coerce_decision(decision: Optional[str], *, has_shown_draft: bool) -> tuple[str, str]:
    """Returns (final_decision, reason); reason "" when nothing was coerced,
    else "invalid" | "commit_without_draft"."""
    if decision is None:
        return DRAFT, "invalid"
    if decision == COMMIT and not has_shown_draft:
        return DRAFT, "commit_without_draft"
    return decision, ""


# ── Verdict parsing (calibrated markers, design §2.1) ────────────────────────
# Math: mined on 35,159 expert replies labelled by the strict checker on the
# answered turn. Text-only gap (terminate stripped): P(UNSAT|wrong)=0.956,
# P(UNSAT|correct)=0.196. Separation is form, not content.
_PUSHBACK_MATH = re.compile(
    r"\?|\b("
    r"not|still|could you|can you|please|but|wait|hmm|might|seems?|"
    r"doesn'?t|don'?t|isn'?t|wrong|mistakes?|incorrect|errors?|"
    r"double[- ]check(ed|ing)?|re-?(check|examine|evaluate|visit|consider|work|do)(ed|ing)?|"
    r"tr(y|ied|ying)|different|instead|should|how did|why|missing|actually|however|"
    r"unclear|confus(ed|ing|ion)?"
    r")\b",
    re.I,
)
# A satisfied idiom that also happens to contain a soft marker ("the calculations
# check out", "I think you've got it right") is satisfied unless a hard marker
# or a question is present.
_SAT_IDIOM_MATH = re.compile(
    r"\b(checks? out|i think you'?ve got it|i'?m satisfied|makes sense|looks (good|right|correct)|"
    r"well[- ]reasoned|got it)\b",
    re.I,
)
_HARD_PUSHBACK_MATH = re.compile(
    r"\?|\b(but|still|not|wrong|mistakes?|double[- ]check|re-?check)\b", re.I
)

# Writing: mined on 23,407 simulator reactions to shown drafts (964774 log).
# Critique 91.1% / accept 8.9%; 39% of accept-prefixed replies continue with a
# request, so pushback wins over the prefix.
_PUSHBACK_WRITING = re.compile(
    r"\?|\b("
    r"can you|could you|would you|please|should|add|include|incorporate|elaborate|expand|"
    r"focus|mention|emphasi[sz]e|more (about|detail|details|on)|rather than|instead|missing|"
    r"didn'?t|doesn'?t|don'?t|not (mention|include|specify|cover)|i'?d like|i want|i need|"
    r"change|remove|shorten|narrow|clarify|specif(y|ic)|but"
    r")\b",
    re.I,
)
_ACCEPT_WRITING = re.compile(
    r"^\W*(ok|okay|yes|yeah|fine|sure|great|perfect|looks good|sounds good|"
    r"that'?s (right|correct|fine|good)|that is (right|correct)|i don'?t mind|"
    r"leave it to you|up to you|no,? that'?s (all|fine|it))\b",
    re.I,
)

# ── Format-only requests ─────────────────────────────────────────────────────
# The simulator prompt tells the user that a correct-but-unboxed answer must not
# be accepted: "tell the AI to state its final answer inside \boxed{}". The
# policy learns to predict exactly that, and the reply comes out as an ACCEPTANCE
# plus a formatting request -- "That makes sense. Can you confirm the final
# answer in a box?". The markers above then read the question mark as hard
# pushback and the inner loop revises a draft the user was happy with. Measured
# on run 969590 step 180 that revision cut the draft by 98% at the median and
# collapsed it to a bare \boxed{...} 7 times in 10, i.e. the chain of thought
# was generated and then thrown away.
#
# A request for something the draft ALREADY contains is not an objection, so
# when the draft carries a final \boxed{} such a reply is SATISFIED. Without a
# box in the draft the request stands and the loop still revises. This changes
# the inner loop's exit test: runs from 2026-09-06 are not directly comparable
# to 967309 / 969590 / 969591 / 975929 / 975930 on revision counts or shown
# length. `draft` defaults to None, which reproduces the old behaviour exactly,
# so the telemetry helper `is_pushback` is unaffected.
_BOXED_IN_DRAFT = re.compile(r"\\boxed\s*\{", re.I)
# Every clause must be a known acceptance or a boxing request. A blacklist of
# objections lets "correct the calculation and box it" bypass the revision
# gate. Unknown wording must retain the original parser's verdict.
_FORMAT_ACCEPTANCE = re.compile(
    r"(?:that makes sense|(?:i am|i'm) satisfied(?: with (?:the|your) explanation)?|"
    r"(?:the )?(?:logic and )?calculations (?:all )?check out|"
    r"(?:that(?:'s| is)|it(?:'s| is)) (?:right|correct|good)|"
    r"(?:it )?looks (?:good|right|correct)|thanks(?: a lot)?|thank you|ok(?:ay)?|yes)",
    re.I,
)
_BOX_OBJECT = r"(?:it|(?:(?:the|your) )?(?:final )?(?:answer|result))"
_BOX_REQUEST = re.compile(
    r"(?:(?:can|could|would) you )?(?:please )?(?:"
    r"box " + _BOX_OBJECT + r"|"
    r"(?:put|state|write|give|provide|show|confirm) " + _BOX_OBJECT
    + r" (?:in(?:side)? (?:a|the) box|in boxed form|using \\boxed\{\}(?: notation)?))"
    r"(?: please)?",
    re.I,
)


def has_complete_final_box(draft: str) -> bool:
    """Require the last box opening to have balanced, nonempty contents."""
    matches = list(_BOXED_IN_DRAFT.finditer(draft))
    if not matches:
        return False
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(draft)):
        depth += (draft[i] == "{") - (draft[i] == "}")
        if depth == 0:
            return bool(draft[start:i].strip())
    return False


def is_format_only_request(prediction: str, draft: Optional[str]) -> bool:
    """The reply asks only for form, and the draft already has that form."""
    if not draft or not has_complete_final_box(draft):
        return False
    clauses = [c.strip() for c in re.split(r"[.!?,;…]+", prediction.replace("’", "'")) if c.strip()]
    requests = [_BOX_REQUEST.fullmatch(c) is not None for c in clauses]
    return any(requests) and all(
        request or _FORMAT_ACCEPTANCE.fullmatch(clause) is not None
        for clause, request in zip(clauses, requests)
    )


_TAG_RESIDUE = re.compile(
    r"\[/?(ANTICIPATED|PREDICTION_START|PREDICTION_END|DRAFT_START|DRAFT_END|"
    r"ASK_START|ASK_END|PLAN_START|PLAN_END|THINK_START|THINK_END)\]",
    re.I,
)


def clean_prediction(text: Optional[str]) -> str:
    """Strip controller residue and collapse whitespace."""
    if not text:
        return ""
    t = _TAG_RESIDUE.sub(" ", text)
    return re.sub(r"\s+", " ", t).strip()


def parse_verdict(
    prediction: Optional[str],
    task_type: Optional[str] = None,
    *,
    draft: Optional[str] = None,
) -> tuple[str, str]:
    """(verdict, reason). reason in {"empty", "terminate", "pushback", "idiom",
    "accept", "short", "format_request", "default"}. Never loops on garbage:
    empty -> SATISFIED.

    Pass `draft` (the text the reply is reacting to) to enable the format-only
    rule: a reply whose only request is a form the draft already satisfies is an
    acceptance, not pushback. Omit it for pure telemetry to keep the historical
    labels."""
    raw = prediction or ""
    if TERMINATE_SIGNAL in raw:
        return SATISFIED, "terminate"
    t = clean_prediction(raw)
    if not t:
        return SATISFIED, "empty"
    tt = task_type or "math"
    if tt in SPEC_TASK_TYPES:
        if _PUSHBACK_WRITING.search(t):
            return UNSATISFIED, "pushback"
        if _ACCEPT_WRITING.match(t):
            return SATISFIED, "accept"
        if len(t.split()) <= 4:
            return SATISFIED, "short"
        return UNSATISFIED, "default"
    # math
    if _PUSHBACK_MATH.search(t):
        if _SAT_IDIOM_MATH.search(t) and not _HARD_PUSHBACK_MATH.search(t):
            return SATISFIED, "idiom"
        if is_format_only_request(t, draft):
            return SATISFIED, "format_request"
        return UNSATISFIED, "pushback"
    return SATISFIED, "default"


def is_pushback(text: Optional[str], task_type: Optional[str] = None) -> bool:
    """Telemetry helper: the same parser applied to a real / counterfactual reply."""
    return parse_verdict(text, task_type)[0] == UNSATISFIED


# ── Visible messages ─────────────────────────────────────────────────────────
ASSUMPTION_TEMPLATE = (
    '\n\nI expect the answer is: "{prediction}". If that is right, just reply "ok"; '
    "otherwise correct me."
)
_ASSUMPTION_MAX_CHARS = 400


def compose_ask(question: str, prediction: str) -> str:
    """The visible ASK message: question + the prediction as an assumption."""
    q = (question or "").strip()
    p = clean_prediction(prediction)
    if len(p) > _ASSUMPTION_MAX_CHARS:
        p = p[:_ASSUMPTION_MAX_CHARS].rstrip() + "…"
    if not p:
        return q
    return q + ASSUMPTION_TEMPLATE.format(prediction=p.replace('"', "'"))


def visible_message(
    decision: str,
    *,
    draft: str = "",
    ask: str = "",
    prediction: str = "",
    last_shown: str = "",
    task_type: Optional[str] = None,
) -> str:
    """Text the simulator sees for a round."""
    task_type = task_type or "math"
    if decision == ASK:
        return compose_ask(ask, prediction)
    if decision == DRAFT:
        return (draft or "").strip()
    # COMMIT: re-send the last shown draft; spec tasks get the submission marker.
    body = (last_shown or draft or "").strip()
    if task_type in SPEC_TASK_TYPES:
        from ctom.submission import ANSWER_MARKER, answer_is_final

        if answer_is_final(body):
            return body
        return f"{ANSWER_MARKER}\n{body}"
    return body


def anticipated_user_content(prediction: str) -> str:
    """Content of the policy-written user turn as stored in the message list."""
    return f"{ANTICIPATED_MARKER}\n{(prediction or '').strip()}\n{PREDICTION_END_TAG}"


# ── Counterfactual targets ───────────────────────────────────────────────────
def cf_history(sim_messages: Sequence[Any], assistant_text: str) -> list[dict]:
    """History the simulator is replayed on to label one prediction slot: the
    sim-visible transcript up to the round, plus the draft / bare question."""
    out = [dict(m) if isinstance(m, dict) else {"role": m.role, "content": m.content} for m in sim_messages]
    out.append({"role": "assistant", "content": (assistant_text or "").strip()})
    return out


def substitute_terminate_target(target: Optional[str], task_type: Optional[str] = None) -> tuple[str, bool]:
    """A terminate-only counterfactual reply becomes the canonical satisfied
    utterance (returns (text, True)); a mixed reply keeps its words with the
    signal stripped; anything else passes through."""
    t = (target or "").strip()
    if not t:
        return "", False
    if is_terminate_only_target(t):
        return TERMINATE_SUBSTITUTE.get(task_type or "math", TERMINATE_SUBSTITUTE["math"]), True
    return strip_termination_signal(t), False


def sim_view(messages: Sequence[Any], hidden: Sequence[int], overrides: dict[int, str]) -> list[Any]:
    """The simulator's transcript: drop hidden entries (internal drafts and
    anticipated user turns), substitute the composed visible text where a
    round's message differs from what the policy saw."""
    skip = set(int(i) for i in hidden)
    out = []
    for i, m in enumerate(messages):
        if i in skip:
            continue
        if i in overrides:
            if isinstance(m, dict):
                m = dict(m)
                m["content"] = overrides[i]
            else:
                m = {"role": m.role, "content": overrides[i]}
        out.append(m)
    return out
