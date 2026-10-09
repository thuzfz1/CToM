# Copyright 2026 The CToM Authors
"""BDR — belief / decide / revise: shared constants and pure helpers.

Design: the retired BDR predecessor. This module is imported by the agent loop
(rollout), the reward manager (scoring), the eval harness (mirror) and the
tests, so everything here is pure Python with no verl/torch dependency.

One assistant turn (prediction_mode=sft_v6) is:

    [DRAFT_START]   <commit candidate>                      [DRAFT_END]
    [BELIEF_START]  <p=.. LABEL ...; p=.. LABEL ...>         [BELIEF_END]
    [DECISION] COMMIT | SHOW | ASK | REVISE
    [ASK_START]     <question + stated assumption>          [ASK_END]   (ASK only)

REVISE starts an *internal* turn: the controller appends INTERNAL_MESSAGE as
a user-role message and the policy generates again. Nothing from an internal
turn reaches the simulator; sim_visible() is the single filter every simulator
call goes through.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence

from ctom.utils import SPEC_TASK_TYPES

# ── Controller-owned tags ────────────────────────────────────────────────────
DRAFT_START_TAG = "[DRAFT_START]"
DRAFT_END_TAG = "[DRAFT_END]"
BELIEF_START_TAG = "[BELIEF_START]"
BELIEF_END_TAG = "[BELIEF_END]"
DECISION_TAG = "[DECISION]"
DECISION_TAG_NO_REVISE = "[DECISION: REVISE unavailable]"
ASK_START_TAG = "[ASK_START]"
ASK_END_TAG = "[ASK_END]"
# The decision is a single word; the phase is sampled with this hard cap and a
# newline stop so a policy that rambles cannot spend the turn budget here.
DECISION_MAX_TOKENS = 4

# ── Decisions ────────────────────────────────────────────────────────────────
COMMIT = "COMMIT"
SHOW = "SHOW"
ASK = "ASK"
REVISE = "REVISE"
DECISIONS = (COMMIT, SHOW, ASK, REVISE)
# SHOW = send the draft WITHOUT submitting so the author reacts to it. Only
# meaningful where the simulator reacts to a shown draft (writing). On math a
# wrong COMMIT already gets the expert's reaction, so SHOW is the same action.
SHOW_TASK_TYPES = ("writing",)

# ── Internal-turn controller message ─────────────────────────────────────────
INTERNAL_MARKER = "[INTERNAL]"
INTERNAL_MESSAGE = (
    f"{INTERNAL_MARKER} Your revised draft is now the current draft. "
    "Evaluate it again."
)

# ── Belief vocabulary (label is parsed; free text after it is allowed) ───────
BELIEF_LABELS: dict[str, tuple[str, ...]] = {
    "math": ("CONFIRM", "DOUBT", "ERROR", "WRONG_ANSWER", "ASKS_BOXED", "UNCLEAR"),
    "writing": ("ACCEPT", "FIX", "ANSWERS", "DONT_MIND", "TERMINATE"),
    "backend": ("ACCEPT", "REJECT", "DONT_KNOW", "ANSWERS"),
}
_ALL_LABELS = tuple(sorted({l for labels in BELIEF_LABELS.values() for l in labels}))


def belief_labels_for(task_type: Optional[str]) -> tuple[str, ...]:
    return BELIEF_LABELS.get(task_type or "math", BELIEF_LABELS["math"])


_PROB_RE = re.compile(r"p\s*[=:]\s*(1(?:\.0+)?|0?\.\d+|0|1)", re.I)
_PCT_RE = re.compile(r"(\d{1,3})\s*%")


def parse_belief(text: str, task_type: Optional[str] = None) -> list[dict[str, Any]]:
    """Parse belief lines into [{"p": float|None, "label": str, "detail": str}].

    Lenient on purpose: the belief has no reward of its own, so the parser's
    only jobs are the assumption line (leading item) and telemetry. A line
    counts if it names a known label for the domain (case-insensitive, as a
    whole word); the probability is optional (`p=0.6`, `0.6`, `60%`).
    """
    labels = belief_labels_for(task_type)
    out: list[dict[str, Any]] = []
    if not text:
        return out
    for raw_line in re.split(r"[\n;]+", text):
        line = raw_line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        found = None
        for label in labels:
            m = re.search(rf"(?<![A-Z_]){re.escape(label)}(?![A-Z_])", line, re.I)
            if m:
                found = (label, m)
                break
        if found is None:
            continue
        label, m = found
        p: Optional[float] = None
        pm = _PROB_RE.search(line)
        if pm:
            p = float(pm.group(1))
        else:
            pct = _PCT_RE.search(line)
            if pct:
                p = min(1.0, int(pct.group(1)) / 100.0)
        detail = line[m.end():].strip(" :,-—")
        out.append({"p": p, "label": label, "detail": detail})
    return out


def belief_top(text: str, task_type: Optional[str] = None) -> Optional[dict[str, Any]]:
    """The belief item with the highest stated probability (first if untied /
    unstated)."""
    items = parse_belief(text, task_type)
    if not items:
        return None
    best = items[0]
    for it in items[1:]:
        if (it["p"] or 0.0) > (best["p"] or 0.0):
            best = it
    return best


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


def coerce_decision(
    decision: Optional[str], *, task_type: Optional[str], revise_allowed: bool
) -> tuple[str, str]:
    """Apply the domain and budget rules. Returns (final_decision, reason) where
    reason is "" when nothing was coerced, else one of
    "invalid" | "show_coerced" | "revise_at_cap"."""
    if decision is None:
        return COMMIT, "invalid"
    if decision == SHOW and (task_type or "math") not in SHOW_TASK_TYPES:
        return COMMIT, "show_coerced"
    if decision == REVISE and not revise_allowed:
        return COMMIT, "revise_at_cap"
    return decision, ""


# ── Visible message ──────────────────────────────────────────────────────────
def visible_message(decision: str, draft: str, ask_text: str, task_type: Optional[str]) -> str:
    """The text the simulator sees for a non-REVISE decision.

    COMMIT on a submit task prepends the submission marker (text only: the
    DECISION token is what the policy learns; the marker is controller-owned).
    """
    task_type = task_type or "math"
    if decision == ASK:
        return (ask_text or "").strip()
    if decision == SHOW:
        return (draft or "").strip()
    # COMMIT
    if task_type in SPEC_TASK_TYPES:
        from ctom.submission import ANSWER_MARKER, answer_is_final

        body = (draft or "").strip()
        if answer_is_final(body):
            return body
        return f"{ANSWER_MARKER}\n{body}"
    return (draft or "").strip()


def sim_visible(messages: Sequence[Any], internal_indices: Iterable[int]) -> list[Any]:
    """Drop internal-turn entries (assistant internal drafts and the [INTERNAL]
    controller messages) so the simulator only ever sees COMMIT/SHOW/ASK
    messages and real user turns."""
    skip = set(int(i) for i in internal_indices)
    return [m for i, m in enumerate(messages) if i not in skip]


_CONFIRM_RE = re.compile(r"^\W*(ok|okay|yes|correct|right|confirmed)\b[\W]*$", re.I)


def user_confirmed(reply: Optional[str]) -> bool:
    """Whether a user reply is a bare confirmation of a stated assumption."""
    if not reply:
        return False
    r = reply.strip()
    return len(r) <= 40 and bool(_CONFIRM_RE.match(r))


def ask_has_assumption(ask_text: Optional[str]) -> bool:
    return bool(ask_text) and re.search(r"assum\w*\s*:", ask_text, re.I) is not None


# ── Draft scoring ────────────────────────────────────────────────────────────
# tau is the improvement a self-revision must clear to count. BigCodeBench
# pass_rate is a fraction of unit tests, so it moves in visible steps (a task
# with 5 tests steps by 0.2); 0.1 sits below one test on every suite with <=10
# tests, which is the bulk of them, for the
# same reason.
DEFAULT_TAU = {"math": 0.0, "writing": 0.02, "bigcodebench": 0.1}


def default_tau(task_type: Optional[str]) -> float:
    return DEFAULT_TAU.get(task_type or "math", 0.0)


def score_draft(
    text: str,
    task_type: Optional[str],
    ground_truth: str,
    interaction_kwargs: Optional[dict] = None,
) -> float:
    """Domain score of ONE deliverable text in [0, 1].

    math         -> strict final-\\boxed{} check (0/1)
    writing      -> ROUGE-L against the hidden article (the training signal; the
                    paper-protocol LLM score stays eval-only)
    backend      -> hidden-test pass fraction (differential testing)
    bigcodebench -> unittest pass_rate in the sandbox interpreter
    """
    task_type = task_type or "math"
    text = text or ""
    if not text.strip():
        return 0.0
    if task_type == "writing":
        from ctom.task_writing import document_similarity, strip_document_fences
        from ctom.submission import find_answer_marker

        body = text
        m = find_answer_marker(body)
        if m:
            body = body[m.end():]
        metric = str((interaction_kwargs or {}).get("writing_metric", "") or "rouge_l")
        return float(document_similarity(strip_document_fences(body).strip(), ground_truth, metric=metric))
    if task_type == "bigcodebench":
        from ctom.task_coding import pass_rate
        from ctom.submission import _extract_code_from_text

        tests = str((interaction_kwargs or {}).get("test_source", "") or "")
        if not tests:
            return 0.0
        # NOTE this is one SUBPROCESS per scored draft. A VOI/BDR episode scores
        # every draft and every counterfactual fork, so the cost is linear in
        # rounds, not constant like writing's ROUGE-L. Budget for it before
        # running sft_v6 on this domain.
        return float(pass_rate(_extract_code_from_text(text), tests))
    from ctom.task_math import check_math_answer

    return 1.0 if check_math_answer(text, ground_truth) else 0.0


# ── VOI reward (pure) ────────────────────────────────────────────────────────
VOI_DEFAULTS = {
    "alpha": 1.0,
    "beta": 0.15,
    "gamma": 0.15,
    "eps": 0.05,
    "pi": 0.2,  # per invalid phase; the launcher passes invalid_phase_penalty here
}

VOI_CELLS = ("needed_improved", "selffix_improved", "needed_flat", "selffix_flat")
# Every key compute_voi()["stats"] carries. The reward manager registers all of
# them in REWARD_EXTRA_INFO_SCHEMA (verl indexes every rollout with rollout 0's
# key list, so a key that appears on one path only is a mid-step KeyError).
VOI_STAT_KEYS = (
    "voi/num_scored_turns",
    "voi/cf_missing",
    "voi/g_mean",
    "voi/delta_mean",
    "voi/b1",
    "voi/b_last",
    "voi/cf_mean",
    "voi/turn_sum",
    "voi/task_raw",
    "voi/ask_rate",
    "voi/show_rate",
    "voi/internal_per_round",
    "voi/assumption_confirmed",
    "voi/invalid_phases",
) + tuple(f"voi/cell_{c}" for c in VOI_CELLS)


def compute_voi(
    rounds: Sequence[dict],
    cf_drafts: Sequence[Sequence[str]],
    *,
    task_type: Optional[str],
    ground_truth: str,
    interaction_kwargs: Optional[dict] = None,
    replies: Optional[Sequence[Optional[str]]] = None,
    alpha: float = VOI_DEFAULTS["alpha"],
    beta: float = VOI_DEFAULTS["beta"],
    gamma: float = VOI_DEFAULTS["gamma"],
    eps: float = VOI_DEFAULTS["eps"],
    pi: float = VOI_DEFAULTS["pi"],
    tau: Optional[float] = None,
    task_rescale: Optional[tuple[float, float]] = None,
    scorer=score_draft,
) -> dict[str, Any]:
    """Per-visible-turn VOI rewards and the terminal task reward.

    rounds[t]  : {"draft", "visible", "decision", "ask", "invalid_count", ...}
                 for visible turn t (one per real user round, in order).
    cf_drafts[t]: counterfactual self-revision drafts forked after visible
                 turn t ([] when unavailable -> g_t = 0 and cf_missing += 1).
    replies[t] : the real user reply after visible turn t (telemetry only).

    A visible turn t has a scored user reply iff round t+1 exists. The last
    visible turn carries only the format penalty and the terminal task reward.
    """
    task_type = task_type or "math"
    tau = default_tau(task_type) if tau is None else float(tau)
    n = len(rounds)
    cache: dict[str, float] = {}

    def _s(text: str) -> float:
        key = text or ""
        if key not in cache:
            cache[key] = float(scorer(key, task_type, ground_truth, interaction_kwargs))
        return cache[key]

    b = [_s(r.get("draft", "")) for r in rounds]
    turn_rewards: list[float] = []
    cells = {c: 0.0 for c in VOI_CELLS}
    g_vals: list[float] = []
    d_vals: list[float] = []
    cf_scores: list[float] = []
    cf_missing = 0
    scored = 0
    ask_turns = 0
    ask_confirmed = 0
    for t in range(n):
        inv = int(rounds[t].get("invalid_count", 0) or 0)
        r = -pi * inv
        has_reply = t + 1 < n
        if has_reply:
            scored += 1
            delta = b[t + 1] - b[t]
            cfs = list(cf_drafts[t]) if t < len(cf_drafts) and cf_drafts[t] else []
            if cfs:
                hits = [1.0 if (_s(c) - b[t]) >= max(tau, 1e-9) else 0.0 for c in cfs]
                g = sum(hits) / len(hits)
                cf_scores.extend(_s(c) for c in cfs)
            else:
                g = 0.0
                cf_missing += 1
            improved = delta >= max(tau, 1e-9)
            if improved:
                r += alpha * delta * (1.0 - g) - beta * g
                cells["needed_improved"] += 1.0 - g
                cells["selffix_improved"] += g
            else:
                r += -gamma * g
                cells["needed_flat"] += 1.0 - g
                cells["selffix_flat"] += g
            r -= eps
            g_vals.append(g)
            d_vals.append(delta)
            if rounds[t].get("decision") == ASK:
                ask_turns += 1
                reply = replies[t] if replies is not None and t < len(replies) else None
                if user_confirmed(reply):
                    ask_confirmed += 1
        turn_rewards.append(r)

    # Terminal task reward.
    if n == 0:
        r_task_raw = 0.0
    elif task_type in SPEC_TASK_TYPES:
        # Submit tasks: the last round's draft is the document / code, whether
        # or not it was submitted (matches extract_final_document's fallback).
        r_task_raw = b[-1]
    else:
        # Math: what the user actually saw last. An ASK that hoards a correct
        # boxed answer in the draft earns nothing until it is shown.
        r_task_raw = _s(rounds[-1].get("visible", rounds[-1].get("draft", "")))
    r_task = r_task_raw
    if task_rescale is not None:
        base, scale = task_rescale
        r_task = max(-1.0, min(1.0, (r_task_raw - float(base)) / (float(scale) or 1.0)))
    if turn_rewards:
        turn_rewards[-1] += r_task

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    frac = {c: (cells[c] / scored if scored else 0.0) for c in VOI_CELLS}
    decisions = [r.get("decision") for r in rounds]
    stats = {
        "voi/num_scored_turns": float(scored),
        "voi/cf_missing": float(cf_missing),
        "voi/g_mean": _mean(g_vals),
        "voi/delta_mean": _mean(d_vals),
        "voi/b1": float(b[0]) if b else 0.0,
        "voi/b_last": float(b[-1]) if b else 0.0,
        "voi/cf_mean": _mean(cf_scores),
        "voi/turn_sum": float(sum(turn_rewards) - (r_task if turn_rewards else 0.0)),
        "voi/task_raw": float(r_task_raw),
        "voi/ask_rate": (decisions.count(ASK) / n) if n else 0.0,
        "voi/show_rate": (decisions.count(SHOW) / n) if n else 0.0,
        "voi/internal_per_round": _mean([float(r.get("internal_turns", 0) or 0) for r in rounds]),
        "voi/assumption_confirmed": (ask_confirmed / ask_turns) if ask_turns else 0.0,
        "voi/invalid_phases": float(sum(int(r.get("invalid_count", 0) or 0) for r in rounds)),
    }
    for c in VOI_CELLS:
        stats[f"voi/cell_{c}"] = frac[c]
    assert set(stats) == set(VOI_STAT_KEYS), sorted(set(stats) ^ set(VOI_STAT_KEYS))
    return {
        "turn_rewards": turn_rewards,
        "r_task": float(r_task),
        "r_task_raw": float(r_task_raw),
        "b": b,
        "stats": stats,
    }


def voi_invariant_ok(*, beta: float, eps: float, rounds: int, task_max: float = 1.0) -> bool:
    """A solved episode with `rounds` wasteful, self-fixable user turns must
    still beat an unsolved single-round episode (design doc §4.4)."""
    return task_max - rounds * (beta + eps) > 0.0
