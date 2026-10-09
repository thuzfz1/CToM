#!/usr/bin/env python3
# The sft_v7 EVAL mirror (evaluation/eval_math.py: run_v7_round, v7_anticipate,
# v7_slot_ids) against the training loop it has to reproduce
# (CToMAgentLoop._run_v7_round / ._v7_anticipate).
#
# Why these tests: an eval-side protocol that drifts from training does not
# error, it silently scores the wrong thing. sft_v4/v5 already shipped that
# failure (eval r_pred scored predictions against the real next user turn while
# training conditioned them on an internal draft). So the checks below pin the
# things that would drift in silence:
#   - the user-slot token split (header / close / generation prompt)
#   - which decisions terminate, and what COMMIT re-sends
#   - the inner revision loop: fires on UNSATISFIED, stops at the cap
#   - empty ASK falls through to DRAFT instead of sending an empty message
#   - internal drafts and anticipated turns never reach `visible`
#   - the parsers are ctom.controller's, not a copy
#
# Run:
#   RAYON_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_eval_parity.py
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctom import controller as _ctom  # noqa: E402

eval_math = pytest.importorskip("evaluation.eval_math")

# The policy tokenizer. These tests do the real prompt arithmetic, so they need
# the tokenizer the runs used; they skip instead of failing when it is absent.
# Point CTOM_TOKENIZER at a local directory to run fully offline.
TOKENIZER = os.environ.get("CTOM_TOKENIZER", "Qwen/Qwen3-8B")


def _tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as exc:  # no local copy and no network
        pytest.skip(f"tokenizer {TOKENIZER!r} unavailable: {exc}")


class StubAssistant(eval_math.AsyncAssistantClient):
    """Real tokenizer, real prompt arithmetic, scripted phase completions."""

    def __init__(self, script, **kw):
        self.script = list(script)
        self.calls = []
        self.model_name = "stub"
        self.temperature = 0.001
        self.top_p = 1.0
        self.num_retries = 1
        self.output_mode = "sft_v7"
        self.pred_answer_max_tokens = kw.get("draft_cap", 2048)
        self.prediction_max_tokens = kw.get("pred_cap", 384)
        self.trajectory_response_length = kw.get("traj", 30720)
        self.v7_think_max_tokens = kw.get("think_cap", 256)
        self.v7_ask_max_tokens = kw.get("ask_cap", 512)
        self.v7_max_inner_revisions = kw.get("max_inner", 3)
        self._v7_slot_cache = None
        self.tokenizer = _tokenizer()
        # Exactly the derivation AsyncAssistantClient.__init__ uses, so the stub
        # exercises the real _incremental_user_ids arithmetic rather than an
        # approximation of it.
        t1 = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
        )
        t2 = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
        )
        self._system_prompt_tokens = t1[: -(len(t2) - len(t1))]

    async def _generate_completion_phase(self, prompt, max_tokens, stop_text):
        self.calls.append((stop_text, max_tokens))
        if max_tokens <= 0:
            return "", []
        text = self.script.pop(0) if self.script else ""
        return text, self._encode_text(text)


def _round(script, *, last_shown="", task_type="math", **kw):
    a = StubAssistant(script, **kw)
    prompt = a.tokenizer.apply_chat_template(
        [{"role": "user", "content": "solve x"}], add_generation_prompt=True, tokenize=True
    )
    out = asyncio.run(
        eval_math.run_v7_round(
            a, list(prompt), 0, task_type=task_type, last_shown=last_shown
        )
    )
    return a, out


# ── the user-slot token split ────────────────────────────────────────────────
def test_slot_ids_match_the_documented_qwen3_layout():
    a = StubAssistant([])
    header, close, gen = a.v7_slot_ids()
    dec = a.tokenizer.decode
    assert dec(header).endswith(f"{_ctom.ANTICIPATED_MARKER}\n"), dec(header)
    assert "user" in dec(header)
    assert dec(gen).strip().endswith("assistant"), dec(gen)
    # close must be the turn terminator only -- if the generation prompt leaked
    # into it, every anticipated turn would open a spurious assistant turn.
    assert "assistant" not in dec(close), dec(close)
    assert a.v7_slot_ids() is a.v7_slot_ids(), "slot ids must be cached"


def test_slot_split_reproduces_the_training_derivation():
    """Same two-render diff CToMAgentLoop._v7_user_slot_ids performs."""
    a = StubAssistant([])
    tok = a.tokenizer
    mk = lambda t: a._incremental_user_ids(f"{_ctom.ANTICIPATED_MARKER}\n{t}")  # noqa: E731
    x, y = mk("ZZZZ"), mk("QQQQ")
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    j = 0
    while j < min(len(x), len(y)) - i and x[-1 - j] == y[-1 - j]:
        j += 1
    s0 = tok.apply_chat_template(
        [{"role": "user", "content": "x"}], add_generation_prompt=False, tokenize=False
    )
    s1 = tok.apply_chat_template(
        [{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False
    )
    gen_ref = tok(s1[len(s0):], add_special_tokens=False)["input_ids"]
    header, close, gen = a.v7_slot_ids()
    assert header == list(x[:i])
    assert gen == list(gen_ref)
    assert header + [0] + close + gen == header + [0] + list(x[len(x) - j:])


# ── decisions ────────────────────────────────────────────────────────────────
def test_draft_satisfied_shows_the_draft_and_continues():
    a, out = _round(["plan", "DRAFT", "the answer is 4", "ok thanks"])
    rec = out["record"]
    assert rec["decision"] == _ctom.DRAFT
    assert rec["verdicts"] == [_ctom.SATISFIED], rec["verdicts"]
    assert out["visible"] == "the answer is 4"
    assert out["last_shown"] == "the answer is 4"
    assert out["terminated"] is False
    assert rec["inner_iters"] == 0


def test_commit_resends_the_last_shown_draft_and_terminates():
    a, out = _round(["plan", "COMMIT"], last_shown="previous answer")
    assert out["record"]["decision"] == _ctom.COMMIT
    assert out["visible"] == "previous answer"
    assert out["terminated"] is True
    assert out["record"]["predictions"] == [], "COMMIT anticipates nothing"


def test_generated_tokens_count_every_sampled_phase_once():
    script = ["plan", "DRAFT", "discarded draft", "Please recheck?", "shown draft", "thanks"]
    a, out = _round(script)
    assert out["record"]["generated_tokens"] == sum(len(a._encode_text(t)) for t in script)
    assert out["record"]["controller_version"] == _ctom.CONTROLLER_VERSION


def test_commit_without_a_shown_draft_is_coerced():
    a, out = _round(["plan", "COMMIT", "a draft", "ok"], last_shown="")
    assert out["record"]["decision"] == _ctom.DRAFT
    assert out["record"]["coerce_reason"] == "commit_without_draft"


def test_ask_composes_question_plus_assumption():
    a, out = _round(["plan", "ASK", "is n indexed from 1?", "yes it is"])
    rec = out["record"]
    assert rec["decision"] == _ctom.ASK
    assert rec["ask"] == "is n indexed from 1?"
    assert out["visible"] == _ctom.compose_ask("is n indexed from 1?", "yes it is")
    assert "just reply" in out["visible"]
    assert out["last_shown"] == "", "an ASK shows no draft"


def test_empty_ask_falls_through_to_draft():
    a, out = _round(["plan", "ASK", "   ", "fallback draft", "ok"])
    rec = out["record"]
    assert rec["decision"] == _ctom.DRAFT
    assert rec["coerce_reason"] == "empty_ask"
    assert rec["validity"]["ask_valid"] is False
    assert out["visible"] == "fallback draft"


def test_unparseable_decision_is_coerced_and_marked_invalid():
    a, out = _round(["plan", "???", "a draft", "ok"])
    rec = out["record"]
    assert rec["decision"] == _ctom.DRAFT
    assert rec["validity"]["decision_valid"] is False
    assert rec["invalid_count"] >= 1


# ── the inner revision loop ──────────────────────────────────────────────────
def test_unsatisfied_verdict_triggers_an_internal_revision():
    a, out = _round([
        "plan", "DRAFT",
        "first draft", "no that is wrong, check step 2",   # UNSATISFIED
        "second draft", "ok thanks",                        # SATISFIED
    ])
    rec = out["record"]
    assert rec["verdicts"] == [_ctom.UNSATISFIED, _ctom.SATISFIED], rec["verdicts"]
    assert rec["inner_iters"] == 1
    assert rec["drafts"] == ["first draft", "second draft"]
    assert out["visible"] == "second draft", "only the LAST draft is shown"
    assert "first draft" not in out["visible"], "internal drafts must stay hidden"
    assert rec["counters"]["draft_changed"] == 1


def test_inner_loop_stops_at_the_configured_cap():
    push = "no that is wrong"
    a, out = _round(
        ["plan", "DRAFT"] + ["d1", push, "d2", push, "d3", push, "d4", push, "d5", push],
        max_inner=2,
    )
    rec = out["record"]
    assert rec["inner_iters"] == 2, rec["inner_iters"]
    assert len(rec["drafts"]) == 3, rec["drafts"]
    assert rec["counters"]["inner_cap_hit"] == 1
    assert out["visible"] == "d3"


def test_anticipated_turns_never_appear_in_the_visible_message():
    a, out = _round(["plan", "DRAFT", "the draft", "I think the answer is 7"])
    assert out["visible"] == "the draft"
    assert _ctom.ANTICIPATED_MARKER not in out["visible"]
    assert _ctom.PREDICTION_END_TAG not in out["visible"]
    assert "I think the answer is 7" not in out["visible"]


# ── budget ───────────────────────────────────────────────────────────────────
def test_round_is_skipped_when_the_trajectory_budget_is_exhausted():
    a, out = _round(["plan", "DRAFT", "d", "ok"], traj=8)
    assert out["record"] is None
    assert out["terminated"] is True
    assert a.calls == [], "no sampling should happen without budget for a round"


def test_phase_caps_are_passed_to_the_sampler():
    a, out = _round(["plan", "DRAFT", "d", "ok"], think_cap=77, draft_cap=333, pred_cap=99)
    stops = {s: m for s, m in a.calls}
    assert stops[_ctom.THINK_END_TAG] == 77
    assert stops[_ctom.DRAFT_END_TAG] == 333
    assert stops[_ctom.PREDICTION_END_TAG] == 99
    assert stops["\n"] == _ctom.DECISION_MAX_TOKENS


# ── parser provenance ────────────────────────────────────────────────────────
def test_verdicts_come_from_the_shared_controller_not_an_eval_side_copy(monkeypatch):
    seen = {}
    real = _ctom.parse_verdict

    def spy(text, task_type=None, **kwargs):
        seen["called"] = True
        seen["task_type"] = task_type
        seen["kwargs"] = kwargs
        return real(text, task_type, **kwargs)

    monkeypatch.setattr(eval_math._ctom, "parse_verdict", spy)
    _round(["plan", "DRAFT", "d", "ok"], task_type="writing")
    assert seen.get("called") is True, "eval must call ctom.controller.parse_verdict"
    assert seen["task_type"] == "writing", "task_type must reach the parser"
    # The draft has to reach the parser too, or the format-only rule silently
    # does nothing in eval while it is active in training.
    assert seen["kwargs"].get("draft") == "d", "eval must pass the draft to the parser"
