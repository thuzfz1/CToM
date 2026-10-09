"""The sft_v7 phase trace must survive the rollout.

Run 969590 finished with no record of any PLAN, decision or discarded draft:
`v7_rounds` was built, attached to extra_fields and dropped. These tests pin the
writer that persists it, and pin that it can never take a rollout down with it.
"""
import json
import os
from pathlib import Path

import pytest

from ctom import agent_loop as loop


class _AD:
    """Minimal stand-in for AgentData: only what the writer reads."""

    def __init__(self, rounds):
        self.v7_rounds = rounds
        self.request_id = "req-1"
        self.interaction_kwargs = {
            "single_turn_prompt": "What is 2+2?",
            "expertise": "expert",
            "task_type": "math",
        }
        self.user_turns = 1
        self.assistant_turns = 2
        self.phase_validity = [{"think_valid": True, "invalid_count": 0}]
        self.counterfactual_targets = ["that is wrong"]


def _round():
    return {
        "think": "the user wants a closed form",
        "decision": "DRAFT",
        "decision_raw": "DRAFT",
        "coerce_reason": "",
        "drafts": ["first draft, wrong", "second draft, right"],
        "predictions": ["I don't think that is right", "ok"],
        "verdicts": ["UNSATISFIED", "SATISFIED"],
        "verdict_reasons": ["marker", "idiom"],
        "ask": "",
        "visible": "second draft, right",
        "inner_iters": 1,
        "slots": [0, 1],
    }


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    loop._V7_TRACE_STATE.clear()
    monkeypatch.delenv("CTOM_V7_TRACE", raising=False)
    monkeypatch.delenv("CTOM_CONVERSATION_LOG", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    yield
    loop._V7_TRACE_STATE.clear()


def _only_file(d: Path) -> Path:
    files = sorted(d.glob("*.jsonl"))
    assert len(files) == 1, files
    return files[0]


def test_every_phase_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv("CTOM_V7_TRACE", str(tmp_path / "v7_trace.jsonl"))
    loop._v7_trace_write(_AD([_round()]))
    rec = json.loads(_only_file(tmp_path).read_text().strip())
    assert rec["problem"] == "What is 2+2?"
    assert rec["expertise"] == "expert"
    r = rec["rounds"][0]
    assert r["think"].startswith("the user wants")
    assert r["decision"] == "DRAFT"
    # the discarded draft, its anticipated reply and the verdict that rejected it
    assert r["drafts"] == ["first draft, wrong", "second draft, right"]
    assert r["predictions"][0] == "I don't think that is right"
    assert r["verdicts"] == ["UNSATISFIED", "SATISFIED"]
    assert r["inner_iters"] == 1
    assert rec["phase_validity"] and rec["counterfactual_targets"] == ["that is wrong"]


def test_one_file_per_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("CTOM_V7_TRACE", str(tmp_path / "v7_trace.jsonl"))
    loop._v7_trace_write(_AD([_round()]))
    name = _only_file(tmp_path).name
    assert name != "v7_trace.jsonl", "a shared append target interleaves across workers"
    assert name.startswith("v7_trace.") and name.endswith(f".{os.getpid()}.jsonl")


def test_appends_one_line_per_episode(tmp_path, monkeypatch):
    monkeypatch.setenv("CTOM_V7_TRACE", str(tmp_path / "t.jsonl"))
    for _ in range(3):
        loop._v7_trace_write(_AD([_round()]))
    assert len(_only_file(tmp_path).read_text().strip().splitlines()) == 3


def test_off_switch_and_empty_rounds(tmp_path, monkeypatch):
    monkeypatch.setenv("CTOM_V7_TRACE", "off")
    loop._v7_trace_write(_AD([_round()]))
    assert not list(tmp_path.glob("*"))
    loop._V7_TRACE_STATE.clear()
    monkeypatch.setenv("CTOM_V7_TRACE", str(tmp_path / "t.jsonl"))
    loop._v7_trace_write(_AD([]))
    assert not list(tmp_path.glob("*")), "an episode with no v7 round writes nothing"


def test_defaults_beside_the_conversation_log(tmp_path, monkeypatch):
    monkeypatch.setenv("CTOM_CONVERSATION_LOG", str(tmp_path / "conversations.jsonl"))
    loop._v7_trace_write(_AD([_round()]))
    assert _only_file(tmp_path).name.startswith("v7_trace.")


def test_unwritable_path_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CTOM_V7_TRACE", str(tmp_path / "no" / "such" / "t.jsonl"))
    blocked = tmp_path / "no"
    blocked.mkdir()
    blocked.chmod(0o400)
    try:
        loop._v7_trace_write(_AD([_round()]))  # must not raise
    finally:
        blocked.chmod(0o700)


def test_no_path_resolvable_is_silent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    loop._v7_trace_write(_AD([_round()]))
    assert not list(tmp_path.glob("**/*.jsonl"))


# ── the record the trace saves must be the real one, in generation order ─────
# These drive the actual round loop (not a hand-built record) so the test fails
# if a phase stops being recorded or the drafts stop being ordered.
from ctom import controller  # noqa: E402
from tests.test_agent_loop import _agent_data, _loop  # noqa: E402


def test_trace_carries_the_real_round_in_generation_order(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setenv("CTOM_V7_TRACE", str(tmp_path / "v7_trace.jsonl"))
    # plan, decision, then three drafts: the first two are doubted, the third accepted.
    loop = _loop([
        "n is small, just enumerate", "DRAFT",
        "draft A", "Hmm, step 2 looks wrong?",
        "draft B", "I'm still not sure.",
        "draft C", "Thanks, that makes sense.",
    ])
    data = _agent_data()
    asyncio.run(loop._run_v7_round(data, {}))
    from ctom import agent_loop as pl

    pl._v7_trace_write(data)  # the loop writes once per episode, at the end
    rec = json.loads(_only_file(tmp_path).read_text().strip())
    r = rec["rounds"][0]
    assert r["think"] == "n is small, just enumerate"
    assert r["decision"] == controller.DRAFT
    # order is the point: drafts as generated, the shown one last
    assert r["drafts"] == ["draft A", "draft B", "draft C"]
    assert r["visible"] == r["drafts"][-1] == "draft C"
    assert r["verdicts"] == [controller.UNSATISFIED, controller.UNSATISFIED, controller.SATISFIED]
    assert r["predictions"] == [
        "Hmm, step 2 looks wrong?", "I'm still not sure.", "Thanks, that makes sense."]
    assert r["inner_iters"] == 2
    assert len(rec["phase_validity"]) == 1
    assert rec["controller_version"] == controller.CONTROLLER_VERSION
    assert rec["written_at"]
    assert r["generated_tokens"] > 0


def test_ask_round_keeps_its_question_and_assumption(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setenv("CTOM_V7_TRACE", str(tmp_path / "v7_trace.jsonl"))
    loop = _loop(["need one fact", "ASK", "Is n even?", "Yes, n is even."])
    data = _agent_data()
    asyncio.run(loop._run_v7_round(data, {}))
    from ctom import agent_loop as pl

    pl._v7_trace_write(data)
    r = json.loads(_only_file(tmp_path).read_text().strip())["rounds"][0]
    assert r["decision"] == controller.ASK and r["ask"] == "Is n even?"
    assert r["drafts"] == [] and r["predictions"] == ["Yes, n is even."]
    assert "Is n even?" in r["visible"] and "Yes, n is even." in r["visible"]


def test_the_writer_is_wired_into_the_rollout(tmp_path):
    """The record is dropped downstream by design, so the call site is the fix."""
    import inspect

    from ctom.agent_loop import CToMAgentLoop

    src = inspect.getsource(CToMAgentLoop.run)
    assert "_v7_trace_write(agent_data)" in src, "the trace is never written"
    assert src.index("_v7_trace_write(agent_data)") < src.index("output = AgentLoopOutput("), \
        "the trace must be written before the rounds are handed off"
