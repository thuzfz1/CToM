"""Regression tests for the shape of the simulator's "response" field.

`generate_response` is annotated `-> tuple[bool, str, float, dict]`, and every
consumer relies on slot 2 being a str. When the simulator emitted a NESTED
object — {"response": {...}} instead of {"response": "..."} — a dict reached
`extract_sim_response`, whose first act is `text.strip()`, and the
AttributeError killed the 8-node run 1002602 at step 99 (2026-09-17), rolling it
back 47 steps to global_step_50. The guard below (`if not response`) only tested
falsiness, and a non-empty dict passes that.

The fix coerces any non-str payload into the existing empty-response case at the
one place the field is read, so all three exposed call sites in
agent_loop.py (469, 1341, 1893) are covered at once.

Run:
    .venv/bin/python -m pytest tests/test_simulator_parsing.py
"""
import json

from ctom.user_simulator import _parse_sim_json_response
from ctom.utils import extract_sim_response


def _wrap(payload):
    return json.dumps({"current_status": "c", "thought": "t", "response": payload})


def test_nested_object_becomes_empty():
    for payload in ({"text": "hi"}, ["a", "b"], 17, True):
        out = _parse_sim_json_response(_wrap(payload))
        assert isinstance(out["response"], str), (payload, out)
        assert out["response"] == "", (payload, out)
        # It parsed fine — the model just gave an unusable shape — so this is the
        # empty-response case, NOT a parse failure that falls back to raw text.
        assert out["_empty_response"] is True, (payload, out)
        assert out["_parse_failed"] is False, (payload, out)


def test_normal_string_is_untouched():
    out = _parse_sim_json_response(_wrap("Could you clarify step 2?"))
    assert out["response"] == "Could you clarify step 2?", out
    assert out["_parse_failed"] is False, out
    assert "_empty_response" not in out, out


def test_genuinely_empty_string_keeps_old_behaviour():
    out = _parse_sim_json_response(_wrap(""))
    assert out["response"] == "", out
    assert out["_empty_response"] is True, out
    assert out["_parse_failed"] is False, out


def test_downstream_extract_no_longer_raises():
    # The exact crash: extract_sim_response(<dict>) -> AttributeError. Feeding it
    # what the parser now returns must be inert.
    out = _parse_sim_json_response(_wrap({"text": "hi"}))
    assert extract_sim_response(out["response"] or "") == ""


def test_raw_text_fallback_still_wins_when_json_is_absent():
    out = _parse_sim_json_response("no json here at all")
    assert out["response"] == "no json here at all", out
    assert out["_parse_failed"] is True, out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
