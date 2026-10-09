"""Pin the sft_v7 verdict parsers to the reply sets they were calibrated on
(eval_outputs/verdict_markers_src/, mined 2026-09-04; design doc §2.1).

Run:  RAYON_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_controller.py -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctom import controller  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "eval_outputs", "verdict_markers_src")


def _load(name):
    path = os.path.join(SRC, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    return json.load(open(path))


def _rate_unsat(replies, task_type, *, text_only):
    vs = []
    for r in replies:
        if text_only:
            r = r.replace(controller.TERMINATE_SIGNAL, "").strip()
            if not r:
                continue
        vs.append(controller.parse_verdict(r, task_type)[0] == controller.UNSATISFIED)
    return sum(vs) / len(vs), len(vs)


def test_math_gap_text_only():
    d = _load("math_replies3.json")
    pw, nw = _rate_unsat(d["wrong"], "math", text_only=True)
    pc, nc = _rate_unsat(d["correct"], "math", text_only=True)
    print(f"math text-only: P(UNSAT|wrong)={pw:.3f} (n={nw}) P(UNSAT|correct)={pc:.3f} (n={nc}) gap={pw-pc:+.3f}")
    assert pw >= 0.93, pw
    assert pc <= 0.25, pc
    assert pw - pc >= 0.70


def test_math_gap_with_terminate():
    d = _load("math_replies3.json")
    pw, _ = _rate_unsat(d["wrong"], "math", text_only=False)
    pc, _ = _rate_unsat(d["correct"], "math", text_only=False)
    print(f"math with-terminate: {pw:.3f} vs {pc:.3f} gap={pw-pc:+.3f}")
    assert pw - pc >= 0.80


def test_writing_critique_recall_and_accept_split():
    d = _load("writing_replies.json")
    pcrit, _ = _rate_unsat(d["critique"], "writing", text_only=False)
    pacc, _ = _rate_unsat(d["accept"], "writing", text_only=False)
    pask, _ = _rate_unsat(d["ask_reply"], "writing", text_only=False)
    print(f"writing: critique->UNSAT {pcrit:.3f}; accept-prefixed->UNSAT {pacc:.3f}; ask replies->UNSAT {pask:.3f}")
    assert pcrit >= 0.98
    assert 0.25 <= pacc <= 0.55        # accept-then-request cases
    assert 0.65 <= pask <= 0.90


@pytest.mark.parametrize("text,task,expected", [
    ("", "math", controller.SATISFIED),
    ("[[TERMINATE CHAT]]", "math", controller.SATISFIED),
    ("Thanks, that makes sense.", "math", controller.SATISFIED),
    ("The logic and calculations all check out. I'm satisfied.", "math", controller.SATISFIED),
    ("Hmm, I'm still not sure about step 3.", "math", controller.UNSATISFIED),
    ("Can you double-check the sum?", "math", controller.UNSATISFIED),
    ("Thanks, but step 3 is wrong.", "math", controller.UNSATISFIED),
    ("Nothing else, thanks!", "math", controller.SATISFIED),
    ("ok", "writing", controller.SATISFIED),
    ("That's correct, and I'd like to add the pricing section.", "writing", controller.UNSATISFIED),
    ("Nokia", "writing", controller.SATISFIED),
    ("The draft doesn't mention my profession.", "writing", controller.UNSATISFIED),
    ("Can you narrow it to the three styles I mentioned?", "writing", controller.UNSATISFIED),
])
def test_verdict_examples(text, task, expected):
    assert controller.parse_verdict(text, task)[0] == expected


def test_decision_parse_and_coerce():
    assert controller.parse_decision("DRAFT\n") == controller.DRAFT
    assert controller.parse_decision(" [ASK]") == controller.ASK
    assert controller.parse_decision("commit") == controller.COMMIT
    assert controller.parse_decision("maybe") is None
    assert controller.coerce_decision(None, has_shown_draft=False) == (controller.DRAFT, "invalid")
    assert controller.coerce_decision(controller.COMMIT, has_shown_draft=False) == (controller.DRAFT, "commit_without_draft")
    assert controller.coerce_decision(controller.COMMIT, has_shown_draft=True) == (controller.COMMIT, "")


def test_terminate_substitution():
    assert controller.substitute_terminate_target("[[TERMINATE CHAT]]", "math") == ("That makes sense, thanks.", True)
    assert controller.substitute_terminate_target("Thanks! [[TERMINATE CHAT]]", "math") == ("Thanks!", False)
    assert controller.substitute_terminate_target("", "math") == ("", False)
    assert controller.substitute_terminate_target("[[TERMINATE CHAT]]", "writing") == ("ok", True)


def test_visible_messages():
    ask = controller.visible_message(controller.ASK, ask="Is x positive?", prediction="Yes, x is positive.")
    assert ask.startswith("Is x positive?") and 'I expect the answer is: "Yes, x is positive."' in ask
    assert controller.visible_message(controller.DRAFT, draft="  the draft ") == "the draft"
    assert controller.visible_message(controller.COMMIT, last_shown="D", task_type="math") == "D"
    assert controller.visible_message(controller.COMMIT, last_shown="Doc.", task_type="writing").startswith("I WANT TO ANSWER:")


def test_sim_view_hides_and_overrides():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "d0"}, {"role": "user", "content": "[ANTICIPATED]\nno"},
            {"role": "assistant", "content": "d1"}, {"role": "user", "content": "[ANTICIPATED]\nok"}]
    out = controller.sim_view(msgs, hidden=[2, 3, 5], overrides={4: "D1 visible"})
    assert [m["content"] for m in out] == ["s", "q", "D1 visible"]


# ── format-only requests are not pushback ────────────────────────────────────
# Run 969590 revised drafts the predicted user had already accepted, because the
# question mark in "…Can you confirm the final answer in a box?" is a hard
# pushback marker. The revision cut the draft by 98% at the median and collapsed
# it to a bare \boxed{...} 7 times in 10 — the reasoning was thrown away.
_BOXED = r"Coordinates give AE = 2 + 2\sqrt{7}, so the answer is \boxed{31}"
_UNBOXED = "Coordinates give AE = 2 + 2 sqrt 7, so the answer is 31"


def test_a_box_request_on_an_already_boxed_draft_is_satisfied():
    reply = "That makes sense. I'm satisfied with the explanation. Can you confirm the final answer in a box?"
    assert controller.parse_verdict(reply, "math") == (controller.UNSATISFIED, "pushback")      # without the draft
    assert controller.parse_verdict(reply, "math", draft=_BOXED) == (controller.SATISFIED, "format_request")


def test_a_box_request_on_an_unboxed_draft_still_revises():
    reply = "That makes sense. Can you write the final answer in a box?"
    assert controller.parse_verdict(reply, "math", draft=_UNBOXED) == (controller.UNSATISFIED, "pushback")


def test_an_objection_is_never_excused_by_the_box_rule():
    for reply in [
        "That's not right, and please box it.",
        "Can you double-check step 2? Also box the answer.",
        "I'm still not convinced. Walk me through it again and box the result.",
        "The final answer looks wrong; state it in a box.",
    ]:
        assert controller.parse_verdict(reply, "math", draft=_BOXED)[0] == controller.UNSATISFIED, reply


def test_the_rule_is_off_without_a_draft_so_telemetry_is_unchanged():
    reply = "That makes sense. Can you confirm the final answer in a box?"
    assert controller.is_pushback(reply, "math") is True
    assert controller.parse_verdict(reply, "math")[0] == controller.UNSATISFIED


def test_ordinary_verdicts_are_untouched_by_the_draft_argument():
    for reply, expected in [
        ("Hmm, step 2 looks wrong?", controller.UNSATISFIED),
        ("I'm still not convinced. Can you walk me through it again?", controller.UNSATISFIED),
        ("That makes sense, thanks.", controller.SATISFIED),
        ("", controller.SATISFIED),
    ]:
        assert controller.parse_verdict(reply, "math", draft=_BOXED)[0] == expected, reply
        assert controller.parse_verdict(reply, "math", draft=_BOXED) == controller.parse_verdict(reply, "math") \
            or reply.startswith("That makes sense. Can you")


@pytest.mark.parametrize("reply", [
    "Can you correct the calculation and box the final answer?",
    "Please recompute the result and put it in a box.",
    "Could you justify step 2 and box the result?",
    "Can you give the final answer to three decimal places?",
    "That makes sense. Can you box it and verify the sign?",
    "Can you put it in a box? Check the denominator too.",
])
def test_format_escape_requires_every_clause_to_be_format_only(reply):
    assert controller.parse_verdict(reply, "math", draft=_BOXED) == (controller.UNSATISFIED, "pushback")


@pytest.mark.parametrize("draft", [r"\boxed{", r"\boxed{}", r"\boxed{   }",
                                        r"\boxed{31} then \boxed{32"])
def test_format_escape_requires_a_complete_nonempty_last_box(draft):
    assert controller.parse_verdict("Can you put it in a box?", "math", draft=draft)[0] == controller.UNSATISFIED


@pytest.mark.parametrize("reply", [
    "That makes sense… can you put it in a box?",
    "That makes sense, thanks. Could you please box the final answer?",
    "Please write the answer in boxed form.",
    r"Can you state the final answer using \boxed{} notation?",
])
def test_narrow_box_requests_accept_nested_latex(reply):
    assert controller.parse_verdict(reply, "math", draft=r"So \boxed{\frac{1}{2}}") == (controller.SATISFIED, "format_request")
