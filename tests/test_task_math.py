#!/usr/bin/env python3
"""Focused offline tests for the strict shared math verifier."""

from concurrent.futures import ThreadPoolExecutor

from ctom.task_math import (
    check_math_answer,
    extract_boxed_answers,
    extract_final_boxed_answer,
)
from ctom.reward import compute_task_reward


def test_nested_and_final_box_extraction():
    text = r"Intermediate \boxed{\frac{1}{2}}, final \boxed{\frac{3}{4}}."
    assert extract_boxed_answers(text) == [r"\frac{1}{2}", r"\frac{3}{4}"]
    assert extract_final_boxed_answer(text) == r"\frac{3}{4}"


def test_symbolically_equivalent_fraction():
    assert check_math_answer(r"Therefore \boxed{\frac{2}{4}}", r"\frac{1}{2}")


def test_final_box_wins():
    response = r"An early guess is \boxed{66}, but the final answer is \boxed{12}."
    assert check_math_answer(response, "12")
    assert not check_math_answer(response, "66")


def test_incidental_number_and_unboxed_answer_do_not_pass():
    assert not check_math_answer(r"We encounter 66, but conclude \boxed{12}.", "66")
    assert not check_math_answer("The answer is 66.", "66")


def test_wrong_box_gets_no_partial_reward():
    messages = [
        {"role": "user", "content": "What is the answer?"},
        {"role": "assistant", "content": r"I think it is \boxed{12}."},
    ]
    assert compute_task_reward(messages, "66", "solving") == 0.0


def test_symbolic_verification_inside_worker_thread():
    future = ThreadPoolExecutor(max_workers=1).submit(
        check_math_answer,
        r"Therefore \boxed{\frac{2}{4}}",
        r"\frac{1}{2}",
    )
    assert future.result(timeout=10)


def test_wedged_symbolic_verification_fails_closed():
    """A sympy check that never returns must be killed, not stall the caller.

    Regression test for the silent training hangs (jobs 802197/802202/811350):
    the forked verifier child inherits this monkeypatched sleep, and the
    parent must give up at _VERIFY_TIMEOUT_SECONDS and score the answer 0.
    """
    import time

    from ctom import task_math

    original_verify = task_math._symbolic_verify
    original_timeout = task_math._VERIFY_TIMEOUT_SECONDS
    task_math._symbolic_verify = lambda gold, candidate: time.sleep(600)
    task_math._VERIFY_TIMEOUT_SECONDS = 1.0
    try:
        start = time.monotonic()
        assert not check_math_answer(r"\boxed{2^{2}}", r"\boxed{3}")
        assert time.monotonic() - start < 30
    finally:
        task_math._symbolic_verify = original_verify
        task_math._VERIFY_TIMEOUT_SECONDS = original_timeout


if __name__ == "__main__":
    tests = [
        test_nested_and_final_box_extraction,
        test_symbolically_equivalent_fraction,
        test_final_box_wins,
        test_incidental_number_and_unboxed_answer_do_not_pass,
        test_wrong_box_gets_no_partial_reward,
        test_symbolic_verification_inside_worker_thread,
        test_wedged_symbolic_verification_fails_closed,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
