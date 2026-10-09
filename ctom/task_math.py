"""Strict, shared verifier for CToM math training and evaluation.

Only the final ``\\boxed{...}`` answer in an assistant turn is eligible for
credit. This avoids rewarding an incidental occurrence of the ground-truth
number in the reasoning. Equivalence is checked with ``math_verify`` so
different but mathematically equal forms (for example 1/2 and 2/4) match.
"""

import logging
import multiprocessing
import re

from math_verify import LatexExtractionConfig, parse, verify

logger = logging.getLogger(__name__)

_LATEX_CONFIG = (LatexExtractionConfig(),)
_MAX_SYMBOLIC_ANSWER_CHARS = 1024
_VERIFY_TIMEOUT_SECONDS = 15.0
# fork, not spawn: the child must reuse the parent's already-imported
# sympy/math_verify rather than paying multi-second re-imports per call.
_MP_CTX = multiprocessing.get_context("fork")


def extract_boxed_answers(text: str) -> list[str]:
    """Return all complete ``\\boxed{...}`` contents, including nested braces."""
    answers: list[str] = []
    search_from = 0
    while True:
        match = re.search(r"\\boxed\s*\{", text[search_from:])
        if match is None:
            break

        start = search_from + match.end()
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1

        if depth != 0:
            break
        answers.append(text[start : pos - 1])
        search_from = pos
    return answers


def extract_final_boxed_answer(text: str) -> str:
    """Return the final complete boxed answer, or an empty string."""
    answers = extract_boxed_answers(text)
    return answers[-1].strip() if answers else ""


def _normalize_exact(answer: str) -> str:
    """Conservative normalization used before symbolic verification."""
    answer = answer.strip().strip("$")
    answer = answer.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    answer = answer.replace(r"\left", "").replace(r"\right", "")
    return re.sub(r"\s+", "", answer)


def _symbolic_verify(gold: str, candidate: str) -> bool:
    """Unbounded math_verify equivalence check.

    With the internal timeouts disabled, sympy can spin forever on
    pathological expressions — only call this inside a killable child
    process (_symbolic_verify_with_timeout).
    """
    parsed_gold = parse(
        rf"\boxed{{{gold}}}",
        _LATEX_CONFIG,
        fallback_mode="no_fallback",
        extraction_mode="any_match",
        parsing_timeout=None,
    )
    parsed_candidate = parse(
        rf"\boxed{{{candidate}}}",
        _LATEX_CONFIG,
        fallback_mode="no_fallback",
        extraction_mode="any_match",
        parsing_timeout=None,
    )
    return bool(
        parsed_gold
        and parsed_candidate
        and verify(
            parsed_gold,
            parsed_candidate,
            strict=True,
            timeout_seconds=None,
        )
    )


def _symbolic_verify_child(gold: str, candidate: str, conn) -> None:
    # No logging in the child: it may inherit locks held by other threads
    # of the forked parent. Failures surface in the parent as False.
    try:
        result = _symbolic_verify(gold, candidate)
    except Exception:
        result = False
    try:
        conn.send(result)
    finally:
        conn.close()


def _symbolic_verify_with_timeout(gold: str, candidate: str) -> bool:
    """Run the sympy equivalence check under a hard wall-clock limit.

    math_verify's signal-based timeouts cannot run inside veRL's threaded
    reward loop, and thread-based timeouts cannot interrupt sympy's C-level
    loops. A SIGKILL-able child process is the only reliable guard; on
    timeout the answer is scored as incorrect (fail closed) so one
    pathological expression cannot stall training forever.
    """
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)
    proc = _MP_CTX.Process(
        target=_symbolic_verify_child, args=(gold, candidate, child_conn), daemon=True
    )
    proc.start()
    child_conn.close()
    try:
        if parent_conn.poll(_VERIFY_TIMEOUT_SECONDS):
            return bool(parent_conn.recv())
        logger.warning(
            "math_verify exceeded %.0fs; assigning no task credit: gold=%r candidate=%r",
            _VERIFY_TIMEOUT_SECONDS,
            gold[:200],
            candidate[:200],
        )
        return False
    except (EOFError, OSError):
        # Child died without reporting (e.g. sympy segfault). Fail closed.
        return False
    finally:
        parent_conn.close()
        if proc.is_alive():
            proc.kill()
        proc.join(5)


def check_math_answer(response: str, ground_truth: str) -> bool:
    """Check the final boxed response against the ground truth.

    A response without a complete boxed answer receives no credit. Ground
    truth may itself be boxed or may be a bare LaTeX expression.
    """
    candidate = extract_final_boxed_answer(response)
    if not candidate:
        return False

    gold = extract_final_boxed_answer(ground_truth) or ground_truth.strip()
    if not gold:
        return False

    if _normalize_exact(candidate) == _normalize_exact(gold):
        return True
    if len(candidate) > _MAX_SYMBOLIC_ANSWER_CHARS or len(gold) > _MAX_SYMBOLIC_ANSWER_CHARS:
        logger.warning("Math answer too long for symbolic verification; assigning no task credit")
        return False

    try:
        return _symbolic_verify_with_timeout(gold, candidate)
    except Exception as exc:
        # Fail closed: verifier trouble must never become positive reward.
        logger.warning("math_verify failed; assigning no task credit: %s", exc)
        return False
