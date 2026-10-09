# BigCodeBench unit-test executor — the R_task of the CollabLLM `bigcodebench`
# domain ("runnable code->pass_rate" in their metric signature).
#
# ── Why this is NOT submission ───────────────────────────────────────
# the reference and the candidate function and compare values. BigCodeBench ships
# a `unittest.TestCase` suite per task instead, so the score is the fraction of
# TEST METHODS that pass — there is no reference output to diff against, and the
# tests routinely touch the filesystem, matplotlib figures, sqlite, mocked HTTP
# would reject essentially every BigCodeBench task, so it is deliberately not
# reused here; isolation comes from the subprocess + scratch cwd + timeout.
#
# ── The separate interpreter ─────────────────────────────────────────────────
# Task code runs under $CTOM_BCB_PYTHON, a venv that exists ONLY to execute
# BigCodeBench code (data_prep/coding_sandbox_requirements.txt). It is not the trainer's
# venv, for two independent reasons:
#   1. Correctness. The tasks were authored against numpy<2 / pandas 2.x; the
#      trainer runs numpy 1.26 with pandas 3.0, under which a large share of the
#      suites fail for reasons that have nothing to do with the policy.
#   2. Blast radius. Installing sklearn / seaborn / nltk / flask / django into the
#      trainer to make those suites runnable would put the torch + vllm + sglang
#      resolution at risk for the sake of the reward function.
# Measured on the first 60 large-split tasks: 23/60 reference solutions passed
# under the trainer venv, 57/60 under the sandbox. The difference is entirely
# ModuleNotFoundError.
#
# SECURITY NOTE: this executes model-generated Python
# already does. A fresh subprocess in a throwaway cwd with a wall-clock timeout
# is the boundary; there is no seccomp/network jail. Run only on disposable
# compute, which is what a training allocation is.

import asyncio
import atexit
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Reuse the submission-marker machinery: the marker contract is shared with every
# other submit-task family, and it must not fork.
from ctom.submission import (  # noqa: F401
    ANSWER_MARKER,
    answer_is_final,
    find_answer_marker,
    strip_code_fences,
)

logger = logging.getLogger(__name__)

# Wall-clock budget for ONE task's whole suite. Measured median ~0.7 s over 60
# tasks; the cap is for the pathological rollout (an infinite loop, an unmocked
# socket), not the normal one. Every rollout pays this in the worst case, so it
# is a training-throughput knob as much as a correctness one.
PASS_RATE_TIMEOUT = float(os.getenv("CTOM_BCB_TIMEOUT", "30.0"))

# How many suites may execute at once. Each is a subprocess that imports pandas
# and friends, so this caps memory and CPU contention on the reward node: without
# it a step of 144 rollouts would try to fork 144 interpreters. 16 is well under
# the 64 CPUs a training node allocates.
GRADING_CONCURRENCY = int(os.getenv("CTOM_BCB_CONCURRENCY", "16"))

# BigCodeBench's entry point, identical across all 1140 tasks.
ENTRY_POINT = "task_func"

# ── Background grading pool ──────────────────────────────────────────────────
# Grading is a blocking subprocess; the reward manager that calls it is an
# asyncio coordinator. Running the wait on a dedicated worker pool lets the
# coordinator keep serving other rollouts while a suite runs, and sizing the pool
# to GRADING_CONCURRENCY makes the machine-load bound a property of the pool
# rather than a convention every caller has to remember.
#
# A dedicated pool, not asyncio's default executor: the default is shared with
# every other to_thread caller in the process and sized min(32, cpu+4), so its
# width would be neither ours to choose nor stable.
_pool_lock = threading.Lock()
_grading_pool: Optional[ThreadPoolExecutor] = None


def grading_pool() -> ThreadPoolExecutor:
    """The process-wide pool that blocking test runs execute on."""
    global _grading_pool
    with _pool_lock:
        if _grading_pool is None:
            _grading_pool = ThreadPoolExecutor(
                max_workers=GRADING_CONCURRENCY, thread_name_prefix="bcb-grade"
            )
        return _grading_pool


@atexit.register
def _shutdown_grading_pool() -> None:
    with _pool_lock:
        if _grading_pool is not None:
            _grading_pool.shutdown(wait=False, cancel_futures=True)


# One semaphore per event loop: a Ray worker may drive more than one, and a
# semaphore bound to the wrong loop raises at await time. The semaphore bounds
# how many gradings are in flight; the pool bounds how many run at once. Both,
# because the pool alone would let an unbounded number of coroutines queue up
# holding their rollout state alive.
_semaphores: dict[int, asyncio.Semaphore] = {}


def _semaphore() -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    sem = _semaphores.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(GRADING_CONCURRENCY)
        _semaphores[loop_id] = sem
    return sem

_SENTINEL = "@@CTOM_BCB@@"

# The child program: candidate code, then the task's TestCases, then a runner
# that reports pass/total on stderr behind a sentinel. unittest is loaded under
# an alias so a task that rebinds the name `unittest` cannot break the harness.
_RUNNER = f'''

import io as _ctom_io, json as _ctom_json, sys as _ctom_sys
import unittest as _ctom_unittest

def _ctom_main():
    _suite = _ctom_unittest.TestLoader().loadTestsFromTestCase(TestCases)
    _total = _suite.countTestCases()
    _res = _ctom_unittest.TextTestRunner(
        stream=_ctom_io.StringIO(), verbosity=0
    ).run(_suite)
    _bad = (
        len(_res.failures)
        + len(_res.errors)
        + len(getattr(_res, "unexpectedSuccesses", []))
    )
    _payload = {{"passed": max(0, _total - _bad), "total": _total}}
    _ctom_sys.stderr.write("\\n{_SENTINEL}" + _ctom_json.dumps(_payload) + "\\n")

_ctom_main()
'''


def sandbox_python() -> str:
    """Interpreter that runs task code. $CTOM_BCB_PYTHON, else this one.

    Falling back is deliberate — a unit test or a laptop check should still run —
    but it is loud, because silently scoring under the trainer's package set is
    how a whole study ends up measuring ModuleNotFoundError.
    """
    exe = os.getenv("CTOM_BCB_PYTHON", "").strip()
    if exe and os.path.isfile(exe) and os.access(exe, os.X_OK):
        return exe
    if exe:
        logger.warning("CTOM_BCB_PYTHON=%r is not executable; using %s", exe, sys.executable)
    else:
        logger.warning(
            "CTOM_BCB_PYTHON unset; running BigCodeBench tests under %s. "
            "Expect ModuleNotFoundError on ~40%% of tasks.", sys.executable
        )
    return sys.executable


# ── Candidate repair ─────────────────────────────────────────────────────────
# Top-level `def name(` / `async def name(` at column 0.
_TOPLEVEL_DEF_RE = re.compile(r"^(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.M)


def alias_entry_point(code: str) -> tuple[str, bool]:
    """Bind ENTRY_POINT to the candidate's function when it chose another name.

    DIAGNOSTIC ONLY — this is not the scoring path. Since 2026-09-11 the required
    signature is PUBLISHED in the opening request, so a candidate that defines
    some other name ignored a public instruction, and strict scoring is the honest
    measurement. What this buys is the ability to report how much of a zero was a
    naming mismatch rather than wrong behaviour: score the same candidate with and
    without it and read the gap.

    (It is also the deterministic stand-in for CollabLLM's LLM extraction pass,
    which rewrites the code into the required opening block via
    `extraction_requirement`. Keeping it available keeps that comparison possible.)

    Aliases the LAST top-level def, which is the convention these solutions
    follow (helpers first, the task function last). Returns (code, aliased).
    """
    names = _TOPLEVEL_DEF_RE.findall(code or "")
    if not names or ENTRY_POINT in names:
        return code, False
    return f"{code}\n\n{ENTRY_POINT} = {names[-1]}\n", True


# ── Execution ────────────────────────────────────────────────────────────────

def run_tests(
    code: str,
    test_source: str,
    timeout: Optional[float] = None,
    *,
    repair_entry_point: bool = False,
) -> dict:
    """Run one task's suite against one candidate.

    Returns {"rate", "passed", "total", "error", "aliased"}. `rate` is 0.0 for
    every failure mode — empty candidate, syntax error, import error, timeout —

    Scoring is STRICT by default: the candidate must define ENTRY_POINT itself.
    That is only fair because the required signature is PUBLIC — it is appended to
    the opening request by data_prep/build_coding.py, so a policy that ignores
    it chose to. Pass repair_entry_point=True for the diagnostic score that
    aliases a renamed function (see alias_entry_point).
    """
    out = {"rate": 0.0, "passed": 0, "total": 0, "error": "", "aliased": False}
    code = strip_code_fences(code or "").strip()
    if not code:
        out["error"] = "empty_candidate"
        return out
    if not test_source or not test_source.strip():
        out["error"] = "no_tests"
        return out

    if repair_entry_point:
        code, out["aliased"] = alias_entry_point(code)

    program = code + "\n\n" + test_source + "\n" + _RUNNER
    env = dict(
        os.environ,
        # Headless: a task that plots must not try to open a display.
        MPLBACKEND="Agg",
        PYTHONDONTWRITEBYTECODE="1",
        # Task code is single-threaded; the reward manager runs many of these at
        # once and BLAS oversubscription is what turns 0.7 s into 30 s.
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        # Never let the trainer's site-packages shadow the sandbox's.
        PYTHONPATH="",
    )
    # PYTHONHOME would relocate the sandbox interpreter's own stdlib; VIRTUAL_ENV
    # is inherited from whatever activated the trainer. Both are cleared so the
    # child's environment is decided entirely by which interpreter we invoked.
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)

    try:
        with tempfile.TemporaryDirectory(prefix="ctom_bcb_") as td:
            path = os.path.join(td, "candidate.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(program)
            proc = subprocess.run(
                [sandbox_python(), path],
                cwd=td,               # tests that write files stay inside it
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout if timeout is not None else PASS_RATE_TIMEOUT,
                stdin=subprocess.DEVNULL,
            )
    except subprocess.TimeoutExpired:
        out["error"] = "timeout"
        return out
    except Exception as exc:                      # noqa: BLE001 - never kill a rollout
        out["error"] = f"spawn_failed: {type(exc).__name__}"
        return out

    for line in (proc.stderr or "").splitlines():
        if line.startswith(_SENTINEL):
            try:
                payload = json.loads(line[len(_SENTINEL):])
            except json.JSONDecodeError:
                break
            total = int(payload.get("total", 0))
            passed = int(payload.get("passed", 0))
            out["total"], out["passed"] = total, passed
            out["rate"] = (passed / total) if total else 0.0
            return out

    # No sentinel: the program died before the suite ran (syntax/import error).
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    out["error"] = tail[-1][:200] if tail else f"no_output(rc={proc.returncode})"
    return out


def pass_rate(code: str, test_source: str, timeout: Optional[float] = None) -> float:
    """Fraction of the task's test methods the candidate passes, in [0, 1]."""
    return float(run_tests(code, test_source, timeout)["rate"])


async def run_tests_async(
    code: str,
    test_source: str,
    timeout: Optional[float] = None,
    *,
    repair_entry_point: bool = False,
) -> dict:
    """`run_tests` on the grading pool, so the caller's event loop stays free.

    The coordinator awaits a worker instead of blocking on the subprocess, and the
    semaphore caps how many suites are in flight. Without this the reward
    manager's concurrency silently becomes 1: every other rollout in the step
    waits out this one's suite (~0.7 s median, PASS_RATE_TIMEOUT worst case).
    """
    async with _semaphore():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            grading_pool(),
            lambda: run_tests(code, test_source, timeout,
                              repair_entry_point=repair_entry_point),
        )


async def pass_rate_async(
    code: str, test_source: str, timeout: Optional[float] = None
) -> float:
    """Async counterpart of `pass_rate`."""
    result = await run_tests_async(code, test_source, timeout)
    return float(result["rate"])
