#!/usr/bin/env python3
"""Gold-solution validation for the BigCodeBench domain.

Runs every task's REFERENCE solution against its own test suite in the sandbox
interpreter and writes the list of task_ids that score 1.0. Only those tasks are
usable for RL: a task whose own answer cannot pass in our environment has a
constant-0 reward for every rollout, so its GRPO group is degenerate (zero
advantage, no gradient) and it drags the reported R_task down by a fixed amount
that has nothing to do with the policy.

This is the preflight for the whole domain — run it before building parquet, and
re-run it whenever data_prep/coding_sandbox_requirements.txt changes.

Usage:
    CTOM_BCB_PYTHON=$SCRATCH/bcb_venv/bin/python \
    python data_prep/validate_coding_tasks.py \
        --files $SCRATCH/data/collabllm-bigcodebench-large/data/*.parquet \
                $SCRATCH/data/collabllm-bigcodebench/data/*.parquet \
        --out data/bigcodebench_runnable.json
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctom.task_coding import run_tests, sandbox_python  # noqa: E402


def collect_tasks(files: list[str]) -> dict[str, dict]:
    """One record per BigCodeBench task_id across all input parquets."""
    tasks: dict[str, dict] = {}
    for path in files:
        df = pd.read_parquet(os.path.expandvars(path))
        for _, row in df.iterrows():
            try:
                payload = json.loads(row["single_turn_completion"])
            except (TypeError, json.JSONDecodeError):
                continue
            tid = str(payload.get("task_id") or "")
            if not tid or tid in tasks:
                continue
            tasks[tid] = {
                "task_id": tid,
                "answer": payload.get("answer", "") or "",
                "test": payload.get("test", "") or "",
            }
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="BigCodeBench gold-solution validation")
    ap.add_argument("--files", nargs="+", required=True, help="source HF parquet file(s)")
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-task budget; generous here, since a gold solution "
                         "that needs >30s at validation will also time out in training")
    args = ap.parse_args()

    print(f"[INFO] sandbox interpreter: {sandbox_python()}")
    tasks = collect_tasks(args.files)
    print(f"[INFO] {len(tasks)} unique BigCodeBench tasks")

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(run_tests, t["answer"], t["test"], args.timeout,
                        repair_entry_point=False): tid
            for tid, t in tasks.items()
        }
        for i, fut in enumerate(as_completed(futs), 1):
            tid = futs[fut]
            try:
                results[tid] = fut.result()
            except Exception as exc:                     # noqa: BLE001
                results[tid] = {"rate": 0.0, "passed": 0, "total": 0,
                                "error": f"harness: {type(exc).__name__}", "aliased": False}
            if i % 50 == 0:
                print(f"[INFO] {i}/{len(tasks)}")

    runnable = sorted(t for t, r in results.items() if r["rate"] >= 1.0)
    partial = sorted(t for t, r in results.items() if 0.0 < r["rate"] < 1.0)
    dead = sorted(t for t, r in results.items() if r["rate"] <= 0.0)

    reasons: dict[str, int] = {}
    for tid in dead:
        err = results[tid]["error"] or "test_failure"
        key = err.split(":")[0][:60]
        reasons[key] = reasons.get(key, 0) + 1

    print(f"\n[RESULT] runnable {len(runnable)} | partial {len(partial)} | dead {len(dead)}")
    for key, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {count:4d}  {key}")

    out_path = os.path.expandvars(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "interpreter": sandbox_python(),
                "num_tasks": len(tasks),
                "runnable": runnable,
                "partial": {t: results[t] for t in partial},
                "dead": {t: results[t]["error"] for t in dead},
            },
            fh,
            indent=1,
        )
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
