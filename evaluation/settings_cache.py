"""Validate v7 eval reuse: 0=complete match, 1=retry, 2=use a new output label."""
import argparse
import json
from pathlib import Path

from ctom.controller import CONTROLLER_VERSION


def cache_status(summary, expected, population):
    config = summary.get("config", {})
    for key, value in {"v7_action_protocol": "legacy", **expected, "controller_version": CONTROLLER_VERSION}.items():
        actual = config.get(key)
        if key == "v7_action_protocol" and actual is None:
            actual = "legacy"
        if key == "test_data" and actual is not None:
            matches = Path(actual).resolve() == Path(value).resolve()
        else:
            matches = actual == value
        if not matches:
            return 2, f"{key}: saved={actual!r}, requested={value!r}"
    if summary.get("total_samples") != population:
        return 2, f"saved population differs from requested {population}"
    if summary.get("successful") != population or summary.get("errors", 0):
        return 1, "incomplete or errored evaluation"
    return 0, "complete matching evaluation"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", type=Path)
    ap.add_argument("--expect", nargs="+", required=True)
    args = ap.parse_args()
    if not args.summary.exists():
        return 1
    expected = {}
    for item in args.expect:
        key, value = item.split("=", 1)
        try:
            expected[key] = json.loads(value)
        except json.JSONDecodeError:
            expected[key] = value
    import pyarrow.parquet as pq
    count = pq.read_metadata(expected["test_data"]).num_rows
    limit = expected["max_samples"]
    population = min(count, limit) if limit > 0 else count
    population *= expected.get("n_samples", 1)
    try:
        summary = json.loads(args.summary.read_text())
    except (OSError, json.JSONDecodeError):
        return 1
    status, reason = cache_status(summary, expected, population)
    if status == 2:
        print(f"Incompatible cached eval {args.summary}: {reason}. Choose a fresh --out_label.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
