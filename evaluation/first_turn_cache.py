#!/usr/bin/env python3
"""
first_turn_cache.py — Pre-generate fixed first-turn user messages (q0) for evaluation.

Generates q0 for every sample in the test data using the user simulator,
and saves them to a JSON file. This file is then loaded by eval_math.py
via --q0_cache_path so that every checkpoint evaluation uses identical
opening messages for fair comparison.

Usage:
    # Using local vLLM user sim server(s):
    python first_turn_cache.py \
        --test_data data/math-l3/rl_validation.parquet \
        --prompt_dir prompts \
        --user_sim_api_base http://127.0.0.1:8101/v1,http://127.0.0.1:8102/v1 \
        --output data/math-l3/q0_cache.json

    # Using Together API:
    python first_turn_cache.py \
        --test_data data/math-l3/rl_validation.parquet \
        --prompt_dir prompts \
        --output data/math-l3/q0_cache.json

Only needs to be run once per test dataset. The output JSON maps
sample index -> q0 message string.
"""

import argparse
import asyncio
import json
import os
import sys
import time

# Add parent directory to path so we can import from eval_math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_math import (
    SYSTEM_PROMPT,
    UserSimulator,
    load_test_data,
)


async def generate_q0_for_sample(
    sample: dict,
    user_sim: UserSimulator,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    """Generate q0 for a single sample. Returns (sample_key, q0_text)."""
    async with semaphore:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        q0, should_terminate, sim_info = await user_sim.generate_user_turn(
            messages=messages,
            problem=sample["single_turn_prompt"],
            ground_truth=sample["ground_truth"],
            expertise=sample["expertise"],
            goal=sample["goal"],
        )
        return str(sample["index"]), q0


async def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate fixed q0 (first-turn user messages) for evaluation"
    )
    parser.add_argument("--test_data", required=True,
                        help="Path to test parquet file")
    parser.add_argument("--prompt_dir", required=True,
                        help="Directory containing the user-simulator prompt templates")
    parser.add_argument("--output", required=True,
                        help="Output path for q0 cache JSON (e.g. data/math-l3/q0_cache.json)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of concurrent generation workers")

    # User simulator config
    parser.add_argument("--user_sim_api_base", default=None,
                        help="Local vLLM server URL(s), comma-separated")
    parser.add_argument("--user_sim_api_key", default=None)
    parser.add_argument("--user_sim_novice_model", default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--user_sim_expert_model", default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--user_sim_temperature", type=float, default=0.7)
    parser.add_argument("--user_sim_max_tokens", type=int, default=1024)

    args = parser.parse_args()

    # Load test data
    samples = load_test_data(args.test_data, max_samples=args.max_samples, seed=args.seed)
    if not samples:
        print("[ERROR] No samples loaded.")
        sys.exit(1)

    # Check if output already exists
    if os.path.isfile(args.output):
        with open(args.output, "r") as f:
            existing = json.load(f)
        print(f"[INFO] Found existing q0 cache with {len(existing)} entries at {args.output}")
        # Check if all samples are already cached
        missing = [s for s in samples if str(s["index"]) not in existing]
        if not missing:
            print("[INFO] All samples already cached. Nothing to do.")
            return
        print(f"[INFO] {len(missing)} samples missing from cache. Generating those.")
        samples = missing
        q0_cache = existing
    else:
        q0_cache = {}

    # Initialize user simulator
    user_sim = UserSimulator(
        prompt_dir=args.prompt_dir,
        api_base=args.user_sim_api_base,
        api_key=args.user_sim_api_key,
        novice_model=args.user_sim_novice_model,
        expert_model=args.user_sim_expert_model,
        max_tokens=args.user_sim_max_tokens,
        temperature=args.user_sim_temperature,
    )

    # Generate q0 for all samples concurrently
    print(f"[INFO] Generating q0 for {len(samples)} samples with {args.num_workers} workers...")
    start = time.time()

    semaphore = asyncio.Semaphore(args.num_workers)
    tasks = [
        generate_q0_for_sample(sample, user_sim, semaphore)
        for sample in samples
    ]

    results = await asyncio.gather(*tasks)

    # Collect results
    failed = 0
    for key, q0 in results:
        if q0:
            q0_cache[key] = q0
        else:
            failed += 1
            print(f"[WARN] Failed to generate q0 for sample {key}")

    elapsed = time.time() - start
    print(f"[INFO] Generated {len(results) - failed}/{len(results)} q0 messages in {elapsed:.1f}s")
    if failed:
        print(f"[WARN] {failed} samples failed q0 generation")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(q0_cache, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(q0_cache)} q0 messages to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
