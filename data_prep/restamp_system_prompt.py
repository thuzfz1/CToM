#!/usr/bin/env python3
"""Re-stamp the assistant system prompt in an existing RL split.

Why re-stamp instead of re-running build_math.py: regenerating from the raw
MATH source would re-draw the train/validation split and re-render every field,
so any difference between the resulting run and its predecessors would be partly
a data-pipeline difference. This rewrites ONE string — the system message in
`prompt[0]` and `raw_prompt[0]` — and copies every other column through
untouched, so split membership, indices, ground truths and interaction_kwargs
stay byte-identical. Same argument as restamp_system_prompt.py.

The source split must carry a single uniform system prompt (verified here), and
by default must match the prompt the caller says it currently has, so a split
cannot be double-stamped or stamped from the wrong baseline.

Usage:
    python3 data_prep/restamp_system_prompt.py \
        --in_dir data/math-l5 --out_dir data/math-l5 \
        --in_suffix .revise_prompt --out_suffix .revise_prompt_v2 \
        --prompt revise
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLITS = ("rl_train", "rl_validation")


# Prompt constants live beside the renderer for each dataset family: math splits
# come from build_math, writing splits from build_writing, coding splits
# from build_coding. They are NOT interchangeable -- the math revise
# prompt tells the model to \boxed{} its answer, which is meaningless for a
# Python function, and the coding one carries the "I WANT TO ANSWER:" submission
# marker that the code reward path looks for. Selecting the wrong source stamps a
# syntactically valid but semantically wrong contract, so --source is required
# rather than guessed.
PROMPT_SOURCES = {
    "math": "data_prep.build_math",
    "medium": "data_prep.build_writing",
    "bigcodebench": "data_prep.build_coding",
}


def _known_prompts(source: str = "math"):
    import importlib

    mod = importlib.import_module(PROMPT_SOURCES[source])

    out = {}
    for name, attr in (("v7", "V7_SYSTEM_PROMPT"),
                       ("bdr", "BDR_SYSTEM_PROMPT"),
                       ("revise", "REVISE_SYSTEM_PROMPT"),
                       ("tag", "TAG_SYSTEM_PROMPT"),
                       ("default", "SYSTEM_PROMPT")):
        if hasattr(mod, attr):
            out[name] = getattr(mod, attr)
    return out


def _system_of(msgs):
    if msgs is None or len(msgs) == 0:
        return None
    first = msgs[0]
    if first.get("role") != "system":
        return None
    return first.get("content")


def _restamp(msgs, new_prompt):
    """Return a copy of the message array with the system content replaced."""
    out = []
    for i, m in enumerate(msgs):
        m = dict(m)
        if i == 0 and m.get("role") == "system":
            m["content"] = new_prompt
        out.append(m)
    return np.array(out, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--in_suffix", required=True)
    ap.add_argument("--out_suffix", required=True)
    ap.add_argument("--prompt", required=True,
                    help="which system prompt to stamp IN (v7 | bdr | revise | tag | default)")
    ap.add_argument("--source", default="math", choices=sorted(PROMPT_SOURCES),
                    help="which renderer the prompt constants come from: "
                         "math=build_math, medium=build_writing, "
                         "bigcodebench=build_coding")
    ap.add_argument("--allow_same", action="store_true",
                    help="proceed even if the split already carries the target prompt")
    args = ap.parse_args()

    known = _known_prompts(args.source)
    if args.prompt not in known:
        sys.exit(f"ERROR: --prompt must be one of {sorted(known)} for --source {args.source}")
    new_prompt = known[args.prompt]

    if args.in_suffix == args.out_suffix and args.in_dir == args.out_dir:
        sys.exit("ERROR: refusing to overwrite the source split in place")

    os.makedirs(args.out_dir, exist_ok=True)
    for split in SPLITS:
        src = os.path.join(args.in_dir, f"{split}{args.in_suffix}.parquet")
        dst = os.path.join(args.out_dir, f"{split}{args.out_suffix}.parquet")
        if not os.path.isfile(src):
            sys.exit(f"ERROR: missing {src}")
        df = pd.read_parquet(src)

        for col in ("prompt", "raw_prompt"):
            if col not in df.columns:
                sys.exit(f"ERROR: {src} has no '{col}' column")
            sys_col = df[col].map(_system_of)
            if sys_col.isna().any():
                sys.exit(f"ERROR: {src}:{col} has rows whose first message is not a system message")
            if sys_col.nunique() != 1:
                sys.exit(f"ERROR: {src}:{col} carries {sys_col.nunique()} distinct system prompts; "
                         "re-stamping assumes a uniform split")
            if sys_col.iloc[0] == new_prompt and not args.allow_same:
                sys.exit(f"ERROR: {src}:{col} already carries the '{args.prompt}' prompt "
                         "(pass --allow_same to force)")

        before = _system_of(df["prompt"].iloc[0])
        for col in ("prompt", "raw_prompt"):
            df[col] = df[col].map(lambda v: _restamp(v, new_prompt))

        df.to_parquet(dst, index=False)

        # verify what landed on disk, not what we think we wrote
        chk = pd.read_parquet(dst)
        assert len(chk) == len(df), "row count changed"
        for col in ("prompt", "raw_prompt"):
            got = chk[col].map(_system_of)
            assert got.nunique() == 1 and got.iloc[0] == new_prompt, f"{col} not stamped"
        assert list(chk["index"]) == list(df["index"]), "row order/identity changed"
        print(f"{os.path.basename(src)} -> {os.path.basename(dst)}  "
              f"rows={len(chk)}  system prompt {len(before)} -> {len(new_prompt)} chars")

    print("\nStamped prompt (first 3 lines):")
    for line in new_prompt.splitlines()[:3]:
        print("   ", line)


if __name__ == "__main__":
    main()
