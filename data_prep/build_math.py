#!/usr/bin/env python3
"""
CToM math data builder

Converts the MATH dataset (or similar) into the CToM training format
for personalized multi-turn RL training.

Supported source formats:
  - "math": Raw MATH dataset (hendrycks/competition_math)
            Columns: problem, solution, level, type
  - "ctom": Pre-processed CToM format (already has prompt, single_turn_prompt, etc.)

Usage — from raw MATH Level 5:

    python recipe/data_prep/build_math.py \
        --dataset hendrycks/competition_math \
        --source_format math \
        --math_level 5 \
        --local_dir $HOME/data/math-l5-expert \
        --persona_mode random

Usage — all Level 5 problems × all 4 personas:

    python recipe/data_prep/build_math.py \
        --dataset hendrycks/competition_math \
        --source_format math \
        --math_level 5 \
        --local_dir $HOME/data/math-l5-expert-all \
        --persona_mode all
"""

import argparse
import json
import os
import random
import re
import uuid
from typing import Any, Optional

from datasets import Dataset, concatenate_datasets, load_dataset

# ──────────────────────────────────────────────────────────────────────
# System prompt: general instruction only, no user persona information.
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """/no_think
You are a helpful, proactive, and highly interactive assistant engaged in a multi-turn conversation.

You MUST respond in the following JSON format for EVERY response:
{"answer": "<your response to the user>", "prediction": "<your prediction of the user's next message>"}

The "answer" field contains your actual response to the user's question or request.
The "prediction" field contains your prediction of what the user will say next, reflecting your understanding of their current state and what they are likely to ask or say next.

Guidelines:
- Maintain context and continuity across turns.
- Keep interactions concise and efficient.
- Always output valid JSON with exactly these two keys: "answer" and "prediction".
- Do NOT include any text outside the JSON object."""

# ──────────────────────────────────────────────────────────────────────
# Tag-format system prompt (2026-07-20). Supersedes SYSTEM_PROMPT above for
# new runs; the JSON one is kept because existing parquets and the 766450 SFT
# checkpoint are baked against it.
#
# Two fixes over the JSON prompt, both traced to the 0.345-vs-0.660 diagnosis
# of the same SFT ckpt (see the eval 833024 / 846014 comparison):
#   1. LaTeX backslashes (\frac, \div, \times) are invalid JSON escapes, so
#      37% of assistant turns failed to parse. Turns that DID parse were the
#      degenerate ones (median answer 66 chars, bare values like "4") because
#      writing no LaTeX is the easiest way to emit valid JSON. Delimiters take
#      the math out of a string literal entirely, so that trade-off disappears.
#   2. The JSON prompt never asked for \boxed{...}, and the strict verifier
#      (ctom/task_math.py) credits ONLY a boxed answer — so ~31% of
#      conversations were unscoreable regardless of correctness.
#
# Delimiters match ctom/prediction_loss.py's sft_v3 tags so the prediction-loss
# path and the emitted format share one vocabulary.
# ──────────────────────────────────────────────────────────────────────
TAG_SYSTEM_PROMPT = """/no_think
You are a helpful, proactive, and highly interactive assistant engaged in a multi-turn conversation.

You MUST structure EVERY response with these four delimiters, in this exact order:
[ANSWER_START]
<your response to the user>
[ANSWER_END]
[PREDICTION_START]
<your prediction of the user's next message>
[PREDICTION_END]

The ANSWER section contains your actual response to the user's question or request. Write it as
normal text — use LaTeX, math, and newlines freely; nothing needs to be escaped.
The PREDICTION section contains your prediction of what the USER will say next — write it from the
user's perspective, in first person, as something a real user would actually say. It must sound
like a USER message (curious, asking a follow-up, expressing confusion), NOT an assistant message.

Guidelines:
- Maintain context and continuity across turns.
- Keep interactions concise and efficient.
- Show your reasoning in the ANSWER section before stating the result.
- When you give a final answer to a math problem, present it inside \\boxed{...}.
- Emit the four delimiters exactly as written above, each on its own line.
- Do NOT include any text outside the delimited sections."""

# ──────────────────────────────────────────────────────────────────────
# Predict-then-revise system prompt (sft_v4, 2026-08-05).
#
# sft_v3's ordering — answer, then prediction — means the prediction is
# generated after the text it would need to improve, so it cannot inform it.
# Measured on 872522: the aux loss and the NLL reward both read a snapshot taken
# BEFORE the prediction is sampled, so the sampled prediction text reaches no
# training objective at all; its only channel is the next turn's context.
#
# This prompt reorders the turn into draft -> prediction -> revision. The
# prediction now sits upstream of the revision, which is the only part the user
# sees and the only part that gets scored, so anticipating the user is on the
# causal path to the reward instead of beside it.
#
# The delimiters are emitted by the controller (agent_loop
# _generate_sft_v4_turn), not by the model — this prompt exists to tell the model
# what each phase is FOR. The \\boxed{...} instruction is deliberately attached
# to the REVISION: ctom/task_math.py credits only the final boxed answer,
# and the revision is what reaches the verifier.
#
# 2026-08-11 — the REVISION paragraph was rewritten because the old wording
# ("do not refer to the draft or to your prediction") forbade only EXPLICIT
# reference and left the implicit framing open: revisions opened with
# "You're right to question the assumption that the side is horizontal. Let me
# clarify:" — answering a turn the user never took, since they never saw the
# draft or the prediction. Measured on step-362 logs, ~5% of FIRST-turn revisions
# did this (unambiguous there: the user has said nothing but the problem).
# Present in the raw base controller too (5.0-5.5%), so it is a prompt defect
# rather than something RL taught; v4 trained its way out of it (5.5% -> 0.5%)
# but v5 did not (5.0% -> 5.0%). The fix names the failure concretely instead of
# stating the principle abstractly, since the abstract form demonstrably did not
# bind. Datasets rendered before this carry the OLD text — see the
# .revise_prompt vs .revise_prompt_v2 suffixes.
# ──────────────────────────────────────────────────────────────────────
REVISE_SYSTEM_PROMPT = """/no_think
You are a helpful, proactive, and highly interactive assistant engaged in a multi-turn conversation.

You MUST structure EVERY response with these six delimiters, in this exact order:
[ANSWER_START]
<your first attempt at responding to the user>
[ANSWER_END]
[PREDICTION_START]
<your prediction of how the user will react to that first attempt>
[PREDICTION_END]
[REVISION_START]
<your final response, improved in light of that prediction>
[REVISION_END]

The ANSWER section is a working draft. The user never sees it. Write it as normal text — use
LaTeX, math, and newlines freely; nothing needs to be escaped.
The PREDICTION section is your prediction of what the USER will say in response to the draft —
write it from the user's perspective, in first person, as something a real user would actually
say. It must sound like a USER message (curious, asking a follow-up, expressing confusion, or
challenging a step), NOT an assistant message. Be specific about WHICH part of the draft they
would react to.
The REVISION section is the message the user actually receives. It is the ONLY part they ever
see: they did not see your draft, and they have NOT said the reaction you predicted. Write it as
a direct reply to the user's LAST ACTUAL MESSAGE, pre-empting the predicted reaction silently —
answer the confusion before it is raised, justify the step before it is challenged, include what
they would have asked for next.
Never write as though the user already objected. Do not open with "You're right to question...",
"Good point", "Good question", "As you noted", "Let me clarify", "Thanks for catching that", or
any phrase that presupposes something the user has not said — on the first turn they have said
nothing but the problem itself. Give the improved explanation directly, as if it had been your
answer all along.

Guidelines:
- Maintain context and continuity across turns.
- Keep interactions concise and efficient.
- Show your reasoning in the draft before stating the result.
- When you give a final answer to a math problem, present it inside \\boxed{...} in the REVISION.
- Emit the six delimiters exactly as written above, each on its own line.
- Do NOT include any text outside the delimited sections."""

# BDR (belief / decide / revise), prediction_mode=sft_v6 — the retired BDR predecessor.
# The controller writes every bracketed tag; the policy writes only phase content.
# The DRAFT is always the complete commit candidate (never a question), the
# BELIEF is an unscored verdict distribution, and the DECISION is one word.
BDR_SYSTEM_PROMPT = """/no_think
You are a helpful assistant solving a math problem together with a human over several turns. The human can judge whether your work is right but will not solve the problem for you.

Every one of your turns has these phases, in this order. The system writes each bracketed tag; you write only the content that follows it.
[DRAFT_START]
<your complete current solution: show the reasoning, then the final answer inside \\boxed{...}>
[DRAFT_END]
[BELIEF_START]
<two or three lines predicting how the human will react to this draft, one per line, each "p=<probability> <LABEL> <short reason>". Labels: CONFIRM (accepts it), DOUBT (senses an error but cannot say where), ERROR step k (points at a specific step), WRONG_ANSWER (says the final answer is wrong), ASKS_BOXED (asks for the boxed form), UNCLEAR (cannot judge). Probabilities sum to 1.>
[BELIEF_END]
[DECISION] <exactly one word: COMMIT, ASK, or REVISE>
[ASK_START]  (only after ASK)
<one concise question for the human, then a final line "Assumption: <the belief you are betting on, as a checkable claim>. Reply ok or correct it.">
[ASK_END]

How to decide:
- REVISE when your belief names a problem you can fix yourself. You get a fresh turn to write a better draft; the human sees nothing.
- ASK when your belief names something only the human can settle. The human sees only your question and assumption.
- COMMIT when the draft is ready to be your answer. The human sees the draft exactly as written.
The draft is always your best complete solution, never a question. The human never sees the draft or the belief unless you COMMIT.

Guidelines:
- Reason in the draft before stating the result; always box the final answer.
- Be concise. Never re-ask what was answered. Never presuppose a reply the human has not given.
- Write only the content of each phase. Do not repeat the tags. Output nothing outside the phases."""

# sft_v7 (decide / draft-predict-revise / ask-with-assumption) — the paper's method section.
V7_SYSTEM_PROMPT = """/no_think
You are a helpful assistant solving a math problem together with a human over several turns. The human can judge whether your work is right but will not solve the problem for you.

Every one of your turns begins with a short plan (two or three sentences, never a full solution) and a decision. The system writes each bracketed tag; you write only the content after it.
[PLAN_START]
<two or three sentences (no more): what is still uncertain, whether the human could settle it, and what to do now>
[PLAN_END]
[DECISION] <exactly one word: DRAFT, ASK, or COMMIT>

After DRAFT the system asks for
[DRAFT_START]
<your complete solution: the reasoning, then the final answer inside \\boxed{...}>
[DRAFT_END]
and then opens a user turn whose first line is [ANTICIPATED]. In that turn YOU write what this human would most likely say in reply to the draft: in the human's own voice, first person, one to three sentences, then the line [PREDICTION_END]. If your anticipated reply points at a problem, you get a fresh turn to write an improved draft and the human sees nothing yet; if it accepts the draft, the draft is sent to the human as your answer.

After ASK the system asks for
[ASK_START]
<one concise question for the human>
[ASK_END]
and then the same [ANTICIPATED] user turn, where you write the answer you expect. The human sees your question together with that expected answer and replies "ok" or corrects it.

COMMIT ends the conversation with the last draft the human saw. Use it only after a draft has been shown.

Guidelines:
- A user turn that starts with [ANTICIPATED] is your own forecast, not the human. The human's real messages never carry that line.
- Reason inside the draft before stating the result; always box the final answer.
- Be concise. Never re-ask what was answered. Never presuppose a reply the human has not given.
- Write only the content of each phase. Do not repeat the tags. Output nothing outside the phases."""

# All persona combinations (passed to user simulator only, NOT the assistant)
PERSONAS = [
    {"expertise": "expert", "goal": "solving"},
    {"expertise": "novice", "goal": "solving"},
]


def extract_boxed_answer(solution: str) -> str:
    """Extract the final answer from \\boxed{...} in a MATH solution."""
    match = re.search(r"\\boxed\{", solution)
    if not match:
        return solution.strip()
    start = match.end()
    depth = 1
    pos = start
    while pos < len(solution) and depth > 0:
        if solution[pos] == "{":
            depth += 1
        elif solution[pos] == "}":
            depth -= 1
        pos += 1
    if depth == 0:
        return solution[start: pos - 1].strip()
    return solution.strip()


# ──────────────────────────────────────────────────────────────────────
# Source format converters
# ──────────────────────────────────────────────────────────────────────

def convert_math_to_ctom_format(example: dict) -> dict:
    """
    Convert a raw MATH dataset example to intermediate CToM format.

    Input columns (hendrycks/competition_math):
        - problem: str        (the math problem text)
        - solution: str       (step-by-step solution with \\boxed{answer})
        - level: str          (e.g., "Level 5")
        - type: str           (e.g., "Algebra", "Number Theory")

    Output columns (intermediate format for collapse_example):
        - prompt: list[dict]  (empty — q0 will be generated by user simulator)
        - single_turn_prompt: str  (the problem text — given to user simulator)
        - ground_truth: str   (the boxed answer extracted from solution)
        - solution: str       (full solution for reference)
        - subject: str        (math subject)
        - difficulty: str     (level string)
    """
    ground_truth = extract_boxed_answer(example.get("solution", ""))

    return {
        "prompt": [],  # empty — will be prepended with system prompt in collapse_example
        "single_turn_prompt": example["problem"],
        "ground_truth": ground_truth,
        "solution": example.get("solution", ""),
        "subject": example.get("type", ""),
        "difficulty": example.get("level", ""),
    }


def assign_persona(example: dict, seed: int = 42, personas: list = PERSONAS) -> dict:
    """Assign a random persona deterministically based on example identity."""
    key = example.get("conv_id", example.get("single_turn_prompt", str(uuid.uuid4())))
    rng = random.Random(f"{seed}_{key}")
    persona = rng.choice(personas)
    example["expertise"] = persona["expertise"]
    example["goal"] = persona["goal"]
    return example


def collapse_example(example: dict[str, Any]) -> dict[str, Any]:
    """
    Collapse an intermediate-format example into the final CToM training format.

    The prompt for RL only contains [system]. The initial user question q0
    is generated live by the persona-aware user simulator at rollout time.
    """
    ground_truth = example.get("ground_truth", "")
    expertise = example.get("expertise", "novice")
    goal = example.get("goal", "solving")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    single_turn_prompt = example.get("single_turn_prompt", "")
    assert single_turn_prompt, "Missing 'single_turn_prompt' — the problem text."

    extra_info = {
        "interaction_kwargs": {
            "name": "ctom",
            "single_turn_prompt": single_turn_prompt,
            "task_desc": example.get("task_desc", "solving math problems"),
            "expertise": expertise,
            "goal": goal,
            "ground_truth": ground_truth,
        },
    }
    # Preserve optional metadata
    for key in ("solution", "subject", "difficulty"):
        if key in example:
            extra_info[key] = example[key]

    return {
        "prompt": messages,
        "ground_truth": ground_truth,
        "raw_prompt": messages,
        "extra_info": extra_info,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "data_source": "ctom",
        "agent_name": "ctom_agent",
        "index": str(uuid.uuid4()),
    }


# ---------- IO helpers ----------
def save_parquet(ds_split: Dataset, filename: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{filename}.parquet")
    ds_split.to_parquet(path)
    print(f"[OK] Wrote {filename}.parquet -> {path} ({len(ds_split)} rows)")


def maybe_copy_to_hdfs(local_dir: str, hdfs_dir: Optional[str]) -> None:
    if not hdfs_dir:
        return
    try:
        from verl.utils.hdfs_io import copy, makedirs
    except Exception as e:
        print(f"[WARN] Skipping HDFS copy: {e}")
        return
    makedirs(hdfs_dir)
    copy(src=local_dir, dst=hdfs_dir)
    print(f"[OK] Copied {local_dir} -> {hdfs_dir}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="CToM math data builder")
    ap.add_argument(
        "--dataset", default="qwedsacf/competition_math",
        help="HF dataset path or local dir/file.",
    )
    ap.add_argument(
        "--source_format", default="math", choices=["math", "ctom"],
        help="Format of the source dataset. "
             "'math': raw MATH dataset (problem/solution/level/type). "
             "'ctom': pre-processed with prompt/single_turn_prompt columns.",
    )
    ap.add_argument(
        "--math_level", type=int, default=5,
        help="Filter MATH problems to this difficulty level (1-5). "
             "Only used when --source_format=math. Set to 0 to use all levels.",
    )

    ap.add_argument(
        "--split", default="train",
        help="Which split to load from the source dataset (e.g., 'train', 'test').",
    )
    ap.add_argument("--task_desc", default="solving math problems", help="Task description.")
    ap.add_argument("--local_dir", default="data/math-l3", help="Output directory.")
    ap.add_argument("--hdfs_dir", default=None, help="Optional HDFS destination.")
    ap.add_argument("--validation_size", type=float, default=0.1, help="Validation split fraction.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--num_proc", type=int, default=1, help="Parallel workers.")
    ap.add_argument(
        "--persona_mode", default="random", choices=["random", "all"],
        help="'random': assign one random persona per example. "
             "'all': duplicate each example 4x with all personas.",
    )
    ap.add_argument(
        "--goals", default="solving",
        help="Persona goals to keep. The paper uses solving only. "
             "The math-l5 RL data was built with only the 'solving' personas, so "
             "pass --goals solving to generate level-matched variants.",
    )
    ap.add_argument(
        "--expertises", default="expert",
        help="Comma-separated persona expertise levels to keep (subset of "
             "expert,novice). Defaults to expert-only; pass --expertises expert,novice "
             "only for an explicitly requested mixed-persona case study.",
    )
    args = ap.parse_args()

    goals = {g.strip() for g in args.goals.split(",") if g.strip()}
    expertises = {e.strip() for e in args.expertises.split(",") if e.strip()}
    personas = [
        p for p in PERSONAS
        if p["goal"] in goals and p["expertise"] in expertises
    ]
    if not personas:
        ap.error(f"--goals '{args.goals}' / --expertises '{args.expertises}' match no persona")

    out_dir = os.path.expanduser(args.local_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load dataset ──
    print(f"[INFO] Loading dataset: {args.dataset} (split={args.split})")
    ds_all = load_dataset(args.dataset, split=args.split)
    print(f"[INFO] Loaded {len(ds_all)} examples")

    # ── Step 2: Convert source format ──
    if args.source_format == "math":
        # Filter by difficulty level
        if args.math_level > 0:
            level_str = f"Level {args.math_level}"
            ds_all = ds_all.filter(
                lambda x: x.get("level", "") == level_str,
                num_proc=args.num_proc,
            )
            print(f"[INFO] Filtered to {len(ds_all)} Level {args.math_level} problems")

        # Convert MATH format -> intermediate format
        print("[INFO] Converting MATH format to intermediate format...")
        ds_all = ds_all.map(
            convert_math_to_ctom_format,
            remove_columns=ds_all.column_names,
            num_proc=args.num_proc,
        )
    elif args.source_format == "ctom":
        # Already in the right format, just pass through
        pass

    # ── Step 3: Add task_desc ──
    ds_all = ds_all.map(lambda x: {"task_desc": args.task_desc}, num_proc=args.num_proc)

    # ── Step 4: Assign personas ──
    if args.persona_mode == "all":
        all_rows = []
        for ex in ds_all:
            for persona in personas:
                row = {**ex, **persona}
                all_rows.append(row)
        ds_all = Dataset.from_list(all_rows)
        print(f"[INFO] Expanded to {len(ds_all)} rows ({len(personas)} personas per example)")
    else:
        ds_all = ds_all.map(
            lambda x: assign_persona(x, seed=args.seed, personas=personas),
            num_proc=args.num_proc,
        )

    # ── Step 5: Collapse to final CToM format ──
    print(f"[INFO] Collapsing to CToM format on {len(ds_all)} rows...")
    ds_all = ds_all.map(
        function=collapse_example,
        remove_columns=ds_all.column_names,
        num_proc=args.num_proc,
    )

    # Deduplicate by prompt + extra_info (same problem + same persona = duplicate)
    def dedup(dataset):
        seen = set()
        unique_rows = []
        for ex in dataset:
            key = json.dumps(
                ex["extra_info"]["interaction_kwargs"],
                sort_keys=True,
                ensure_ascii=False,
            )
            if key not in seen:
                seen.add(key)
                unique_rows.append(ex)
        return Dataset.from_list(unique_rows)

    ds_all = dedup(ds_all)
    print(f"[INFO] {len(ds_all)} unique examples after deduplication")

    # ── Step 6: Split and save ──
    print(f"[INFO] Splitting with validation_size={args.validation_size}, seed={args.seed}")
    split = ds_all.train_test_split(test_size=args.validation_size, seed=args.seed, shuffle=True)
    train_ds, val_ds = split["train"], split["test"]
    print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}")

    save_parquet(train_ds, "rl_train", out_dir)
    save_parquet(val_ds, "rl_test", out_dir)

    maybe_copy_to_hdfs(local_dir=out_dir, hdfs_dir=args.hdfs_dir)
    print(f"[DONE] rl_train.parquet and rl_validation.parquet written to {out_dir}")


if __name__ == "__main__":
    main()
