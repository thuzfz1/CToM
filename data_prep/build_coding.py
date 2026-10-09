#!/usr/bin/env python3
"""CollabLLM BigCodeBench-Chat dataset processor.

Converts the CollabLLM multi-turn BigCodeBench data into the CToM veRL parquet
format so the existing multi-turn RL pipeline (CToMUserSimulator + CToMAgentLoop
+ CToMRewardManager) can train on collaborative *coding*.

Source rows (`collabllm/collabllm-multiturn-bigcodebench[-large]`, one HF `train`
split) are per-turn samples of simulated conversations, several rows per task.
The columns used here:

    prompt[0].content       the underspecified opening request ("Can you help me
                            write a function that counts how many times each
                            integer appears...") -- replayed verbatim as q0
    single_turn_prompt      the full BigCodeBench task description, including the
                            required signature block: what the human knows and
                            the agent does not
    single_turn_completion  JSON {answer, dataset, entry_point, task_id, test}:
                            `test` is the unittest suite that defines R_task,
                            `answer` the reference solution
    single_turn_metadata    {entry_point, extraction_requirement, task_id, test}

One task per BigCodeBench task_id (rows are collapsed on conv_id, 1:1 with it).

This is build_writing.py's structure in a third domain, so it reuses that
machinery rather than adding a parallel one: q0 is replayed rather than invented,
the human holds hidden information and reveals the minimum asked for, and the
episode ends when the agent submits behind "I WANT TO ANSWER:".

── What is public and what is hidden (2026-09-11) ────────────────────────────
The opening request carries the underspecified ask PLUS the required calling
signature -- the import block and `def task_func(...)` line BigCodeBench fixes
for the task. Everything that says what the function must DO stays hidden with
the human: the behaviour, the return contract, the edge cases, the error
handling. Measured over the 500 training tasks, the published prefix is 102
characters at the median and the behavioural spec left hidden is 66 words at the
median and never under 20, so the collaboration still has real content to
recover.

Why publish the signature. It is not a behavioural requirement, and hiding it
made the score turn partly on whether the policy happened to guess a fixed string
-- `task_func` is identical across all 1140 BigCodeBench tasks. With it public,
STRICT scoring is the honest measurement: a candidate that defines another name
ignored an instruction it was given. This also matches BigCodeBench's own
single-turn protocol, where the prefix is always part of the prompt.

Note what the prefix carries besides the name: the task's import list, and for
101 of 500 tasks (20%) a module-level constant as well. That is a real hint about
the implementation, and it is the price of strict scoring. It is the same hint
every BigCodeBench evaluation gives.

── R_task ────────────────────────────────────────────────────────────────────
CollabLLM's metric here is "runnable code->pass_rate": extract the final code
from the transcript, run the task's unittest suite, score the fraction of test
methods that pass. We keep the metric and replace the extraction: theirs is an
LLM pass steered by `extraction_requirement`, ours is the submission marker and
strict entry-point matching.

── Why --runnable is REQUIRED ────────────────────────────────────────────────
A task whose OWN reference solution cannot pass in the execution sandbox scores
0 for every rollout forever: its GRPO group is degenerate (identical returns ->
zero advantage -> no gradient) and it applies a fixed downward bias to reported
R_task that has nothing to do with the policy. data_prep/validate_coding_tasks.py
produces the keep-list; 594 of 619 tasks survived on 2026-09-11, and only those
594 are built. --allow_unvalidated exists for debugging and prints a loud
warning; no run should ever use it.

Usage:
    python data_prep/build_coding.py \
        --large_file $SCRATCH/data/collabllm-bigcodebench-large/data/train-00000-of-00001.parquet \
        --small_file $SCRATCH/data/collabllm-bigcodebench/data/train-00000-of-00001.parquet \
        --runnable   data/bigcodebench_runnable.json \
        --local_dir  $PWD/data/collabllm-bigcodebench

Download first with:
    huggingface-cli download collabllm/collabllm-multiturn-bigcodebench-large \
        --repo-type dataset --local-dir $SCRATCH/data/collabllm-bigcodebench-large
    huggingface-cli download collabllm/collabllm-multiturn-bigcodebench \
        --repo-type dataset --local-dir $SCRATCH/data/collabllm-bigcodebench
"""

import argparse
import json
import os
import random
import re
import sys
import uuid
from typing import Any

import pandas as pd
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── System prompts ────────────────────────────────────────────────────────────
# Five renderings of the same contract, one per output format the pipeline
# supports, mirroring build_writing.py. They must agree with the format the
# launcher resolves (ray_trainer compares the parquet system prompt against it).
#
# Every one of them states the entry-point name. That is NOT a leak of the hidden
# specification: `task_func` is the same in all 1140 BigCodeBench tasks, so it
# carries no information about THIS task, while withholding it would make the
# score depend on whether the policy happened to guess a fixed string. CollabLLM
# hands the same fact to their scorer instead, through `extraction_requirement`.

_CONTRACT = """The human's opening request gives you the exact code your answer must start with — the imports and the `def task_func(...)` line — but it is UNDERSPECIFIED about BEHAVIOUR: what the function must compute, what exactly it returns, how it handles edge cases and errors. Those are known only to the human. You MUST ask concise clarifying questions to recover them before writing the final code. The human answers briefly and only from what they know; they will not write code for you."""

_SUBMIT_RULES = """- Your code MUST define `task_func` with exactly the signature given in the opening request, and must begin with the code block shown there. A different name or signature scores zero however correct the logic is.
- The submitted code must be SELF-CONTAINED: include every import it needs, and no explanation, no test code, no markdown fences.
- You have at most 14 back-and-forth rounds; be efficient — ask only what you need."""

SYSTEM_PROMPT = f"""/no_think
You are a helpful, proactive assistant collaborating with a human on a Python programming task over multiple turns.

{_CONTRACT}

You MUST respond in the following JSON format for EVERY response:
{{"answer": "<your message to the human>", "prediction": "<your prediction of the human's next message>"}}

- While gathering information, "answer" is your next question or comment to the human.
- When (and only when) you have gathered enough information, make "answer" begin with the exact marker:
  I WANT TO ANSWER:
  immediately followed by the COMPLETE final Python code.
- "prediction" is always your best guess of what the human will say next.
{_SUBMIT_RULES}
- Always output valid JSON with exactly these two keys: "answer" and "prediction". Output nothing outside the JSON object."""

TAG_SYSTEM_PROMPT = f"""/no_think
You are a helpful, proactive assistant collaborating with a human on a Python programming task over multiple turns.

{_CONTRACT}

You MUST structure EVERY response with these four delimiters, in this exact order:
[ANSWER_START]
<your message to the human>
[ANSWER_END]
[PREDICTION_START]
<your prediction of the human's next message>
[PREDICTION_END]

- While gathering information, the ANSWER section is your next concise question or comment to the human.
- When (and only when) you have gathered enough information, make the ANSWER section begin with the exact marker:
  I WANT TO ANSWER:
  immediately followed by the COMPLETE final Python code.
- The PREDICTION section is always your best guess of what the human will say next. Write it from the human's perspective as a natural user message.
{_SUBMIT_RULES}
- Emit the four delimiters exactly as written above, each on its own line.
- Do NOT output JSON or include any text outside the delimited sections."""

# sft_v4/sft_v5: draft -> prediction -> revision. The argument for this domain is
# the same as for writing and stronger than for math: the human holds the
# requirements the function cannot be written without, so their next message IS
# the missing specification, and anticipating it is anticipating the material.
# The submission marker lives in the REVISION only -- the draft is discarded
# before the human sees anything, so a marker there submits nothing.
#
# Written tight on purpose: this is a fixed prefix re-sent every turn, so every
# token is overhead against data.max_prompt_length. Each rule is load-bearing and
# pinned by tests/test_reward.py.
REVISE_SYSTEM_PROMPT = """/no_think
You write a Python function for a human over multiple turns. Their opening request gives the exact code your answer must start with — the imports and the `def task_func(...)` line — but not what the function must DO: the computation, the exact return value, the edge cases and the error behaviour are known only to them. Ask concise clarifying questions; they answer briefly and only from what they know, and they never write code.

Every response uses exactly these six delimiters, each on its own line, in this order:
[ANSWER_START]
<draft of your next message>
[ANSWER_END]
[PREDICTION_START]
<how the human replies to that draft>
[PREDICTION_END]
[REVISION_START]
<the message the human actually receives>
[REVISION_END]

ANSWER is a private draft. The human never sees it.
PREDICTION is that human's reply, in their own first person: the actual requirement, input, edge case or name they would disclose, never an acknowledgement. They answer only your latest question, in at most two sentences.
REVISION is the only part they see. They did not see the draft and have NOT said the prediction. Spend it: if you already know their answer, ask what you still do not know; if the draft asked something they cannot answer, ask something else; if the anticipated answers are enough, write the code now. Never presuppose their reply -- no "You're right", "Good question", "Thanks for confirming".

To submit, begin the REVISION with the exact marker:
I WANT TO ANSWER:
then the complete self-contained Python code, with every import it needs and no commentary, no test code and no markdown fences. It must begin with the code block from the opening request and define `task_func` with exactly that signature — a different name scores zero however correct the logic is. The marker only counts in the REVISION; in the draft it submits nothing.

At most 14 rounds. Never re-ask what was answered. Output no JSON and nothing outside the delimiters."""

# BDR (belief / decide / revise), prediction_mode=sft_v6 — the retired BDR predecessor.
BDR_SYSTEM_PROMPT = """/no_think
You write a Python function for a human over several turns. Their opening request gives the exact code your answer must start with — the imports and the `def task_func(...)` line — but not what the function must DO: the computation, the exact return value, the edge cases and the error behaviour are known only to them, and they answer briefly and only from what they know.

Every one of your turns has these phases, in this order. The system writes each bracketed tag; you write only the content that follows it.
[DRAFT_START]
<the complete function as it stands now: self-contained Python beginning with the code block from the opening request and defining `task_func` with exactly that signature, every import included, no commentary, no test code, no markdown fences, no marker. Write your best guess of what they want, even before you know the requirements.>
[DRAFT_END]
[BELIEF_START]
<two or three lines predicting how the human would react if they saw this draft, one per line, each "p=<probability> <LABEL> <short reason>". Labels: ACCEPT (it is what they want), FIX <what> (they would name this as wrong or missing), ANSWERS <fact> (they would supply this requirement if asked), DONT_MIND (they would leave it to you), TERMINATE (they would end after submission). Probabilities sum to 1.>
[BELIEF_END]
[DECISION] <exactly one word: COMMIT, SHOW, ASK, or REVISE>
[ASK_START]  (only after ASK)
<one concise question for the human, then a final line "Assumption: <the belief you are betting on, as a checkable claim>. Reply ok or correct it.">
[ASK_END]

How to decide:
- REVISE when your belief names a problem you can fix yourself. You get a fresh turn to write better code; the human sees nothing.
- ASK when your belief names a requirement only the human can supply. The human sees only your question and assumption.
- SHOW when you want the human's reaction to the whole function. They see the code, not the belief, and they will name the single most important thing that is wrong or missing.
- COMMIT to submit the draft as the final code. This ends the task; the system adds the submission marker for you.

Guidelines:
- Be concise in questions; never re-ask what was answered; never presuppose a reply the human has not given.
- The function must run standalone: no undefined names, no placeholder bodies.
- Write only the content of each phase. Do not repeat the tags. Output nothing outside the phases."""

# sft_v7 (decide / draft-predict-revise / ask-with-assumption) — the paper's method section.
V7_SYSTEM_PROMPT = """/no_think
You write a Python function for a human over several turns. Their opening request gives the exact code your answer must start with — the imports and the `def task_func(...)` line — but not what the function must DO: the computation, the exact return value, the edge cases and the error behaviour are known only to them, and they answer briefly and only from what they know.

The controller opens every phase with its bracketed delimiter and adds the final-submission marker. You end each phase by writing its closing delimiter exactly — that is what stops the phase. Write nothing outside a phase.

Every one of your turns begins with a plan and a decision:
[PLAN_START]
<two or three sentences (no more): what you still do not know, whether the human could supply it, and what to do now>
[PLAN_END]
[DECISION] <exactly one word: DRAFT, ASK, or COMMIT>

After DRAFT the controller opens
[DRAFT_START]
<the complete function as it stands now: self-contained Python beginning with the code block from the opening request and defining `task_func` with exactly that signature, every import included, no commentary, no test code, no markdown fences, no marker. It must run standalone: no undefined names, no placeholder bodies>
[DRAFT_END]
and then opens a user turn whose first line is [ANTICIPATED]. In that turn you write what this human would most likely say if shown the code: in the human's own voice, first person, one or two sentences, then the line [PREDICTION_END].

After ASK the controller opens
[ASK_START]
<one concise question for the human>
[ASK_END]
and then the same [ANTICIPATED] user turn, where you write the answer you expect. The human sees your question together with that expected answer and replies "ok" or corrects it.

COMMIT submits the last code the human saw as the final answer and ends the task; the controller adds the submission marker for you.

Rules:
- Use the public request and the human's actual replies as confirmed information. Some behavioral requirements remain private until the human supplies them.
- Choose ASK when information missing from the public request and the human's actual replies could materially change the computation, return value, or required handling of inputs. Ask a focused question that resolves that uncertainty.
- Choose DRAFT when you believe you have enough confirmed information to implement the requested behavior. Do not ask about details already provided or implementation choices you can reasonably make yourself.
- An anticipated human reply is a prediction, not confirmation. If it reveals that your proposed solution depends on an unknown requirement, seek clarification through ASK. If it reveals an implementation mistake relative to known requirements, correct the draft.
- Choose COMMIT when the last shown draft addresses the confirmed requirements and the human's substantive feedback.
- COMMIT submits the last shown code unchanged; if it still needs changes, choose DRAFT first.
- The real human may accept a shown draft or identify a mismatch. Acceptance does not itself end the task.
- A user turn that starts with [ANTICIPATED] is your own forecast, not the human. The human's real messages never carry that line.
- Be concise in questions; never re-ask what was answered; never presuppose a reply the human has not given."""

# The prediction-free baseline (2026-09-17). Same collaboration contract and the
# same submission marker as `default`, with the "prediction" key and every
# mention of it removed: the naive GRPO control must not spend tokens on, or be
# shaped by, a prediction it is never rewarded for. Plain text, not JSON —
# parse_assistant_response falls back to {"answer": <raw text>, "prediction": ""}
# and extract_final_code reads the marker or any runnable code straight off the
# turn, so nothing downstream needs the JSON envelope.
#
# The rounds bullet is dropped on purpose: V7_SYSTEM_PROMPT states no round
# budget, so stating one here would hand the baseline information the v7 arms
# do not get — and _SUBMIT_RULES says 14, which is wrong at the 6-round pin the
# matched control runs under.
_SUBMIT_RULES_NO_ROUNDS = "\n".join(
    line for line in _SUBMIT_RULES.splitlines() if "back-and-forth rounds" not in line
)

PLAIN_SYSTEM_PROMPT = f"""/no_think
You are a helpful, proactive assistant collaborating with a human on a Python programming task over multiple turns.

{_CONTRACT}

- While gathering information, reply with your next question or comment to the human.
- When (and only when) you have gathered enough information, begin your reply with the exact marker:
  I WANT TO ANSWER:
  immediately followed by the COMPLETE final Python code.
{_SUBMIT_RULES_NO_ROUNDS}"""

PROMPTS = {
    "default": SYSTEM_PROMPT,
    "tag": TAG_SYSTEM_PROMPT,
    "revise": REVISE_SYSTEM_PROMPT,
    "bdr": BDR_SYSTEM_PROMPT,
    "v7": V7_SYSTEM_PROMPT,
    "plain": PLAIN_SYSTEM_PROMPT,
}
SUFFIXES = {
    "default": "",
    "tag": ".tag_prompt",
    "revise": ".revise_prompt",
    "bdr": ".bdr_prompt",
    "v7": ".v7_prompt",
    "plain": ".plain_prompt",
}


def _as_messages(value: Any) -> list[dict]:
    """`prompt` arrives as a numpy array of dicts from parquet; normalize."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    out = []
    for m in list(value):
        if isinstance(m, dict):
            out.append({"role": str(m.get("role", "")), "content": str(m.get("content", ""))})
    return out


def _opening_request(row: pd.Series) -> str:
    """The first user message of the sampled conversation — the public q0."""
    msgs = _as_messages(row.get("prompt"))
    if msgs and msgs[0].get("role") == "user":
        return msgs[0]["content"].strip()
    return ""


# BigCodeBench states the required opening code twice, in two wordings. The
# metadata copy is the cleaner of the two (no fences to unwrap) and is present on
# 500/500 training tasks; the prompt copy agrees with it on 499/500 and is the
# fallback.
_META_MARK = "You should start with the following code:"
_PROMPT_MARK = "You should write self-contained code starting with:"
_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def extract_code_prefix(spec: str, metadata: Any) -> str:
    """The required opening code block: imports, module constants, `def task_func(...)`.

    This is the half of the specification that becomes PUBLIC (appended to q0).
    Returns "" when neither source carries it, which makes the task unusable —
    with no published signature, strict scoring would be scoring a guess.
    """
    meta = metadata if isinstance(metadata, dict) else {}
    requirement = str(meta.get("extraction_requirement", "") or "")
    if _META_MARK in requirement:
        block = requirement.split(_META_MARK, 1)[1].strip()
        if block:
            return block
    if _PROMPT_MARK in (spec or ""):
        tail = spec.split(_PROMPT_MARK, 1)[1]
        match = _FENCE_RE.search(tail)
        return (match.group(1) if match else tail).strip()
    return ""


def compose_public_request(q0: str, code_prefix: str) -> str:
    """The opening user message: the vague ask plus the required signature.

    Rendered as a fenced block so the policy can copy it verbatim, and worded as
    a requirement rather than a suggestion — under strict scoring a renamed
    function scores 0 however correct its logic.
    """
    return (
        f"{q0.strip()}\n\n"
        f"Your code must start with exactly this, and define `task_func` with this "
        f"signature:\n```python\n{code_prefix.strip()}\n```"
    )


def strip_code_prefix_from_spec(spec: str) -> str:
    """The spec with the now-public opening-code sentence removed.

    What is left is the BEHAVIOURAL half — the computation, the return contract,
    the edge cases — which is what the human keeps. Leaving the signature in the
    hidden half would only let the simulator spend a turn disclosing something the
    policy was already handed.
    """
    text = (spec or "").strip()
    if _PROMPT_MARK in text:
        text = text.split(_PROMPT_MARK, 1)[0].strip()
    return text


def _completion(row: pd.Series) -> dict:
    raw = row.get("single_turn_completion")
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def collapse_group(rows: pd.DataFrame, prompt_key: str) -> dict[str, Any] | None:
    """Map all rows of one BigCodeBench task into a single CToM parquet row."""
    rows = rows.sort_values("turn_id")
    first = rows.iloc[0]

    q0 = ""
    for _, r in rows.iterrows():          # earliest turn that carries a clean q0
        q0 = _opening_request(r)
        if q0:
            break
    spec = str(first.get("single_turn_prompt") or "").strip()
    payload = _completion(first)
    test_source = str(payload.get("test") or "").strip()
    reference = str(payload.get("answer") or "").strip()
    task_id = str(payload.get("task_id") or "")
    code_prefix = extract_code_prefix(spec, first.get("single_turn_metadata"))
    # No published signature means strict scoring would grade a guess, so the row
    # is unusable rather than merely harder.
    if not q0 or not spec or not test_source or not reference or not code_prefix:
        return None
    hidden = strip_code_prefix_from_spec(spec)
    if not hidden:
        return None
    public_request = compose_public_request(q0, code_prefix)

    messages = [{"role": "system", "content": PROMPTS[prompt_key]}]

    interaction_kwargs = {
        "name": "ctom",
        "task_type": "bigcodebench",
        # NOTE the name collision with the source column: in the CToM schema
        # `single_turn_prompt` is what the simulator replays as q0, so it holds
        # the PUBLIC request — here the vague ask PLUS the required signature.
        # CollabLLM's `single_turn_prompt` is the full spec; its BEHAVIOURAL half
        # is the hidden side and goes to `hidden_information`.
        "single_turn_prompt": public_request,
        "hidden_information": hidden,
        # The published block, kept separately so preflights and the eval can
        # check that what the policy was handed is what the grader requires.
        "code_prefix": code_prefix,
        "task_desc": "collaborative Python function writing",
        # Placeholders: the bigcodebench branch dispatches on task_type and
        # ignores the persona matrix, but downstream asserts/metrics still read
        # these.
        "expertise": "expert",
        "goal": "solving",
        # R_task reads THIS, not ground_truth: BigCodeBench scores by running the
        # suite, so the tests are the scoring reference and the reference
        # solution is only a diagnostic.
        "test_source": test_source,
        "ground_truth": reference,
        "task_id": task_id,
        "entry_point": str(payload.get("entry_point") or "task_func"),
    }

    return {
        "prompt": messages,
        "ground_truth": reference,
        "raw_prompt": messages,
        "extra_info": {"interaction_kwargs": interaction_kwargs},
        "reward_model": {"style": "rule", "ground_truth": reference},
        "data_source": "collabllm_bigcodebench",
        "agent_name": "ctom_agent",
        "index": str(uuid.uuid4()),
    }


def _task_key(df: pd.DataFrame) -> pd.Series:
    """Stable per-task identity: the BigCodeBench task_id, falling back to conv_id."""
    tids = df["single_turn_metadata"].map(
        lambda m: (m or {}).get("task_id", "") if isinstance(m, dict) else ""
    )
    return tids.where(tids.astype(bool), df["conv_id"].astype(str))


def _build_split(df: pd.DataFrame, keys: list[str], name: str, prompt_key: str) -> Dataset:
    rows, dropped = [], 0
    for key in keys:
        row = collapse_group(df[df["_task"] == key], prompt_key)
        if row is None:
            dropped += 1
            continue
        rows.append(row)
    print(f"[INFO] {name}: {len(rows)} usable tasks ({dropped} dropped: no q0/spec/tests)")
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser(description="CollabLLM BigCodeBench-Chat processor")
    ap.add_argument("--large_file", required=True,
                    help="parquet of collabllm/collabllm-multiturn-bigcodebench-large (train split)")
    ap.add_argument("--small_file", default="",
                    help="parquet of collabllm/collabllm-multiturn-bigcodebench; its tasks that "
                         "do not appear in --large_file become the validation split")
    ap.add_argument("--runnable", default="",
                    help="JSON from data_prep/validate_coding_tasks.py. REQUIRED. Tasks absent "
                         "from its `runnable` list are DROPPED — their reference solution does "
                         "not pass in the execution sandbox, so they can only contribute a "
                         "constant 0 and a degenerate GRPO group.")
    ap.add_argument("--allow_unvalidated", action="store_true",
                    help="Build without --runnable. Debugging only: the result contains tasks "
                         "that can never score above 0.")
    ap.add_argument("--local_dir", required=True, help="output directory for the parquet files")
    ap.add_argument("--validation_size", type=int, default=119,
                    help="holdout size when --small_file is absent (default matches the "
                         "119 bigcodebench-only tasks)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.expanduser(os.path.expandvars(args.local_dir))
    os.makedirs(out_dir, exist_ok=True)

    keep: set[str] | None = None
    if args.runnable:
        with open(os.path.expandvars(args.runnable), encoding="utf-8") as fh:
            report = json.load(fh)
        keep = set(report.get("runnable", []))
        if not keep:
            print("[ERROR] the --runnable report lists no runnable tasks", file=sys.stderr)
            return 1
        print(f"[INFO] runnable filter: {len(keep)} of {report.get('num_tasks', '?')} tasks "
              f"(validated under {report.get('interpreter', '?')})")
    elif args.allow_unvalidated:
        print("[WARN] --allow_unvalidated: building WITHOUT the gold-solution filter. Tasks "
              "whose own reference solution fails in the sandbox are kept, and they can only "
              "ever score 0. Do not train on this.")
    else:
        print("[ERROR] --runnable is required. Produce it with:", file=sys.stderr)
        print("  CTOM_BCB_PYTHON=<sandbox> python data_prep/validate_coding_tasks.py \\",
              file=sys.stderr)
        print("      --files <hf parquet(s)> --out data/bigcodebench_runnable.json",
              file=sys.stderr)
        print("  (or pass --allow_unvalidated for a debugging build)", file=sys.stderr)
        return 1

    large = pd.read_parquet(os.path.expandvars(args.large_file))
    large["_task"] = _task_key(large)
    train_keys = sorted(large["_task"].unique())
    print(f"[INFO] {args.large_file}: {len(large)} rows -> {len(train_keys)} tasks")

    if args.small_file:
        small = pd.read_parquet(os.path.expandvars(args.small_file))
        small["_task"] = _task_key(small)
        shared = set(train_keys)
        val_keys = sorted(k for k in small["_task"].unique() if k not in shared)
        print(f"[INFO] {args.small_file}: {len(small)} rows -> "
              f"{small['_task'].nunique()} tasks, {len(val_keys)} not in train")
        val_source = small
    else:
        rng = random.Random(args.seed)
        rng.shuffle(train_keys)
        val_keys, train_keys = train_keys[: args.validation_size], train_keys[args.validation_size:]
        val_keys, train_keys = sorted(val_keys), sorted(train_keys)
        print(f"[WARN] no --small_file — holding out {len(val_keys)} training tasks instead")
        val_source = large

    assert not (set(train_keys) & set(val_keys)), "train/validation task overlap"

    if keep is not None:
        before = (len(train_keys), len(val_keys))
        train_keys = [k for k in train_keys if k in keep]
        val_keys = [k for k in val_keys if k in keep]
        print(f"[INFO] runnable filter: train {before[0]} -> {len(train_keys)}, "
              f"validation {before[1]} -> {len(val_keys)}")

    for prompt_key, suffix in SUFFIXES.items():
        train_ds = _build_split(large, train_keys, f"train[{prompt_key}]", prompt_key)
        val_ds = _build_split(val_source, val_keys, f"validation[{prompt_key}]", prompt_key)
        for ds, stem in ((train_ds, "rl_train"), (val_ds, "rl_validation")):
            path = os.path.join(out_dir, f"{stem}{suffix}.parquet")
            ds.to_parquet(path)
            print(f"[OK] {os.path.basename(path)} <- {len(ds)} tasks")

    print(f"\n[DONE] wrote {out_dir}")
    print("      rl_train{,.tag_prompt,.revise_prompt,.bdr_prompt,.v7_prompt}.parquet")
    print("      rl_validation{,.tag_prompt,.revise_prompt,.bdr_prompt,.v7_prompt}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
