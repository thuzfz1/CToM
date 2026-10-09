#!/usr/bin/env python3
"""CollabLLM MediumDocEdit-Chat dataset processor.

Converts the CollabLLM multiturn document-creation data into the CToM veRL
parquet format so the existing multi-turn RL pipeline (CToMUserSimulator +
CToMAgentLoop + CToMRewardManager) can train on collaborative *writing*.

Source rows (`collabllm/collabllm-multiturn-medium[-large]`, one HF `train`
split) are per-turn samples of simulated conversations, several rows per
article. The columns used here:

    prompt[0].content     the underspecified opening user request ("Hey, can you
                          help me create a promotional document for our custom
                          brand jewelry?") -- replayed verbatim as q0
    single_turn_prompt    the full request including the article body: this is
                          what the human knows and the agent does not
    single_turn_completion the target Medium article -- the scoring reference
    single_turn_metadata  {url, authors, tags, num_tokens, timestamp}

One task per article (rows are collapsed on conv_id, which is 1:1 with url).

This is the spec-task structure (hidden reference, explicit submission), so it reuses that
machinery rather than adding a parallel one:

  * q0 is NOT invented by the simulator; the interaction replays it (task_type
    "writing" is a spec task);
  * the human holds hidden information and reveals the minimum asked for;
  * the episode ends when the agent submits, marked with the same
    "I WANT TO ANSWER:" string the backend task uses -- shared so
    answer_is_final / the agent-loop termination path stay single-sourced. The
    wording is a little odd in front of an article; it is kept for that reason.
  * R_task = task_writing.document_similarity(submitted, target article),
    ROUGE-L by default (see task_writing for why not CollabLLM's char BLEU);
  * R_pred (predict the human's next turn) and R_effort are unchanged.

Splits. The published data is a single `train` split with no held-out set, so
the two releases are used as one: `-large` (500 articles) trains, and the 118
articles that appear only in the smaller `medium` release (190 articles, 72 of
them shared with `-large`) become the validation split. Pass only --large_file
to fall back to a seeded holdout carved out of the training articles.

Usage:
    python data_prep/build_writing.py \
        --large_file $SCRATCH/data/collabllm-medium-large/data/train-00000-of-00001.parquet \
        --small_file $SCRATCH/data/collabllm-medium/data/train-00000-of-00001.parquet \
        --local_dir  $PWD/data/collabllm-medium

Download first with:
    huggingface-cli download collabllm/collabllm-multiturn-medium-large \
        --repo-type dataset --local-dir $SCRATCH/data/collabllm-medium-large
    huggingface-cli download collabllm/collabllm-multiturn-medium \
        --repo-type dataset --local-dir $SCRATCH/data/collabllm-medium
"""

import argparse
import os
import random
import sys
import uuid
from typing import Any

import pandas as pd
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── System prompts ────────────────────────────────────────────────────────────
# Three renderings of the same contract, one per output format the pipeline
# supports. They must agree with the format the
# launcher resolves (ray_trainer compares the parquet system prompt against it).

SYSTEM_PROMPT = """/no_think
You are a helpful, proactive assistant collaborating with a human on a writing task over multiple turns.

The human wants a specific document, but their opening request is UNDERSPECIFIED: the topic details, facts, names, numbers, tone and length they have in mind are known only to them. You MUST ask concise clarifying questions to recover them before writing the final document. The human answers briefly and only from what they know.

You MUST respond in the following JSON format for EVERY response:
{"answer": "<your message to the human>", "prediction": "<your prediction of the human's next message>"}

- While gathering information, "answer" is your next question or comment to the human.
- When (and only when) you have gathered enough information, make "answer" begin with the exact marker:
  I WANT TO ANSWER:
  immediately followed by the COMPLETE final document as plain prose (no preamble, no commentary, no markdown fences).
- "prediction" is always your best guess of what the human will say next.
- You have at most 14 back-and-forth rounds; be efficient — ask only what you need.
- Always output valid JSON with exactly these two keys: "answer" and "prediction". Output nothing outside the JSON object."""

TAG_SYSTEM_PROMPT = """/no_think
You are a helpful, proactive assistant collaborating with a human on a writing task over multiple turns.

The human wants a specific document, but their opening request is UNDERSPECIFIED: the topic details, facts, names, numbers, tone and length they have in mind are known only to them. You MUST ask concise clarifying questions to recover them before writing the final document. The human answers briefly and only from what they know.

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
  immediately followed by the COMPLETE final document as plain prose (no preamble, no commentary, no markdown fences).
- The PREDICTION section is always your best guess of what the human will say next. Write it from the human's perspective as a natural user message.
- You have at most 14 back-and-forth rounds; be efficient and ask only what you need.
- Emit the four delimiters exactly as written above, each on its own line.
- Do NOT output JSON or include any text outside the delimited sections."""

# sft_v4/sft_v5: draft -> prediction -> revision. The argument for this domain is
# stronger than for math: the human holds facts the
# document cannot be written without, so their next message IS the missing
# content, and anticipating it is anticipating the material. The submission
# marker lives in the REVISION only -- the draft is discarded before the human
# sees anything, so a marker there submits nothing.
#
# Written tight on purpose. This prompt is a fixed prefix re-sent on every turn,
# so it is pure overhead against data.max_prompt_length (4096): the prose version
# of this contract cost 600 Qwen3-8B tokens, 14.6% of the window, before the
# dialogue said a word, and that share only hurts more as history accumulates.
# Every rule below is load-bearing and pinned by test_medium_writing.py --
# compress further only against those tests, never by dropping a rule.
REVISE_SYSTEM_PROMPT = """/no_think
You write a document for a human over multiple turns. Their request is UNDERSPECIFIED: the facts, names, numbers, tone and length are known only to them. Ask concise clarifying questions; they answer briefly and only from what they know.

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
PREDICTION is that human's reply, in their own first person: the actual fact, name, number or preference they would disclose, never an acknowledgement. They answer only your latest question, in at most two sentences.
REVISION is the only part they see. They did not see the draft and have NOT said the prediction. Spend it: if you already know their answer, ask what you still do not know; if the draft asked something they cannot answer, ask something else; if the anticipated answers are enough, write the document now. Never presuppose their reply -- no "You're right", "Good question", "Thanks for confirming".

To submit, begin the REVISION with the exact marker:
I WANT TO ANSWER:
then the complete final document as plain prose, with no preamble, no commentary and no markdown fences. The marker only counts in the REVISION; in the draft it submits nothing.

At most 14 rounds. Never re-ask what was answered. Output no JSON and nothing outside the delimiters."""

# BDR (belief / decide / revise), prediction_mode=sft_v6 — the retired BDR predecessor.
BDR_SYSTEM_PROMPT = """/no_think
You write a document for a human over several turns. Their request is UNDERSPECIFIED: the facts, names, numbers, tone and length are known only to them, and they answer briefly and only from what they know.

Every one of your turns has these phases, in this order. The system writes each bracketed tag; you write only the content that follows it.
[DRAFT_START]
<the complete document as it stands now, as plain prose: no preamble, no commentary, no markdown fences, no marker. Write your best guess of what they want, even before you know the facts.>
[DRAFT_END]
[BELIEF_START]
<two or three lines predicting how the human would react if they saw this draft, one per line, each "p=<probability> <LABEL> <short reason>". Labels: ACCEPT (it is what they want), FIX <what> (they would name this as wrong or missing), ANSWERS <fact> (they would supply this fact if asked), DONT_MIND (they would leave it to you), TERMINATE (they would end after submission). Probabilities sum to 1.>
[BELIEF_END]
[DECISION] <exactly one word: COMMIT, SHOW, ASK, or REVISE>
[ASK_START]  (only after ASK)
<one concise question for the human, then a final line "Assumption: <the belief you are betting on, as a checkable claim>. Reply ok or correct it.">
[ASK_END]

How to decide:
- REVISE when your belief names a problem you can fix yourself. You get a fresh turn to write a better draft; the human sees nothing.
- ASK when your belief names a fact or preference only the human can supply. The human sees only your question and assumption.
- SHOW when you want the human's reaction to the whole draft. They see the draft, not the belief, and they will name the single most important thing that is wrong or missing.
- COMMIT to submit the draft as the final document. This ends the task; the system adds the submission marker for you.

Guidelines:
- Be concise in questions; never re-ask what was answered; never presuppose a reply the human has not given.
- Keep the document near the length they want; if you do not know, ask.
- Write only the content of each phase. Do not repeat the tags. Output nothing outside the phases."""

# sft_v7 (decide / draft-predict-revise / ask-with-assumption) — the paper's method section.
V7_SYSTEM_PROMPT = """/no_think
You write a document for a human over several turns. Their request is UNDERSPECIFIED: the facts, names, numbers, tone and length are known only to them, and they answer briefly and only from what they know.

Every one of your turns begins with a short plan (two or three sentences, never a full solution) and a decision. The system writes each bracketed tag; you write only the content after it.
[PLAN_START]
<two or three sentences (no more): what you still do not know, whether the human could supply it, and what to do now>
[PLAN_END]
[DECISION] <exactly one word: DRAFT, ASK, or COMMIT>

After DRAFT the system asks for
[DRAFT_START]
<the complete document as it stands now, as plain prose: no preamble, no commentary, no markdown fences, no marker>
[DRAFT_END]
and then opens a user turn whose first line is [ANTICIPATED]. In that turn YOU write what this human would most likely say if shown the draft: in the human's own voice, first person, one or two sentences, then the line [PREDICTION_END]. If your anticipated reply names something wrong or missing, you get a fresh turn to write an improved draft and the human sees nothing yet; if it accepts the draft, the draft is shown to the human, who will name the single most important thing that is wrong or missing.

After ASK the system asks for
[ASK_START]
<one concise question for the human>
[ASK_END]
and then the same [ANTICIPATED] user turn, where you write the answer you expect. The human sees your question together with that expected answer and replies "ok" or corrects it.

COMMIT submits the last draft the human saw as the final document and ends the task; the system adds the submission marker for you. Use it only after a draft has been shown.

Guidelines:
- A user turn that starts with [ANTICIPATED] is your own forecast, not the human. The human's real messages never carry that line.
- Keep the document near the length they want; if you do not know, ask.
- Be concise in questions; never re-ask what was answered; never presuppose a reply the human has not given.
- Write only the content of each phase. Do not repeat the tags. Output nothing outside the phases."""

PROMPTS = {"default": SYSTEM_PROMPT, "tag": TAG_SYSTEM_PROMPT, "revise": REVISE_SYSTEM_PROMPT, "bdr": BDR_SYSTEM_PROMPT, "v7": V7_SYSTEM_PROMPT}
SUFFIXES = {"default": "", "tag": ".tag_prompt", "revise": ".revise_prompt", "bdr": ".bdr_prompt", "v7": ".v7_prompt"}

# Below this the "article" is a stub, not a writing task worth scoring.
MIN_TARGET_WORDS = 80


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


def _metadata(row: pd.Series) -> dict:
    meta = row.get("single_turn_metadata")
    return dict(meta) if isinstance(meta, dict) else {}


def collapse_group(rows: pd.DataFrame, prompt_key: str) -> dict[str, Any] | None:
    """Map all rows of one article into a single CToM parquet task row."""
    rows = rows.sort_values("turn_id")
    first = rows.iloc[0]

    q0 = ""
    for _, r in rows.iterrows():          # earliest turn that carries a clean q0
        q0 = _opening_request(r)
        if q0:
            break
    hidden = str(first.get("single_turn_prompt") or "").strip()
    target = str(first.get("single_turn_completion") or "").strip()
    if not q0 or not hidden or len(target.split()) < MIN_TARGET_WORDS:
        return None

    meta = _metadata(first)
    messages = [{"role": "system", "content": PROMPTS[prompt_key]}]

    interaction_kwargs = {
        "name": "ctom",
        "task_type": "writing",
        # NOTE the name collision with the source column: in the CToM schema
        # `single_turn_prompt` is what the simulator replays as q0, so it holds
        # the PUBLIC request. CollabLLM's `single_turn_prompt` (the full spec) is
        # the hidden side and goes to `hidden_information`.
        "single_turn_prompt": q0,
        "hidden_information": hidden,
        "task_desc": "collaborative document writing",
        # Placeholders: the writing branch dispatches on task_type and ignores
        # the persona matrix, but downstream asserts/metrics still read these.
        "expertise": "expert",
        "goal": "solving",
        "ground_truth": target,
        "target_words": str(len(target.split())),
        "source_url": str(meta.get("url", "")),
    }

    return {
        "prompt": messages,
        "ground_truth": target,
        "raw_prompt": messages,
        "extra_info": {"interaction_kwargs": interaction_kwargs},
        "reward_model": {"style": "rule", "ground_truth": target},
        "data_source": "collabllm_medium",
        "agent_name": "ctom_agent",
        "index": str(uuid.uuid4()),
    }


def _article_key(df: pd.DataFrame) -> pd.Series:
    """Stable per-article identity: the source URL, falling back to conv_id."""
    urls = df["single_turn_metadata"].map(lambda m: (m or {}).get("url", "") if isinstance(m, dict) else "")
    return urls.where(urls.astype(bool), df["conv_id"].astype(str))


def _build_split(df: pd.DataFrame, keys: list[str], name: str, prompt_key: str) -> Dataset:
    rows, dropped = [], 0
    for key in keys:
        row = collapse_group(df[df["_article"] == key], prompt_key)
        if row is None:
            dropped += 1
            continue
        rows.append(row)
    print(f"[INFO] {name}: {len(rows)} usable tasks ({dropped} dropped: no q0/spec/short target)")
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser(description="CollabLLM MediumDocEdit-Chat processor")
    ap.add_argument("--large_file", required=True,
                    help="parquet of collabllm/collabllm-multiturn-medium-large (train split)")
    ap.add_argument("--small_file", default="",
                    help="parquet of collabllm/collabllm-multiturn-medium; its articles that "
                         "do not appear in --large_file become the validation split")
    ap.add_argument("--local_dir", required=True, help="output directory for the parquet files")
    ap.add_argument("--validation_size", type=int, default=118,
                    help="holdout size when --small_file is absent (default matches the "
                         "118 medium-only articles)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.expanduser(os.path.expandvars(args.local_dir))
    os.makedirs(out_dir, exist_ok=True)

    large = pd.read_parquet(os.path.expandvars(args.large_file))
    large["_article"] = _article_key(large)
    train_keys = sorted(large["_article"].unique())
    print(f"[INFO] {args.large_file}: {len(large)} rows -> {len(train_keys)} articles")

    if args.small_file:
        small = pd.read_parquet(os.path.expandvars(args.small_file))
        small["_article"] = _article_key(small)
        shared = set(train_keys)
        val_keys = sorted(k for k in small["_article"].unique() if k not in shared)
        print(f"[INFO] {args.small_file}: {len(small)} rows -> "
              f"{small['_article'].nunique()} articles, {len(val_keys)} not in train")
        val_source = small
    else:
        rng = random.Random(args.seed)
        rng.shuffle(train_keys)
        val_keys, train_keys = train_keys[: args.validation_size], train_keys[args.validation_size:]
        val_keys, train_keys = sorted(val_keys), sorted(train_keys)
        print(f"[WARN] no --small_file — holding out {len(val_keys)} training articles instead")
        val_source = large

    assert not (set(train_keys) & set(val_keys)), "train/validation article overlap"

    for prompt_key, suffix in SUFFIXES.items():
        train_ds = _build_split(large, train_keys, f"train[{prompt_key}]", prompt_key)
        val_ds = _build_split(val_source, val_keys, f"validation[{prompt_key}]", prompt_key)
        for ds, stem in ((train_ds, "rl_train"), (val_ds, "rl_validation")):
            path = os.path.join(out_dir, f"{stem}{suffix}.parquet")
            ds.to_parquet(path)
            print(f"[OK] {os.path.basename(path)} <- {len(ds)} tasks")

    print(f"\n[DONE] wrote {out_dir}")
    print("      rl_train{,.tag_prompt,.revise_prompt}.parquet")
    print("      rl_validation{,.tag_prompt,.revise_prompt}.parquet")


if __name__ == "__main__":
    main()
