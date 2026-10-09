#!/usr/bin/env python3
"""
eval_math.py — End-to-end evaluation for CToM trained models.

Loads a veRL LoRA checkpoint, merges into the base model, serves via vLLM,
runs full multi-turn rollout with a user simulator (Together API or local
vLLM server), and computes R_task + R_pred + R_effort.

Architecture (matches training-time loop):
  1. User simulator generates q0 from persona + problem
  2. Assistant generates (a_t, e_t) — answer + <user_prediction>
  3. Strip prediction, feed clean answer back to user sim
  4. Repeat until max turns or termination signal
  5. Score: R_task (correct answer / understood), R_pred (F1), R_effort (efficiency)

Usage:
    # Evaluate a LoRA checkpoint
    python eval_math.py \
        --checkpoint_path checkpoints/global_step_200/actor \
        --test_data data/math-l5-expert/rl_validation.parquet \
        --prompt_dir prompts \
        --max_samples 100

    # Evaluate base model (no LoRA)
    python eval_math.py \
        --test_data data/math-l5-expert/rl_validation.parquet \
        --prompt_dir prompts \
        --max_samples 50

    # Use local vLLM user sim server instead of Together API
    python eval_math.py \
        --checkpoint_path checkpoints/global_step_200/actor \
        --test_data data/math-l5-expert/rl_validation.parquet \
        --prompt_dir prompts \
        --user_sim_api_base http://localhost:8100/v1

    # Already-merged model
    python eval_math.py \
        --skip_merge \
        --merged_model_dir ./merged_models/step_200 \
        --test_data data/math-l5-expert/rl_validation.parquet \
        --prompt_dir prompts
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ctom import markers as _markers
from ctom import controller as _ctom
from ctom.task_math import check_math_answer
# Submit marker for the spec task families (coding, writing).
# Single-sourced with the training loop so the two decoders cannot drift.
from ctom.submission import compose_revision, forced_submit_marker


# ============================================================================
# Constants — must match agent_loop.py / user_simulator.py
# ============================================================================

PREDICTION_START_TAG = "<user_prediction>"
PREDICTION_END_TAG = "</user_prediction>"
SFT_V3_ANSWER_START_TAG = "[ANSWER_START]"
SFT_V3_ANSWER_END_TAG = "[ANSWER_END]"
SFT_V3_PREDICTION_START_TAG = "[PREDICTION_START]"
SFT_V3_PREDICTION_END_TAG = "[PREDICTION_END]"
# sft_v4 (predict-then-revise) adds the third controller-owned phase.
SFT_V4_REVISION_START_TAG = "[REVISION_START]"
SFT_V4_REVISION_END_TAG = "[REVISION_END]"
TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"
UNDERSTOOD_TOKEN = "<understood>"

SYSTEM_PROMPT = """/no_think
You are a helpful, proactive, and highly interactive assistant engaged in a multi-turn conversation.

At the end of each response, you MUST predict what the user will say next. Wrap your prediction in tags:
<user_prediction>Your prediction of the user's next message</user_prediction>

This prediction should reflect your understanding of the user's current state of understanding and what they are likely to ask or say next.

Guidelines:
- Maintain context and continuity across turns.
- If the user's message is unclear, ask for clarification.
- Ask specific follow-up questions based on the user's needs.
- Keep interactions concise and efficient.
- Adapt your communication style to the user based on conversational cues.
- Always end with a <user_prediction>...</user_prediction> block."""

PERSONA_MATRIX = {
    ("expert", "solving"): (
        "You are an expert user who wants the answer quickly. "
        "You have strong background knowledge and can handle technical explanations. "
        "You don't need hand-holding — just the correct solution efficiently."
    ),
    ("novice", "solving"): (
        "You are a novice user who wants the answer quickly. "
        "You have limited background knowledge and need simpler explanations. "
        "You may get confused by jargon or overly complex solutions."
    ),
}

USER_PROMPT_TEMPLATE_SOLVING = None  # Loaded from file at runtime
USER_PROMPT_TEMPLATE_LEARNING = None  # Loaded from file at runtime


# ============================================================================
# Prompt template loading (matches CToMUserSimulator.prompt_dir)
# ============================================================================

def load_prompt_templates(prompt_dir: str) -> tuple[str, str]:
    """
    Load user simulator prompt templates from external files.

    Expected files:
        {prompt_dir}/sim_solving.txt

    These templates use .format() placeholders:
        solving:  {persona_desc}, {problem}, {ground_truth}, {chat_history},
                  {termination_signal}, {ending_rule}
    """
    solving_path = os.path.join(prompt_dir, "sim_solving.txt")

    if not os.path.isfile(solving_path):
        raise FileNotFoundError(f"Prompt template not found: {solving_path}")

    with open(solving_path, "r", encoding="utf-8") as f:
        solving_template = f.read()

    print(f"[INFO] Loaded prompt templates from: {prompt_dir}")
    return solving_template


def solving_ending_rule(answer_gated_termination: bool) -> str:
    """The {ending_rule} slot of sim_solving*.txt — mirrors CToMUserSimulator.

    Must stay byte-identical to CToMUserSimulator._solving_ending_rule, or the
    simulator is told a different rule at eval time than it was trained under.
    test_answer_gate_eval_parity.py asserts that equality against the training
    method for both flag values; it is not a stylistic duplicate.
    """
    if not answer_gated_termination:
        return (
            f'- If the AI has fully solved the problem correctly, set response to '
            f'"{TERMINATION_SIGNAL}".'
        )
    return (
        "- NEVER end the conversation yourself and NEVER emit any stop or "
        "termination token — there is none. Just keep responding naturally.\n"
        "- If the AI has reached the correct final answer but has NOT written it "
        "inside \\boxed{}, do not accept it yet: tell the AI to state its final "
        "answer inside \\boxed{}. Do not reveal the reference answer while asking.\n"
        "- If the AI's answer is correct AND already inside \\boxed{}, you will "
        "not be asked again — the conversation is over."
    )


def _answer_is_correct(text: str, ground_truth: str) -> bool:
    """Strict-checker verdict for the answer gate — mirrors CToMUserSimulator.

    Fail-closed like training: a checker error must never end an episode early,
    so any exception reports "not correct" and the dialogue continues.
    """
    if not text or not ground_truth:
        return False
    try:
        return bool(check_math_answer(text, ground_truth))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] answer gate check failed ({exc}); not terminating")
        return False


# ============================================================================
# User sim JSON response parsing (matches _parse_sim_json_response)
# ============================================================================

def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from raw text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _repair_json_escapes(s: str) -> str:
    """Double-up backslashes that aren't valid JSON escape sequences.

    LLMs often emit LaTeX inside JSON strings with raw backslashes
    (e.g. \\frac, \\overline) which are invalid JSON escapes.
    This converts them to properly escaped form so json.loads succeeds.
    """
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)


def _parse_sim_json_response(raw_response: str) -> dict:
    """
    Parse structured JSON output from the user simulator.

    The sim returns JSON (possibly inside ```json ... ``` fences) with keys:
        - current_status / current_understanding
        - thought
        - response   ← the actual user message to forward

    Falls back to using the raw text as the response if parsing fails.
    """
    clean = _strip_think_blocks(raw_response)

    # Try ```json ... ``` fenced block first
    json_match = re.search(r"```json\s*(.*?)\s*```", clean, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try bare JSON object
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {"response": clean.strip(), "_parse_failed": True}

    # Try raw first, then with repaired LaTeX backslash escapes
    for candidate in [json_str, _repair_json_escapes(json_str)]:
        try:
            parsed = json.loads(candidate)
            response = parsed.get("response", "")
            if not response:
                continue
            return {
                "current_status": parsed.get("current_status", parsed.get("current_understanding", "")),
                "thought": parsed.get("thought", ""),
                "response": response,
                "_parse_failed": False,
            }
        except json.JSONDecodeError:
            continue

    return {"response": clean.strip(), "_parse_failed": True}


def _strip_think_blocks_from_msg(msg: dict) -> dict:
    """Strip think blocks from a message dict's content (for chat history formatting)."""
    content = msg.get("content", "")
    cleaned = _strip_think_blocks(content)
    return {**msg, "content": cleaned}


# ============================================================================
# Parsing helpers
# ============================================================================

def parse_answer_and_prediction(content: str) -> tuple[str, str]:
    """Parse assistant output into (answer, predicted_user_response).

    Order of preference:
      1. The trained JSON ``{"answer", "prediction"}`` format (via the canonical
         ctom.utils.parse_assistant_response, which also strips <think> blocks and
         repairs minor JSON issues). This is what the assistant models are actually
         trained to emit — without it the JSON-format models look like they output
         "nonsense JSON" and their prediction is never retrieved.
      2. Legacy ``<user_prediction>...</user_prediction>`` tag format.
      3. Raw text as the answer, no prediction.
    """
    # 1. JSON {answer, prediction}
    try:
        from ctom.utils import parse_assistant_response
        parsed = parse_assistant_response(content)
        ans, pred = str(parsed.get("answer", "")), str(parsed.get("prediction", ""))
        if pred.strip():
            return ans.strip(), pred.strip()
    except Exception:
        parsed = None

    # 2. Legacy <user_prediction> tag
    match = re.search(
        rf"{re.escape(PREDICTION_START_TAG)}(.*?){re.escape(PREDICTION_END_TAG)}",
        content,
        re.DOTALL,
    )
    if match:
        prediction = match.group(1).strip()
        answer = content[: match.start()].strip()
        return answer, prediction

    # 3. No prediction: prefer the JSON answer field if it parsed, else raw text
    if parsed is not None and str(parsed.get("answer", "")).strip():
        return str(parsed["answer"]).strip(), ""
    return content.strip(), ""


def format_chat_history(messages: list[dict], strip_system: bool = True) -> str:
    """Format message list into readable chat history for user sim prompt.
    Strips think blocks from messages (matching CToMUserSimulator._parse_messages)."""
    filtered = messages if not strip_system else [m for m in messages if m["role"] != "system"]
    filtered = [_strip_think_blocks_from_msg(m) for m in filtered]
    lines = [f"**{m['role'].capitalize()}**: {m['content']}" for m in filtered]
    return "\n".join(lines)


def truncate_messages(
    messages: list[dict],
    max_tokens: int,
    keep_last: int = 2,
) -> list[dict]:
    """Drop oldest non-system messages until estimated tokens fit max_tokens.

    Always keeps system messages and the last `keep_last` non-system messages.
    Uses word_count * 1.3 as a rough token estimate.
    Returns original list if already within budget or max_tokens <= 0.
    """
    if max_tokens <= 0:
        return messages

    def _est(m):
        return int(len(m.get("content", "").split()) * 1.3)

    if sum(_est(m) for m in messages) <= max_tokens:
        return messages

    system = [m for m in messages if m["role"] == "system"]
    non_sys = [m for m in messages if m["role"] != "system"]
    protected = non_sys[-keep_last:] if keep_last > 0 else []
    droppable = non_sys[:-keep_last] if keep_last > 0 else list(non_sys)

    while droppable and sum(_est(m) for m in system + droppable + protected) > max_tokens:
        droppable.pop(0)

    result = system + droppable + protected
    if len(result) < len(messages):
        print(f"[INFO] Context truncation: {len(messages)} -> {len(result)} messages "
              f"(~{sum(_est(m) for m in result)} est. tokens)")
    return result


# ============================================================================
# Reward computation (standalone, no veRL dependency)
# ============================================================================

def compute_task_reward(messages: list[dict], ground_truth: str, goal: str) -> float:
    """R_task: 1.0 if a visible turn shows a correct boxed answer, else 0.0.

    Solving: checks ALL assistant messages for the correct answer (not just the
    last one), since the conversation may continue with pleasantries after the
    answer is given.  Also treats the user simulator sending [[TERMINATE CHAT]]
    as corroborating evidence — but the math check is authoritative.

    Learning: 1.0 if any user message contains <understood> OR [[TERMINATE CHAT]]
    (the latter signals the simulated user was satisfied).
    """
    user_msgs = [m for m in messages if m["role"] == "user"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    has_terminate = any(TERMINATION_SIGNAL in m["content"] for m in user_msgs)

    if goal == "solving":
        if not assistant_msgs:
            return 0.0
        # Check every assistant turn for the correct answer
        if any(check_math_answer(m["content"], ground_truth) for m in assistant_msgs):
            return 1.0
        return 0.0
    return 0.0


def compute_f1_prediction_reward(messages: list[dict], predictions: list[str]) -> float:
    """R_pred: average word-overlap F1 between predictions and actual next user turns."""
    if not predictions:
        return 0.0
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    actual_responses = user_msgs[1:]  # predictions[i] predicts user turn i+1
    num_pairs = min(len(predictions), len(actual_responses))
    if num_pairs == 0:
        return 0.0
    total_f1 = 0.0
    for i in range(num_pairs):
        pred, actual = predictions[i], actual_responses[i]
        if not pred or not actual:
            continue
        pred_words = set(pred.lower().split())
        actual_words = set(actual.lower().split())
        if not pred_words or not actual_words:
            continue
        intersection = pred_words & actual_words
        precision = len(intersection) / len(pred_words)
        recall = len(intersection) / len(actual_words)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        total_f1 += f1
    return total_f1 / num_pairs


def compute_effort_reward(
    messages: list[dict],
    max_turns: int = 8,
    max_user_tokens: int = 256,
    alpha: float = 0.5,
    max_assistant_tokens: int = 0,
    beta: float = 0.0,
) -> float:
    """R_effort: efficiency reward penalizing more turns, user tokens, and assistant tokens.

    R_effort = alpha * (1 - n/N)
             + (1-alpha-beta) * (1 - tau_user/T_user)
             + beta * (1 - tau_asst/T_asst)

    When beta=0 (default), reduces to the original two-term formula.
    """
    user_msgs = [m for m in messages if m["role"] == "user"]
    asst_msgs = [m for m in messages if m["role"] == "assistant"]
    n = len(user_msgs)
    tau_user = sum(len(m["content"].split()) for m in user_msgs)
    tau_asst = sum(len(m["content"].split()) for m in asst_msgs)
    turn_eff = max(0.0, 1.0 - n / max_turns) if max_turns > 0 else 0.0
    user_token_eff = max(0.0, 1.0 - tau_user / max_user_tokens) if max_user_tokens > 0 else 0.0
    asst_token_eff = max(0.0, 1.0 - tau_asst / max_assistant_tokens) if max_assistant_tokens > 0 else 0.0
    user_token_weight = 1.0 - alpha - beta
    return alpha * turn_eff + user_token_weight * user_token_eff + beta * asst_token_eff


def compute_effort_reward_per_turn(
    messages: list[dict],
    max_turns: int = 4,
    max_user_tokens: int = 256,
    alpha: float = 0.5,
    max_assistant_tokens: int = 0,
    beta: float = 0.0,
    *,
    tokenizer=None,
    token_units: str = "words",
) -> list[float]:
    """Use training's effort implementation on the visible transcript."""
    from ctom.reward import compute_effort_reward_per_turn as training_effort

    if token_units == "tokens" and tokenizer is None:
        raise ValueError("Token-based effort scoring requires the assistant tokenizer")
    return training_effort(
        messages, max_turns, max_user_tokens, alpha, max_assistant_tokens, beta,
        tokenizer=tokenizer, token_units=token_units,
    )


def _phase_is_valid(validity: dict) -> bool:
    flags = [v for k, v in validity.items() if k.endswith("_valid")]
    return bool(flags) and all(flags)


# ============================================================================
# 1. LoRA Checkpoint Merging
# ============================================================================

def _is_fsdp_checkpoint(checkpoint_path: str) -> bool:
    """Check if a checkpoint directory contains FSDP-sharded weights."""
    for d in [checkpoint_path, os.path.join(checkpoint_path, "actor")]:
        if os.path.isfile(os.path.join(d, "fsdp_config.json")):
            return True
        # Also check for model_world_size_*_rank_*.pt pattern
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith("model_world_size_") and f.endswith(".pt"):
                    return True
    return False


def _find_actor_dir(checkpoint_path: str) -> str:
    """
    Locate the actor directory within a veRL checkpoint.

    veRL checkpoint layout:
        <ckpt>/actor/model_world_size_N_rank_*.pt
        <ckpt>/actor/fsdp_config.json
    or directly:
        <ckpt>/model_world_size_N_rank_*.pt
        <ckpt>/fsdp_config.json
    """
    actor_subdir = os.path.join(checkpoint_path, "actor")
    if os.path.isdir(actor_subdir) and os.path.isfile(os.path.join(actor_subdir, "fsdp_config.json")):
        return actor_subdir
    if os.path.isfile(os.path.join(checkpoint_path, "fsdp_config.json")):
        return checkpoint_path
    # Check for model shard files directly
    for d in [actor_subdir, checkpoint_path]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith("model_world_size_") and f.endswith(".pt"):
                    return d
    raise FileNotFoundError(
        f"Cannot find FSDP checkpoint (fsdp_config.json or model_world_size_*_rank_*.pt) "
        f"in {checkpoint_path} or {checkpoint_path}/actor/"
    )


def merge_checkpoint(
    base_model_path: str,
    checkpoint_path: str,
    merged_output_dir: str,
) -> str:
    """
    Merge a veRL checkpoint into a standalone HuggingFace model.

    Handles two checkpoint formats:
    1. FSDP-sharded (model_world_size_N_rank_*.pt + fsdp_config.json)
       → Uses `python -m verl.model_merger merge --backend fsdp`
    2. Plain LoRA adapter (adapter_model.safetensors + adapter_config.json)
       → Uses PeftModel.from_pretrained + merge_and_unload
    """
    if os.path.exists(os.path.join(merged_output_dir, "config.json")):
        print(f"[INFO] Merged model already exists at {merged_output_dir}, skipping merge.")
        return merged_output_dir

    if _is_fsdp_checkpoint(checkpoint_path):
        # ── FSDP-sharded checkpoint → verl.model_merger ──
        actor_dir = _find_actor_dir(checkpoint_path)
        print(f"[INFO] Detected FSDP-sharded checkpoint at: {actor_dir}")
        print(f"[INFO] Merging via verl.model_merger → {merged_output_dir}")

        result = subprocess.run(
            [
                sys.executable, "-m", "verl.model_merger", "merge",
                "--backend", "fsdp",
                "--local_dir", actor_dir,
                "--target_dir", merged_output_dir,
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"[ERROR] verl.model_merger stderr:\n{result.stderr}")
            raise RuntimeError(f"verl.model_merger failed with return code {result.returncode}")

        print(f"[INFO] Merged model saved to: {merged_output_dir}")
        return merged_output_dir

    else:
        # ── Plain LoRA adapter → PeftModel ──
        from peft import PeftModel as _PeftModel
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        # Find adapter directory
        candidates = [
            checkpoint_path,
            os.path.join(checkpoint_path, "actor", "lora_adapter"),
            os.path.join(checkpoint_path, "actor"),
            os.path.join(checkpoint_path, "lora_adapter"),
        ]
        adapter_path = None
        for path in candidates:
            if os.path.exists(os.path.join(path, "adapter_model.safetensors")) or \
               os.path.exists(os.path.join(path, "adapter_model.bin")):
                adapter_path = path
                break

        if adapter_path is None:
            raise FileNotFoundError(
                f"No LoRA adapter or FSDP checkpoint found under {checkpoint_path}. "
                f"Searched: {candidates}"
            )

        print(f"[INFO] Found plain LoRA adapter at: {adapter_path}")
        print(f"[INFO] Loading base model: {base_model_path}")
        base_model = _AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        print(f"[INFO] Loading LoRA adapter: {adapter_path}")
        model = _PeftModel.from_pretrained(base_model, adapter_path)
        print("[INFO] Merging LoRA weights...")
        merged = model.merge_and_unload()

        os.makedirs(merged_output_dir, exist_ok=True)
        merged.save_pretrained(merged_output_dir)
        _AutoTokenizer.from_pretrained(base_model_path).save_pretrained(merged_output_dir)
        print(f"[INFO] Merged model saved to: {merged_output_dir}")

        del base_model, model, merged
        torch.cuda.empty_cache()
        return merged_output_dir


# ============================================================================
# 2. Data Loading
# ============================================================================

def load_test_data(parquet_path: str, max_samples: int = 0, seed: int = 42) -> list[dict]:
    """
    Load evaluation data from the CToM parquet format.

    Each row has:
      - prompt: [{"role": "system", "content": ...}]   (may be numpy array)
      - extra_info.interaction_kwargs: {name, single_turn_prompt, expertise, goal, ground_truth}
      - ground_truth: str
    """
    print(f"[INFO] Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[INFO] Loaded {len(df)} rows")

    if max_samples > 0 and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        print(f"[INFO] Subsampled to {len(df)} rows")

    samples = []
    for idx, row in df.iterrows():
        # Handle numpy array prompts (known veRL parquet issue).
        # NB: test for emptiness explicitly -- `a or b` calls bool() on the
        # value, which raises on any prompt array with >1 message (e.g. the
        # puremath parquets, whose prompt is [system, user]).
        prompt = row.get("prompt")
        if prompt is None or (hasattr(prompt, "__len__") and len(prompt) == 0):
            prompt = row.get("raw_prompt")
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()

        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        interaction_kwargs = extra_info.get("interaction_kwargs", {})
        ground_truth = interaction_kwargs.get("ground_truth", row.get("ground_truth", ""))

        samples.append({
            "index": row.get("index", str(idx)),
            "system_prompt": prompt,
            "single_turn_prompt": interaction_kwargs.get("single_turn_prompt", ""),
            "expertise": interaction_kwargs.get("expertise", "novice"),
            "goal": interaction_kwargs.get("goal", "solving"),
            "ground_truth": ground_truth,
            "solution": extra_info.get("solution", ""),
            "subject": extra_info.get("subject", ""),
            "difficulty": extra_info.get("difficulty", ""),
        })

    return samples


# ============================================================================
# 3. User Simulator Client
# ============================================================================

class UserSimulator:
    """
    Lightweight user simulator client matching CToMUserSimulator.generate_response.

    Supports Together API or local vLLM server via OpenAI-compatible API.

    Key behaviors matching training-time CToMUserSimulator:
    - Loads the prompt template from prompts/sim_solving.txt
    - Passes ground_truth into the prompt
    - Parses structured JSON output {current_status, thought, response}
    - Strips <think>...</think> blocks before parsing
    - Terminates on TERMINATION_SIGNAL, or on the answer gate
    """

    def __init__(
        self,
        prompt_dir: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        novice_model: str = "Qwen/Qwen3-32B-FP8",
        expert_model: str = "Qwen/Qwen3-32B-FP8",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        num_retries: int = 3,
        answer_gated_termination: bool = False,
    ):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.novice_model = novice_model
        self.expert_model = expert_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_retries = num_retries
        # Must match the training run's interaction config. A checkpoint trained
        # with the gate ON has never seen a simulator that emits a stop token.
        self.answer_gated_termination = answer_gated_termination

        # Multi-backend support (comma-separated URLs, matching training setup)
        if api_base and "," in api_base:
            self._api_bases = [u.strip() for u in api_base.split(",") if u.strip()]
        elif api_base:
            self._api_bases = [api_base]
        else:
            self._api_bases = []
        self._clients: list = []
        self._rr_counter = 0

        # Load prompt templates from files
        self.prompt_template_solving = load_prompt_templates(prompt_dir)
        novice_solving = os.path.join(prompt_dir, "sim_solving_novice.txt")
        self.prompt_template_solving_novice = (
            Path(novice_solving).read_text(encoding="utf-8")
            if os.path.isfile(novice_solving)
            else self.prompt_template_solving
        )

        if not self._api_bases and not self.api_key:
            raise ValueError(
                "Either --user_sim_api_base (local vLLM) or TOGETHER_API_KEY must be set."
            )

    def _get_client(self):
        if not self._clients:
            if self._api_bases:
                from openai import AsyncOpenAI
                for base_url in self._api_bases:
                    self._clients.append(AsyncOpenAI(
                        base_url=base_url,
                        api_key=self.api_key or "not-needed",
                    ))
                print(f"[INFO] User simulator using {len(self._clients)} local backend(s): {', '.join(self._api_bases)}")
            else:
                from together import AsyncTogether
                self._clients.append(AsyncTogether(api_key=self.api_key))
                print("[INFO] User simulator using Together API")
        # Round-robin across backends
        client = self._clients[self._rr_counter % len(self._clients)]
        self._rr_counter += 1
        return client

    def _get_model(self, expertise: str) -> str:
        return self.expert_model if expertise == "expert" else self.novice_model

    async def generate_user_turn(
        self,
        messages: list[dict],
        problem: str,
        ground_truth: str,
        expertise: str,
        goal: str,
        counterfactual: bool = False,
    ) -> tuple[str, bool, dict]:
        """
        Generate a user simulator turn.

        counterfactual=True labels the prediction slot without advancing the
        dialogue, and is exempt from the answer gate — exactly as in training.

        Returns:
            (response_text, should_terminate, extra_info)
        """
        # ── Answer-checker termination gate (solving only) ────────────────────
        # Mirrors CToMUserSimulator: with the gate on, a correct boxed answer ends
        # the episode here and the simulator is never asked. Returning an empty
        # response makes the caller stop without appending a user turn.
        if (
            self.answer_gated_termination
            and not counterfactual
            and goal == "solving"
            and ground_truth
            and messages
            and messages[-1].get("role") == "assistant"
            and _answer_is_correct(messages[-1].get("content", ""), ground_truth)
        ):
            return "", True, {"answer_gate_fired": True}

        persona_desc = PERSONA_MATRIX[(expertise, goal)]
        chat_history = format_chat_history(messages, strip_system=True)

        # Build prompt from external template (matching CToMUserSimulator)
        if goal == "solving":
            template = (
                self.prompt_template_solving_novice
                if expertise == "novice"
                else self.prompt_template_solving
            )
            prompt = template.format(
                persona_desc=persona_desc,
                problem=problem,
                ground_truth=ground_truth,
                chat_history=chat_history,
                termination_signal=TERMINATION_SIGNAL,
                ending_rule=solving_ending_rule(self.answer_gated_termination),
            )
        else:
            template = (
                self.prompt_template_solving_novice
                if expertise == "novice"
                else self.prompt_template_solving
            )
            prompt = template.format(
                persona_desc=persona_desc,
                problem=problem,
                ground_truth=ground_truth,
                chat_history=chat_history,
                understood_token=UNDERSTOOD_TOKEN,
                termination_signal=TERMINATION_SIGNAL,
            )

        client = self._get_client()
        model = self._get_model(expertise)
        # Training retries a syntactically malformed simulator JSON response
        # twice, in addition to transport retries. Match that behavior here.
        parsed = None
        for parse_attempt in range(3):
            raw_response = ""
            for attempt in range(self.num_retries):
                try:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    raw_response = completion.choices[0].message.content or ""
                    break
                except Exception as e:
                    err = str(e).lower()
                    if "rate" in err or "429" in err:
                        wait = min(2 ** attempt, 60)
                        print(f"[WARN] Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"[ERROR] User sim error (attempt {attempt+1}): {e}")
                        if attempt == self.num_retries - 1:
                            return "", True, {"parse_failed": True, "error": str(e)}

            parsed = _parse_sim_json_response(raw_response)
            if not parsed.get("_parse_failed") or parse_attempt == 2:
                break
            print(
                f"[WARN] User sim JSON parse failed "
                f"(attempt {parse_attempt + 1}/3); regenerating."
            )

        response = parsed["response"]

        extra_info = {
            "understood": UNDERSTOOD_TOKEN in response,
            "sim_thought": parsed.get("thought", ""),
            "sim_status": parsed.get("current_status", ""),
            "parse_failed": parsed.get("_parse_failed", False),
        }

        # Determine termination (matching CToMUserSimulator logic exactly)
        should_terminate = False
        if goal == "solving":
            # With the gate on, the checker above is the ONLY way a solving
            # episode ends. Reaching here means the answer was not correct, so
            # the simulator's token is deliberately ignored — that is the
            # false-termination case the gate exists to remove.
            should_terminate = (
                False
                if self.answer_gated_termination
                else TERMINATION_SIGNAL in response
            )

        return response, should_terminate, extra_info


# ============================================================================
# 4. Assistant (vLLM)
# ============================================================================

class AssistantModel:
    """Wrapper around vLLM for the trained assistant model."""

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ):
        from vllm import LLM, SamplingParams

        print(f"[INFO] Loading assistant model via vLLM: {model_path}")
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        print("[INFO] Assistant model loaded.")

    def generate(self, messages: list[dict]) -> str:
        """Generate a response given the conversation history."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    def generate_batch(self, messages_batch: list[list[dict]]) -> list[str]:
        """Batch generation for efficiency."""
        prompts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            for msgs in messages_batch
        ]
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]


class AsyncAssistantClient:
    """
    Async assistant client that connects to vLLM API server(s).

    Replaces the synchronous AssistantModel when the assistant is served
    externally as vLLM OpenAI-compatible API servers — matching how training
    uses separate vLLM rollout workers.

    Supports multiple backends with round-robin distribution (same pattern
    as UserSimulator and training-time CToMUserSimulator).
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        num_retries: int = 3,
        output_mode: str = "standard",
        tokenizer_path: Optional[str] = None,
        pred_answer_max_tokens: int = 2048,
        prediction_max_tokens: int = 384,
        pred_revision_max_tokens: int = 2048,
        trajectory_response_length: int = 8192,
        submit_marker_carryover: bool = False,
        bdr_belief_max_tokens: int = 256,
        bdr_ask_max_tokens: int = 512,
        bdr_max_internal_revisions: int = 1,
        v7_think_max_tokens: int = 256,
        v7_ask_max_tokens: int = 512,
        v7_max_inner_revisions: int = 3,
        v7_action_protocol: str = "legacy",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.output_mode = output_mode
        self.pred_answer_max_tokens = pred_answer_max_tokens
        self.prediction_max_tokens = prediction_max_tokens
        self.pred_revision_max_tokens = pred_revision_max_tokens
        self.trajectory_response_length = trajectory_response_length
        # Submit-marker carry-over, mirroring CToMAgentLoop._generate_sft_v4_turn.
        # Only the submit-task domains (coding, writing) set
        # this; math has no marker, so its eval corpus is provably untouched.
        self.submit_marker_carryover = submit_marker_carryover
        # sft_v6 (BDR): pred_answer_max_tokens is the DRAFT cap; these bound the
        # belief and ask phases and the internal REVISE turns per user round.
        self.bdr_belief_max_tokens = bdr_belief_max_tokens
        self.bdr_ask_max_tokens = bdr_ask_max_tokens
        self.bdr_max_internal_revisions = bdr_max_internal_revisions
        # sft_v7: pred_answer_max_tokens is the DRAFT cap and prediction_max_tokens
        # the ANTICIPATED user turn; these bound the plan and ask phases and the
        # inner revisions per round. Defaults mirror submit_math_v7.sh.
        self.v7_think_max_tokens = v7_think_max_tokens
        self.v7_ask_max_tokens = v7_ask_max_tokens
        self.v7_max_inner_revisions = v7_max_inner_revisions
        self.v7_action_protocol = _ctom.validate_action_protocol(v7_action_protocol)
        self._v7_slot_cache = None
        self.tokenizer = None
        self._system_prompt_tokens: list[int] = []
        if output_mode in ("sft_v3", "sft_v4", "sft_v5", "sft_v6", "sft_v7"):
            if not tokenizer_path:
                raise ValueError(
                    f"{output_mode} assistant mode requires tokenizer_path"
                )
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
            )
            token1 = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                add_generation_prompt=False,
                tokenize=True,
            )
            token2 = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}] * 2,
                add_generation_prompt=False,
                tokenize=True,
            )
            self._system_prompt_tokens = token1[: -(len(token2) - len(token1))]

        if "," in api_base:
            self._api_bases = [u.strip() for u in api_base.split(",") if u.strip()]
        else:
            self._api_bases = [api_base]
        self._clients: list = []
        self._rr_counter = 0

    def _get_client(self):
        if not self._clients:
            from openai import AsyncOpenAI
            for base_url in self._api_bases:
                self._clients.append(AsyncOpenAI(
                    base_url=base_url,
                    api_key="not-needed",
                ))
            print(f"[INFO] Assistant using {len(self._clients)} API backend(s): "
                  f"{', '.join(self._api_bases)}")
        client = self._clients[self._rr_counter % len(self._clients)]
        self._rr_counter += 1
        return client

    async def generate(self, messages: list[dict]) -> str:
        """Generate a response via the vLLM OpenAI-compatible API."""
        client = self._get_client()
        for attempt in range(self.num_retries):
            try:
                completion = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                return completion.choices[0].message.content or ""
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err:
                    wait = min(2 ** attempt, 60)
                    print(f"[WARN] Assistant rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"[ERROR] Assistant API error (attempt {attempt+1}): {e}")
                    if attempt == self.num_retries - 1:
                        raise
        return ""

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def start_sft_v3_prompt(self, messages: list[dict]) -> tuple[list[int], int]:
        """Build the exact first-turn prompt and response-mask length.

        Training counts replayed q0 as masked response-space tokens. Its
        incremental serialization is the chat template for a standalone user
        turn with the template's fixed prefix removed.
        """
        if self.output_mode not in ("sft_v3", "sft_v4", "sft_v5", "sft_v6", "sft_v7"):
            raise RuntimeError(
                "start_sft_v3_prompt called outside a controller-phased mode"
            )
        if self.output_mode == "sft_v7":
            messages = _ctom.action_prompt_messages(messages, getattr(self, "v7_action_protocol", "legacy"))
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        q0_ids = self._incremental_user_ids(messages[-1]["content"])
        return list(prompt_ids), len(q0_ids)

    def _incremental_user_ids(self, content: str) -> list[int]:
        token_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
        )
        return token_ids[len(self._system_prompt_tokens) :]

    def append_sft_v3_user_turn(
        self,
        prompt: list[int],
        response_tokens_used: int,
        content: str,
    ) -> tuple[list[int], int]:
        token_ids = self._incremental_user_ids(content)
        return prompt + token_ids, response_tokens_used + len(token_ids)


    # ── sft_v7 (the paper's method section) ────────────────────────────────
    # Mirrors CToMAgentLoop._run_v7_round / ._v7_anticipate. The round is
    # plan -> decision -> (draft/anticipate/verdict loop | ask/anticipate |
    # commit). Every internal draft and every anticipated user turn stays out
    # of the conversation the simulator sees; only `visible` enters it.
    # The parsers are IMPORTED from ctom.controller rather than reimplemented -- an
    # eval-side copy of the verdict markers drifting from training is exactly
    # what made the sft_v4/v5 eval r_pred score the wrong objective.
    def _v7_generation_prompt_ids(self) -> list[int]:
        """Token ids the chat template appends to open an assistant turn."""
        tok = self.tokenizer
        msgs = [{"role": "user", "content": "x"}]
        s0 = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        s1 = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if not s1.startswith(s0):
            raise RuntimeError("sft_v7: the chat template's generation prompt is not a suffix")
        return list(tok(s1[len(s0):], add_special_tokens=False)["input_ids"])

    def v7_slot_ids(self) -> tuple[list[int], list[int], list[int]]:
        """(header, close, gen_prompt) around a policy-written user turn.

        Derived from the template at runtime by diffing two renders, the same
        way training does, so neither side hard-codes the Qwen3 layout.
        """
        if self._v7_slot_cache is not None:
            return self._v7_slot_cache
        mk = lambda t: self._incremental_user_ids(f"{_ctom.ANTICIPATED_MARKER}\n{t}")  # noqa: E731
        a, b = mk("ZZZZ"), mk("QQQQ")
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        j = 0
        while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
            j += 1
        header, suffix = list(a[:i]), list(a[len(a) - j:])
        gen = self._v7_generation_prompt_ids()
        if not gen or len(gen) >= len(suffix) or suffix[-len(gen):] != gen:
            raise RuntimeError(
                "sft_v7: could not split the user-turn footer into close + generation "
                f"prompt (suffix={suffix}, gen={gen})"
            )
        close = suffix[: len(suffix) - len(gen)]
        self._v7_slot_cache = (header, close, list(gen))
        return self._v7_slot_cache

    async def v7_anticipate(
        self, prompt: list[int], response_tokens_used: int
    ) -> dict[str, Any]:
        """One policy-written user turn: header -> sampled prediction ->
        "\\n[PREDICTION_END]" + close. The caller appends the generation prompt
        if another assistant turn follows."""
        header, close, _gen = self.v7_slot_ids()
        end_tag = "\n" + _ctom.PREDICTION_END_TAG
        end_ids = self._encode_text(end_tag)

        def _append(ids):
            nonlocal prompt, response_tokens_used
            remaining = self.trajectory_response_length - response_tokens_used
            kept = list(ids[: max(remaining, 0)])
            prompt = prompt + kept
            response_tokens_used += len(kept)
            return len(ids) <= remaining

        header_ok = _append(header)
        remaining = self.trajectory_response_length - response_tokens_used
        budget = min(
            self.prediction_max_tokens,
            max(0, remaining - len(end_ids) - len(close) - 1),
        )
        raw, ids = await self._generate_completion_phase(
            prompt, budget, _ctom.PREDICTION_END_TAG
        )
        prompt = prompt + ids
        response_tokens_used += len(ids)
        clipped = budget > 0 and len(ids) >= budget
        text = _strip_stop(raw, _ctom.PREDICTION_END_TAG)
        tag_ok = _append(end_ids)
        close_ok = _append(close)
        return {
            "prompt": prompt,
            "response_tokens_used": response_tokens_used,
            "text": text,
            "generated_tokens": len(ids),
            "clipped": clipped,
            "valid": bool(text.strip()) and not clipped and header_ok and tag_ok and close_ok,
        }

    # ── sft_v6 / BDR (the retired BDR predecessor) ─────────────────────────────
    def bdr_internal_ids(self) -> list[int]:
        """The [INTERNAL] controller message as an incremental user turn — the
        same rendering CToMAgentLoop._bdr_internal_message_ids produces."""
        return self._incremental_user_ids(_markers.INTERNAL_MESSAGE)

    async def generate_bdr_turn(
        self,
        prompt: list[int],
        response_tokens_used: int,
        *,
        task_type: str,
        revise_allowed: bool,
    ) -> dict[str, Any]:
        """Mirror CToMAgentLoop._generate_bdr_turn: draft -> belief -> decision
        (-> ask), same budget arithmetic against the whole-dialogue ceiling."""
        draft_cap = self.pred_answer_max_tokens
        belief_cap = self.bdr_belief_max_tokens
        ask_cap = self.bdr_ask_max_tokens
        prefix = _markers.DRAFT_START_TAG + "\n"
        belief_open = "\n" + _markers.DRAFT_END_TAG + "\n" + _markers.BELIEF_START_TAG + "\n"
        decision_tag = _markers.DECISION_TAG if revise_allowed else _markers.DECISION_TAG_NO_REVISE
        decision_open = "\n" + _markers.BELIEF_END_TAG + "\n" + decision_tag + " "
        ask_open = "\n" + _markers.ASK_START_TAG + "\n"
        ask_close = "\n" + _markers.ASK_END_TAG
        prefix_ids = self._encode_text(prefix)
        belief_open_ids = self._encode_text(belief_open)
        decision_open_ids = self._encode_text(decision_open)
        ask_open_ids = self._encode_text(ask_open)
        ask_close_ids = self._encode_text(ask_close)
        nl_ids = self._encode_text("\n")
        ask_tail = len(ask_open_ids) + ask_cap + len(ask_close_ids)
        tail_after_draft = (
            len(belief_open_ids) + belief_cap + len(decision_open_ids)
            + _markers.DECISION_MAX_TOKENS + 1 + ask_tail
        )
        min_turn = len(prefix_ids) + len(belief_open_ids) + len(decision_open_ids) + _markers.DECISION_MAX_TOKENS + 3
        if self.trajectory_response_length - response_tokens_used < min_turn:
            return {"skipped": True, "prompt": prompt, "response_tokens_used": response_tokens_used}

        def _append(ids):
            nonlocal prompt, response_tokens_used
            remaining = self.trajectory_response_length - response_tokens_used
            kept = ids[: max(remaining, 0)]
            prompt = prompt + kept
            response_tokens_used += len(kept)
            return len(ids) <= remaining

        prefix_ok = _append(prefix_ids)
        remaining = self.trajectory_response_length - response_tokens_used
        draft_budget = min(draft_cap, max(0, remaining - tail_after_draft - 1))
        draft_raw, draft_ids = await self._generate_completion_phase(prompt, draft_budget, _markers.DRAFT_END_TAG)
        prompt = prompt + draft_ids
        response_tokens_used += len(draft_ids)
        draft_clipped = len(draft_ids) >= draft_budget
        draft = _strip_stop(draft_raw, _markers.DRAFT_END_TAG)

        belief_open_ok = _append(belief_open_ids)
        remaining = self.trajectory_response_length - response_tokens_used
        belief_budget = min(
            belief_cap,
            max(0, remaining - len(decision_open_ids) - _markers.DECISION_MAX_TOKENS - 1 - ask_tail),
        )
        belief_raw, belief_ids = await self._generate_completion_phase(prompt, belief_budget, _markers.BELIEF_END_TAG)
        prompt = prompt + belief_ids
        response_tokens_used += len(belief_ids)
        belief_clipped = len(belief_ids) >= belief_budget
        belief = _strip_stop(belief_raw, _markers.BELIEF_END_TAG)

        decision_open_ok = _append(decision_open_ids)
        decision_raw, decision_ids = await self._generate_completion_phase(
            prompt, _markers.DECISION_MAX_TOKENS, "\n"
        )
        prompt = prompt + decision_ids
        response_tokens_used += len(decision_ids)
        parsed = _markers.parse_decision(decision_raw)
        decision, coerce_reason = _markers.coerce_decision(
            parsed, task_type=task_type, revise_allowed=revise_allowed
        )
        nl_ok = _append(nl_ids)

        ask_text, ask_clipped, ask_ok = "", False, True
        if decision == _markers.ASK:
            ask_open_ok = _append(ask_open_ids)
            remaining = self.trajectory_response_length - response_tokens_used
            ask_budget = min(ask_cap, max(0, remaining - len(ask_close_ids) - 1))
            ask_raw, ask_ids = await self._generate_completion_phase(prompt, ask_budget, _markers.ASK_END_TAG)
            prompt = prompt + ask_ids
            response_tokens_used += len(ask_ids)
            ask_clipped = len(ask_ids) >= ask_budget
            ask_close_ok = _append(ask_close_ids)
            ask_ok = ask_open_ok and ask_close_ok
            ask_text = _strip_stop(ask_raw, _markers.ASK_END_TAG)
            if not ask_text.strip():
                decision, coerce_reason = _markers.COMMIT, "empty_ask"

        validity = {
            "draft_valid": bool(draft.strip()) and not draft_clipped and prefix_ok,
            "belief_valid": (
                bool(belief.strip()) and not belief_clipped and belief_open_ok
                and bool(_markers.parse_belief(belief, task_type))
            ),
            "decision_valid": parsed is not None and decision_open_ok and nl_ok,
            "ask_valid": (decision != _markers.ASK) or (bool(ask_text.strip()) and not ask_clipped and ask_ok),
        }
        if coerce_reason == "empty_ask":
            validity["ask_valid"] = False
        validity["invalid_count"] = sum(1 for v in validity.values() if v is False)
        return {
            "skipped": False,
            "prompt": prompt,
            "response_tokens_used": response_tokens_used,
            "draft": draft,
            "belief": belief,
            "decision": decision,
            "decision_raw": decision_raw,
            "coerce_reason": coerce_reason,
            "ask": ask_text if decision == _markers.ASK else "",
            "validity": validity,
            "draft_clipped": draft_clipped,
            "belief_clipped": belief_clipped,
            "ask_clipped": ask_clipped,
        }

    async def _generate_completion_phase(
        self,
        prompt: list[int],
        max_tokens: int,
        stop_text: str,
        allowed_token_ids: Optional[list[int]] = None,
    ) -> tuple[str, list[int]]:
        if max_tokens <= 0:
            return "", []
        client = self._get_client()
        for attempt in range(self.num_retries):
            try:
                completion = await client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    top_p=1.0 if allowed_token_ids is not None else self.top_p,
                    max_tokens=max_tokens,
                    stop=[stop_text],
                    extra_body={
                        "add_special_tokens": False,
                        "return_token_ids": True,
                        "top_k": -1,
                        **(_ctom.categorical_sampling_params({}, allowed_token_ids)
                           if allowed_token_ids is not None else {}),
                    },
                )
                choice = completion.choices[0]
                text = choice.text or ""
                token_ids = getattr(choice, "token_ids", None)
                if token_ids is None:
                    model_extra = getattr(choice, "model_extra", None) or {}
                    token_ids = model_extra.get("token_ids")
                usage = getattr(completion, "usage", None)
                n_tokens = getattr(usage, "completion_tokens", None)
                if token_ids is None:
                    token_ids = self._encode_text(text)
                token_ids = list(token_ids)
                if n_tokens is not None and int(n_tokens) != len(token_ids):
                    raise RuntimeError(
                        "vLLM returned inconsistent sft_v3 completion token counts: "
                        f"usage={n_tokens}, token_ids={len(token_ids)}"
                    )
                return text, token_ids
            except Exception as e:
                print(
                    f"[ERROR] sft_v3 assistant phase error "
                    f"(attempt {attempt + 1}/{self.num_retries}): {e}"
                )
                if attempt == self.num_retries - 1:
                    raise
                await asyncio.sleep(min(2 ** attempt, 15))
        return "", []

    async def generate_sft_v4_turn(
        self,
        prompt: list[int],
        response_tokens_used: int,
    ) -> dict[str, Any]:
        """Mirror CToMAgentLoop._generate_sft_v4_turn at evaluation time.

        draft -> prediction -> revision, with the SAME budget arithmetic as
        training: each phase reserves room for every delimiter and phase that
        must still fit after it, and trajectory_response_length is the whole-
        dialogue ceiling (not a per-turn allowance).

        The returned "answer" is the REVISION, because that is the only part the
        user sees and the only part the task reward scores. The draft is returned
        separately so transcripts can show what the revision changed.
        """
        prefix = SFT_V3_ANSWER_START_TAG + "\n"
        pred_open = (
            "\n" + SFT_V3_ANSWER_END_TAG + "\n" + SFT_V3_PREDICTION_START_TAG + "\n"
        )
        rev_open = (
            "\n" + SFT_V3_PREDICTION_END_TAG + "\n" + SFT_V4_REVISION_START_TAG + "\n"
        )
        suffix = "\n" + SFT_V4_REVISION_END_TAG
        prefix_ids = self._encode_text(prefix)
        pred_open_ids = self._encode_text(pred_open)
        rev_open_ids = self._encode_text(rev_open)
        suffix_ids = self._encode_text(suffix)

        tail_after_draft = (
            len(pred_open_ids)
            + len(rev_open_ids)
            + len(suffix_ids)
            + self.prediction_max_tokens
            + self.pred_revision_max_tokens
        )

        remaining = self.trajectory_response_length - response_tokens_used
        prefix_complete = len(prefix_ids) <= remaining
        prefix_kept = prefix_ids[: max(remaining, 0)]
        prompt += prefix_kept
        response_tokens_used += len(prefix_kept)

        # ── Phase 1: draft (internal) ──
        remaining = self.trajectory_response_length - response_tokens_used
        draft_budget = min(
            self.pred_answer_max_tokens,
            max(0, remaining - tail_after_draft - 1),
        )
        draft_raw, draft_ids = await self._generate_completion_phase(
            prompt,
            draft_budget,
            SFT_V3_ANSWER_END_TAG,
        )
        prompt += draft_ids
        response_tokens_used += len(draft_ids)
        draft_clipped = len(draft_ids) >= draft_budget

        remaining = self.trajectory_response_length - response_tokens_used
        pred_open_complete = len(pred_open_ids) <= remaining
        pred_open_kept = pred_open_ids[: max(remaining, 0)]
        prompt += pred_open_kept
        response_tokens_used += len(pred_open_kept)

        # ── Phase 2: prediction ──
        remaining = self.trajectory_response_length - response_tokens_used
        prediction_budget = min(
            self.prediction_max_tokens,
            max(
                0,
                remaining
                - len(rev_open_ids)
                - len(suffix_ids)
                - self.pred_revision_max_tokens,
            ),
        )
        prediction_raw, prediction_ids = await self._generate_completion_phase(
            prompt,
            prediction_budget,
            SFT_V3_PREDICTION_END_TAG,
        )
        prompt += prediction_ids
        response_tokens_used += len(prediction_ids)
        prediction_clipped = len(prediction_ids) >= prediction_budget

        remaining = self.trajectory_response_length - response_tokens_used
        rev_open_complete = len(rev_open_ids) <= remaining
        rev_open_kept = rev_open_ids[: max(remaining, 0)]
        prompt += rev_open_kept
        response_tokens_used += len(rev_open_kept)

        # ── Phase 3: revision (the visible answer) ──
        # Carry the draft's submission decision into the revision — the eval-side
        # half of the same fix in CToMAgentLoop._generate_sft_v4_turn. Training
        # and eval must decode identically or the submit rate measured here stops
        # describing the policy that was trained.
        # getattr, not attribute access: this class is also constructed by paths
        # that bypass __init__ (see tests/test_eval_parity.py), and a missing
        # attribute here would only surface inside a 3-node eval job.
        forced_marker = (
            forced_submit_marker(draft_raw.strip())
            if getattr(self, "submit_marker_carryover", False) else ""
        )
        marker_complete = True
        if forced_marker:
            marker_ids = self._encode_text(forced_marker)
            remaining = self.trajectory_response_length - response_tokens_used
            marker_complete = len(marker_ids) <= remaining
            marker_kept = marker_ids[: max(remaining, 0)]
            prompt += marker_kept
            response_tokens_used += len(marker_kept)

        remaining = self.trajectory_response_length - response_tokens_used
        revision_budget = min(
            self.pred_revision_max_tokens,
            max(0, remaining - len(suffix_ids) - 1),
        )
        revision_raw, revision_ids = await self._generate_completion_phase(
            prompt,
            revision_budget,
            SFT_V4_REVISION_END_TAG,
        )
        prompt += revision_ids
        response_tokens_used += len(revision_ids)
        revision_clipped = len(revision_ids) >= revision_budget

        remaining = self.trajectory_response_length - response_tokens_used
        suffix_complete = len(suffix_ids) <= remaining
        suffix_kept = suffix_ids[: max(remaining, 0)]
        prompt += suffix_kept
        response_tokens_used += len(suffix_kept)

        draft = draft_raw.strip()
        prediction = prediction_raw.strip()
        revision = revision_raw.strip()
        # The marker is controller text, so the sampled string does not carry it.
        revision = compose_revision(forced_marker, revision)
        return {
            "prompt": prompt,
            "response_tokens_used": response_tokens_used,
            # Same fallback as the training controller: an empty revision keeps
            # the draft rather than losing the turn; phase_validity records it.
            "answer": revision or draft,
            "draft": draft,
            "prediction": prediction,
            "phase_validity": {
                "answer_clipped": draft_clipped,
                "prediction_clipped": prediction_clipped,
                "revision_clipped": revision_clipped,
                "answer_valid": bool(draft) and not draft_clipped and prefix_complete,
                "prediction_valid": (
                    bool(prediction)
                    and not prediction_clipped
                    and pred_open_complete
                ),
                "revision_valid": (
                    bool(revision)
                    and not revision_clipped
                    and rev_open_complete
                    and marker_complete
                    and suffix_complete
                ),
                "submit_marker_forced": bool(forced_marker),
                "answer_tokens": len(draft_ids),
                "prediction_tokens": len(prediction_ids),
                "revision_tokens": len(revision_ids),
            },
        }

    async def generate_sft_v3_turn(
        self,
        prompt: list[int],
        response_tokens_used: int,
    ) -> dict[str, Any]:
        """Mirror CToMAgentLoop._generate_sft_v3_turn at evaluation time."""
        prefix = SFT_V3_ANSWER_START_TAG + "\n"
        middle = (
            "\n"
            + SFT_V3_ANSWER_END_TAG
            + "\n"
            + SFT_V3_PREDICTION_START_TAG
            + "\n"
        )
        suffix = "\n" + SFT_V3_PREDICTION_END_TAG
        prefix_ids = self._encode_text(prefix)
        middle_ids = self._encode_text(middle)
        suffix_ids = self._encode_text(suffix)

        remaining = self.trajectory_response_length - response_tokens_used
        prefix_complete = len(prefix_ids) <= remaining
        prefix_kept = prefix_ids[: max(remaining, 0)]
        prompt += prefix_kept
        response_tokens_used += len(prefix_kept)

        remaining = self.trajectory_response_length - response_tokens_used
        answer_budget = min(
            self.pred_answer_max_tokens,
            max(0, remaining - len(middle_ids) - len(suffix_ids) - 1),
        )
        answer_raw, answer_ids = await self._generate_completion_phase(
            prompt,
            answer_budget,
            SFT_V3_ANSWER_END_TAG,
        )
        prompt += answer_ids
        response_tokens_used += len(answer_ids)
        answer_clipped = len(answer_ids) >= answer_budget

        remaining = self.trajectory_response_length - response_tokens_used
        middle_complete = len(middle_ids) <= remaining
        middle_kept = middle_ids[: max(remaining, 0)]
        prompt += middle_kept
        response_tokens_used += len(middle_kept)

        remaining = self.trajectory_response_length - response_tokens_used
        prediction_budget = min(
            self.prediction_max_tokens,
            max(0, remaining - len(suffix_ids)),
        )
        prediction_raw, prediction_ids = await self._generate_completion_phase(
            prompt,
            prediction_budget,
            SFT_V3_PREDICTION_END_TAG,
        )
        prompt += prediction_ids
        response_tokens_used += len(prediction_ids)
        prediction_clipped = len(prediction_ids) >= prediction_budget

        remaining = self.trajectory_response_length - response_tokens_used
        suffix_complete = len(suffix_ids) <= remaining
        suffix_kept = suffix_ids[: max(remaining, 0)]
        prompt += suffix_kept
        response_tokens_used += len(suffix_kept)

        answer = answer_raw.strip()
        prediction = prediction_raw.strip()
        return {
            "prompt": prompt,
            "response_tokens_used": response_tokens_used,
            "answer": answer,
            "prediction": prediction,
            "phase_validity": {
                "answer_clipped": answer_clipped,
                "prediction_clipped": prediction_clipped,
                "answer_valid": bool(answer) and not answer_clipped and prefix_complete,
                "prediction_valid": (
                    bool(prediction)
                    and not prediction_clipped
                    and middle_complete
                    and suffix_complete
                ),
                "answer_tokens": len(answer_ids),
                "prediction_tokens": len(prediction_ids),
            },
        }


# ============================================================================
# 5. Multi-Turn Rollout Engine
# ============================================================================

def _system_content_from_sample(sample: dict) -> Optional[str]:
    """Extract the baked-in assistant system prompt from a sample's parquet
    `prompt` column (list of role/content dicts). Returns None if absent."""
    prompt = sample.get("system_prompt")
    if not prompt:
        return None
    try:
        for m in list(prompt):
            if isinstance(m, dict) and m.get("role") == "system":
                return m.get("content")
    except TypeError:
        return None
    return None


async def run_single_rollout(
    sample: dict,
    assistant,
    user_sim: UserSimulator,
    max_turns: int = 5,
    verbose: bool = False,
    q0_cache: Optional[dict] = None,
    max_context_tokens: int = 0,
    use_data_system_prompt: bool = False,
    replay_q0: bool = False,
    effort_max_turns: int = 8,
    effort_max_user_tokens: int = 256,
    effort_alpha: float = 0.5,
    effort_max_assistant_tokens: int = 0,
    effort_beta: float = 0.0,
    effort_token_units: str = "words",
    effort_mode: str = "bonus",
    invalid_phase_penalty: float = -1.0,
    reward_w_task: float = 0.5,
    reward_w_pred: float = 0.3,
    reward_w_effort: float = 0.2,
) -> dict:
    """
    Run a full multi-turn rollout for a single problem.

    Replicates the training-time agent loop:
      1. User sim generates q0
      2. Assistant generates (a0, e0)
      3. Strip prediction, feed a0 to user sim → q1
      4. Repeat until max turns or termination

    assistant can be either:
      - AssistantModel (sync, local vLLM) — used with concurrency=1
      - AsyncAssistantClient (async, API) — used with concurrent workers

    If q0_cache is provided, it is used to load/store q0 messages keyed by
    sample["index"] so that the same opening message is used across eval runs.
    """
    is_async = isinstance(assistant, AsyncAssistantClient)
    # Default: the prediction-format SYSTEM_PROMPT. With --assistant_system_prompt_from_data,
    # use the prompt baked into the eval parquet instead — so a model trained with a
    # different contract (e.g. the outcome-only baseline's no-prediction, raw-\boxed{}
    # prompt) is evaluated under the instructions it was actually trained on.
    sys_content = SYSTEM_PROMPT
    if use_data_system_prompt:
        sys_content = _system_content_from_sample(sample) or SYSTEM_PROMPT
    messages = [{"role": "system", "content": sys_content}]
    predictions = []
    phase_validity = []
    drafts = []  # sft_v4 only: the internal draft behind each revision
    bdr_rounds = []  # sft_v6 only: one record per visible turn
    v7_rounds = []   # sft_v7 only: one record per visible turn
    assistant_turns = 0
    user_turns = 0
    error = None
    sample_key = str(sample["index"])

    try:
        # Step 1: User sim generates q0 (initial question)
        # Use cached q0 if available for reproducibility across eval runs
        if replay_q0:
            # Match training-time replay_q0=true: the problem statement itself
            # is the opening user message (deterministic, no sim / no cache).
            q0 = sample["single_turn_prompt"]
        elif q0_cache is not None and sample_key in q0_cache:
            q0 = q0_cache[sample_key]
        else:
            q0, should_terminate, sim_info = await user_sim.generate_user_turn(
                messages=messages,
                problem=sample["single_turn_prompt"],
                ground_truth=sample["ground_truth"],
                expertise=sample["expertise"],
                goal=sample["goal"],
            )
            if q0_cache is not None:
                q0_cache[sample_key] = q0
        if not q0:
            return _make_result(
                sample,
                messages,
                predictions,
                "user_sim_failed_q0",
                phase_validity=phase_validity,
                drafts=drafts,
                bdr_rounds=bdr_rounds,
                v7_rounds=v7_rounds,
                effort_tokenizer=getattr(assistant, "tokenizer", None),
                sft_v3_mode=getattr(assistant, "output_mode", "")
                in ("sft_v3", "sft_v4", "sft_v5", "sft_v7"),
                effort_max_turns=effort_max_turns,
                effort_max_user_tokens=effort_max_user_tokens,
                effort_alpha=effort_alpha,
                effort_max_assistant_tokens=effort_max_assistant_tokens,
                effort_beta=effort_beta,
                effort_token_units=effort_token_units,
                effort_mode=effort_mode,
                invalid_phase_penalty=invalid_phase_penalty,
                reward_w_task=reward_w_task,
                reward_w_pred=reward_w_pred,
                reward_w_effort=reward_w_effort,
            )

        messages.append({"role": "user", "content": q0})
        user_turns += 1
        assistant_mode = getattr(assistant, "output_mode", "")
        sft_v3_mode = assistant_mode in ("sft_v3", "sft_v4", "sft_v5", "sft_v6", "sft_v7")
        # sft_v5 differs from sft_v4 only in how the prediction is SUPERVISED at
        # training time; the decode is the same three controller phases.
        sft_v4_mode = assistant_mode in ("sft_v4", "sft_v5")
        # sft_v6 (BDR): draft/belief/decision with internal REVISE turns; only
        # the COMMIT/SHOW/ASK message enters the conversation.
        bdr_mode = assistant_mode == "sft_v6"
        # sft_v7: plan/decision with an anticipated user turn and internal
        # revisions; only the DRAFT/ASK/COMMIT message enters the conversation.
        v7_mode = assistant_mode == "sft_v7"
        v7_last_shown = ""
        v7_terminated = False
        bdr_task_type = str(sample.get("task_type") or "math")
        if sft_v3_mode:
            model_prompt, response_tokens_used = assistant.start_sft_v3_prompt(messages)
        if verbose:
            print(f"  [User q0]: {q0[:120]}...")

        # Step 2+: Multi-turn loop
        for turn_idx in range(max_turns):
            # sft_v3 is not a free-form chat completion. It uses the same
            # controller-owned answer and prediction calls as training.
            if v7_mode:
                gen = await run_v7_round(
                    assistant, model_prompt, response_tokens_used,
                    task_type=bdr_task_type, last_shown=v7_last_shown,
                )
                model_prompt = gen["prompt"]
                response_tokens_used = gen["response_tokens_used"]
                rec = gen["record"]
                if rec is None:          # no budget left for a whole round
                    break
                v7_last_shown = gen["last_shown"]
                v7_terminated = gen["terminated"]
                answer = gen["visible"]
                # NB: this is scored against the NEXT REAL user turn by
                # compute_f1_prediction_reward, but v7 conditions its
                # prediction on an INTERNAL draft the simulator never saw --
                # the same off-objective caveat that applies to sft_v4/v5.
                # Read r_pred for v7 as diagnostic only.
                prediction = rec["predictions"][-1] if rec["predictions"] else ""
                phase_validity.append(
                    dict(rec["validity"], invalid_count=rec["invalid_count"])
                )
                drafts.append(rec["draft"])
                v7_rounds.append(rec)
            elif bdr_mode:
                gen = await run_bdr_round(
                    assistant, model_prompt, response_tokens_used, task_type=bdr_task_type
                )
                model_prompt = gen["prompt"]
                response_tokens_used = gen["response_tokens_used"]
                if gen["terminated"]:
                    break
                answer = gen["visible"]
                prediction = ""
                rec = gen["record"]
                phase_validity.append(dict(rec["validity"], invalid_count=rec["invalid_count"]))
                drafts.append(rec["draft"])
                rec = dict(rec)
                rec["internal_trace"] = gen["internal_trace"]
                bdr_rounds.append(rec)
            elif sft_v3_mode:
                if sft_v4_mode:
                    generated = await assistant.generate_sft_v4_turn(
                        model_prompt,
                        response_tokens_used,
                    )
                else:
                    generated = await assistant.generate_sft_v3_turn(
                        model_prompt,
                        response_tokens_used,
                    )
                model_prompt = generated["prompt"]
                response_tokens_used = generated["response_tokens_used"]
                answer = generated["answer"]
                prediction = generated["prediction"]
                phase_validity.append(generated["phase_validity"])
                if "draft" in generated:
                    drafts.append(generated["draft"])
            else:
                ctx = truncate_messages(messages, max_context_tokens)
                if is_async:
                    raw_response = await assistant.generate(ctx)
                else:
                    raw_response = assistant.generate(ctx)
                answer, prediction = parse_answer_and_prediction(raw_response)
            predictions.append(prediction)
            assistant_turns += 1

            # Store the CLEAN answer (without prediction) in conversation
            messages.append({"role": "assistant", "content": answer})
            if verbose:
                print(f"  [Assistant a{turn_idx}]: {answer[:120]}...")
                if prediction:
                    print(f"  [Prediction e{turn_idx}]: {prediction[:80]}...")

            # Check if this is the last allowed turn
            if turn_idx >= max_turns - 1:
                break

            # sft_v7 COMMIT returns AgentState.TERMINATED in training.
            if v7_mode and v7_terminated:
                break

            # User sim generates next turn (truncate context for sim prompt)
            user_response, should_terminate, sim_info = await user_sim.generate_user_turn(
                messages=truncate_messages(messages, max_context_tokens),
                problem=sample["single_turn_prompt"],
                ground_truth=sample["ground_truth"],
                expertise=sample["expertise"],
                goal=sample["goal"],
            )

            if not user_response or should_terminate:
                if user_response:
                    messages.append({"role": "user", "content": user_response})
                    user_turns += 1
                break

            messages.append({"role": "user", "content": user_response})
            user_turns += 1
            if sft_v3_mode:
                model_prompt, response_tokens_used = assistant.append_sft_v3_user_turn(
                    model_prompt,
                    response_tokens_used,
                    user_response,
                )
            if verbose:
                print(f"  [User q{turn_idx+1}]: {user_response[:120]}...")

        # After the loop ends, if the last message is from the assistant
        # (i.e., the assistant terminated or we hit max_turns), generate one
        # final user response so the reward function can:
        # 1. Detect [[TERMINATE CHAT]] / <understood> signals for R_task
        # 2. Compare the last prediction against actual user response for R_pred
        if messages and messages[-1].get("role") == "assistant":
            try:
                final_response, final_terminate, final_info = await user_sim.generate_user_turn(
                    messages=truncate_messages(messages, max_context_tokens),
                    problem=sample["single_turn_prompt"],
                    ground_truth=sample["ground_truth"],
                    expertise=sample["expertise"],
                    goal=sample["goal"],
                    # Labels the last prediction slot for R_pred. Training exempts
                    # counterfactual calls from the gate; without this the gated
                    # run loses R_pred on exactly the episodes it solves.
                    counterfactual=True,
                )
                if final_response and final_response.strip():
                    messages.append({"role": "user", "content": final_response})
                    user_turns += 1
                    if verbose:
                        print(f"  [User final]: {final_response[:120]}...")
            except Exception as e:
                if verbose:
                    print(f"  [WARN] Failed to generate final user response: {e}")

    except Exception as e:
        error = str(e)
        traceback.print_exc()

    return _make_result(
        sample,
        messages,
        predictions,
        error,
        phase_validity=phase_validity,
        drafts=drafts,
        bdr_rounds=bdr_rounds,
        v7_rounds=v7_rounds,
        effort_tokenizer=getattr(assistant, "tokenizer", None),
        # NB: "sft_v6" is deliberately absent -- that is a PRE-EXISTING bug
        # (BDR evals score effort conversation-wise while BDR trains turn-wise);
        # left as found so this change touches only sft_v7.
        sft_v3_mode=getattr(assistant, "output_mode", "") in ("sft_v3", "sft_v4", "sft_v5", "sft_v7"),
        effort_max_turns=effort_max_turns,
        effort_max_user_tokens=effort_max_user_tokens,
        effort_alpha=effort_alpha,
        effort_max_assistant_tokens=effort_max_assistant_tokens,
        effort_beta=effort_beta,
        effort_token_units=effort_token_units,
        effort_mode=effort_mode,
        invalid_phase_penalty=invalid_phase_penalty,
        reward_w_task=reward_w_task,
        reward_w_pred=reward_w_pred,
        reward_w_effort=reward_w_effort,
    )




def _strip_stop(text: str, stop_text: str) -> str:
    """vLLM trims a matched stop string from the returned text; a stubbed
    backend may not. Strip once, defensively, so both paths agree."""
    text = (text or "").strip()
    if stop_text and text.endswith(stop_text):
        text = text[: -len(stop_text)].rstrip()
    return text


async def run_bdr_round(
    assistant,
    prompt: list[int],
    response_tokens_used: int,
    *,
    task_type: str,
) -> dict[str, Any]:
    """Eval mirror of CToMAgentLoop._run_bdr_round: internal REVISE turns until
    COMMIT / SHOW / ASK or the internal cap, then the visible message.

    Returns {"prompt", "response_tokens_used", "visible", "record",
    "internal_trace", "terminated"}; `record` has the same keys as the
    training-side round record so ctom.markers.compute_voi can score eval
    transcripts, and `internal_trace` keeps every internal turn for inspection
    (they never enter the conversation the simulator sees).
    """
    max_internal = int(getattr(assistant, "bdr_max_internal_revisions", 1))
    internal = 0
    invalid_count = 0
    first_draft = None
    trace: list[dict] = []
    last: dict[str, Any] = {}
    turn: dict[str, Any] = {}
    forced = ""
    while True:
        revise_allowed = internal < max_internal
        turn = await assistant.generate_bdr_turn(
            prompt, response_tokens_used, task_type=task_type, revise_allowed=revise_allowed
        )
        if turn.get("skipped"):
            if not last:
                return {
                    "prompt": prompt, "response_tokens_used": response_tokens_used,
                    "visible": "", "record": None, "internal_trace": trace, "terminated": True,
                }
            turn = dict(last)
            turn["decision"], turn["ask"] = _markers.COMMIT, ""
            forced = "budget_commit"
            break
        prompt = turn["prompt"]
        response_tokens_used = turn["response_tokens_used"]
        last = turn
        invalid_count += int(turn["validity"]["invalid_count"])
        if first_draft is None:
            first_draft = turn["draft"]
        trace.append({k: turn[k] for k in ("draft", "belief", "decision", "decision_raw", "coerce_reason", "ask", "validity")})
        if turn["decision"] != _markers.REVISE:
            break
        internal += 1
        ids = assistant.bdr_internal_ids()
        if response_tokens_used + len(ids) + 16 > assistant.trajectory_response_length:
            internal -= 1
            turn = dict(turn)
            turn["decision"], turn["ask"] = _markers.COMMIT, ""
            forced = "budget_commit"
            break
        prompt = prompt + ids
        response_tokens_used += len(ids)
    visible = _markers.visible_message(turn["decision"], turn["draft"], turn["ask"], task_type)
    record = {
        "draft": turn["draft"],
        "first_draft": first_draft if first_draft is not None else turn["draft"],
        "belief": turn["belief"],
        "decision": turn["decision"],
        "decision_raw": turn.get("decision_raw", ""),
        "coerce_reason": forced or turn.get("coerce_reason", ""),
        "ask": turn["ask"],
        "visible": visible,
        "internal_turns": internal,
        "invalid_count": invalid_count,
        "validity": dict(turn["validity"]),
    }
    return {
        "prompt": prompt, "response_tokens_used": response_tokens_used,
        "visible": visible, "record": record, "internal_trace": trace, "terminated": False,
    }


async def run_v7_round(
    assistant,
    prompt: list[int],
    response_tokens_used: int,
    *,
    task_type: str,
    last_shown: str,
) -> dict[str, Any]:
    """Eval mirror of CToMAgentLoop._run_v7_round.

    plan -> decision -> (draft/anticipate/verdict loop | ask/anticipate | commit),
    with the same budget arithmetic against the whole-dialogue ceiling. Internal
    drafts and anticipated user turns never enter the conversation; only
    `visible` does, exactly as `ctom/controller.py:sim_view` arranges in training.

    Returns {"prompt", "response_tokens_used", "visible", "record", "terminated",
    "last_shown"}.
    """
    enc = assistant._encode_text
    header, close, gen = assistant.v7_slot_ids()
    think_cap = int(assistant.v7_think_max_tokens)
    draft_cap = int(assistant.pred_answer_max_tokens)
    ask_cap = int(assistant.v7_ask_max_tokens)
    max_inner = int(assistant.v7_max_inner_revisions)
    protocol = _ctom.validate_action_protocol(getattr(assistant, "v7_action_protocol", "legacy"))
    categorical = protocol == "categorical_v1"
    alphabet = _ctom.action_token_ids(enc) if categorical else []

    think_open = _ctom.THINK_START_TAG + "\n"
    decision_open = "\n" + _ctom.THINK_END_TAG + "\n" + _ctom.DECISION_TAG + " "
    draft_open = _ctom.DRAFT_START_TAG + "\n"
    draft_close = "\n" + _ctom.DRAFT_END_TAG
    ask_open = _ctom.ASK_START_TAG + "\n"
    ask_close = "\n" + _ctom.ASK_END_TAG

    pred_tail = len(header) + len(enc("\n" + _ctom.PREDICTION_END_TAG)) + len(close) + 4
    draft_block_min = len(enc(draft_open)) + len(enc(draft_close)) + pred_tail + 8
    min_turn = (
        len(enc(think_open)) + len(enc(decision_open))
        + (1 if categorical else _ctom.DECISION_MAX_TOKENS) + 2 + draft_block_min
    )
    if assistant.trajectory_response_length - response_tokens_used < min_turn:
        return {
            "prompt": prompt, "response_tokens_used": response_tokens_used,
            "visible": "", "record": None, "terminated": True, "last_shown": last_shown,
        }

    def _append(ids):
        nonlocal prompt, response_tokens_used
        remaining = assistant.trajectory_response_length - response_tokens_used
        kept = list(ids[: max(remaining, 0)])
        prompt = prompt + kept
        response_tokens_used += len(kept)
        return len(ids) <= remaining

    generated_tokens = 0

    async def _phase(max_tokens, stop_text, allowed_ids=None):
        nonlocal prompt, response_tokens_used, generated_tokens
        kwargs = {"allowed_token_ids": allowed_ids} if allowed_ids is not None else {}
        raw, ids = await assistant._generate_completion_phase(prompt, max_tokens, stop_text, **kwargs)
        if allowed_ids is not None:
            _ctom.parse_action_token(ids, alphabet, bool(last_shown))
        prompt = prompt + ids
        response_tokens_used += len(ids)
        generated_tokens += len(ids)
        return _strip_stop(raw, stop_text), raw, len(ids) >= max_tokens

    validity: dict[str, Any] = {}
    counters = {
        "inner_iters": 0, "draft_changed": 0, "inner_cap_hit": 0,
        "verdict_satisfied": 0, "verdict_unsatisfied": 0, "verdict_idiom": 0,
        "think_clipped": 0, "draft_clipped": 0, "ask_clipped": 0,
        "prediction_clipped": 0, "prediction_invalid": 0, "budget_stop": 0,
    }

    # ── plan ──
    ok_think = _append(enc(think_open))
    remaining = assistant.trajectory_response_length - response_tokens_used
    think_budget = min(think_cap, max(0, remaining - (min_turn - len(enc(think_open)))))
    think, _raw, think_clipped = await _phase(think_budget, _ctom.THINK_END_TAG)
    validity["think_valid"] = bool(think.strip()) and not think_clipped and ok_think
    counters["think_clipped"] += int(think_clipped)

    # ── decision ──
    ok_dec = _append(enc(decision_open))
    action_offset = response_tokens_used
    _dec_txt, decision_raw, _ = await _phase(
        1 if categorical else _ctom.DECISION_MAX_TOKENS, "\n",
        (alphabet if last_shown else alphabet[:2]) if categorical else None,
    )
    parsed = _ctom.parse_action_token(prompt[-1:], alphabet, bool(last_shown)) if categorical else _ctom.parse_decision(decision_raw)
    decision, coerce_reason = _ctom.coerce_decision(parsed, has_shown_draft=bool(last_shown))
    nl_ok = _append(enc("\n"))
    validity["decision_valid"] = parsed is not None and ok_dec and nl_ok

    predictions: list[str] = []
    drafts_all: list[str] = []
    verdicts: list[str] = []
    verdict_reasons: list[str] = []
    ask_text = ""
    visible = ""
    new_last_shown = last_shown

    if decision == _ctom.COMMIT:
        visible = _ctom.visible_message(_ctom.COMMIT, last_shown=last_shown, task_type=task_type)

    if decision == _ctom.ASK:
        ok_open = _append(enc(ask_open))
        remaining = assistant.trajectory_response_length - response_tokens_used
        ask_budget = min(ask_cap, max(0, remaining - len(enc(ask_close)) - (0 if categorical else pred_tail) - 1))
        ask_text, _raw, ask_clipped = await _phase(ask_budget, _ctom.ASK_END_TAG)
        ok_close = _append(enc(ask_close))
        counters["ask_clipped"] += int(ask_clipped)
        if not ask_text.strip():
            # Nothing to send: fall through to a draft rather than an empty message.
            validity["ask_valid"] = False
            decision = _ctom.DRAFT
            coerce_reason = "empty_ask"
        elif categorical:
            visible = ask_text.strip()
            validity["ask_valid"] = (not ask_clipped) and ok_open and ok_close
        else:
            ant = await assistant.v7_anticipate(prompt, response_tokens_used)
            prompt, response_tokens_used = ant["prompt"], ant["response_tokens_used"]
            generated_tokens += ant["generated_tokens"]
            predictions.append(ant["text"])
            counters["prediction_clipped"] += int(ant["clipped"])
            counters["prediction_invalid"] += int(not ant["valid"])
            visible = _ctom.visible_message(
                _ctom.ASK, ask=ask_text, prediction=ant["text"], task_type=task_type
            )
            validity["ask_valid"] = (not ask_clipped) and ok_open and ok_close
            validity["prediction_valid"] = ant["valid"]

    if decision == _ctom.DRAFT:
        j = 0
        last_draft: Optional[str] = None
        draft = ""
        while True:
            ok_open = _append(enc(draft_open))
            remaining = assistant.trajectory_response_length - response_tokens_used
            draft_budget = min(
                draft_cap, max(0, remaining - len(enc(draft_close)) - pred_tail - 1)
            )
            draft, _raw, draft_clipped = await _phase(draft_budget, _ctom.DRAFT_END_TAG)
            ok_close = _append(enc(draft_close))
            ant = await assistant.v7_anticipate(prompt, response_tokens_used)
            prompt, response_tokens_used = ant["prompt"], ant["response_tokens_used"]
            # Mirrors the training loop: a request for a box the draft already
            # carries is not pushback (ctom/controller.py:is_format_only_request).
            verdict, reason = _ctom.parse_verdict(ant["text"], task_type, draft=draft)
            generated_tokens += ant["generated_tokens"]
            drafts_all.append(draft)
            predictions.append(ant["text"])
            verdicts.append(verdict)
            verdict_reasons.append(reason)
            draft_ok = bool(draft.strip()) and not draft_clipped and ok_open and ok_close
            validity["draft_valid"] = validity.get("draft_valid", True) and draft_ok
            validity["prediction_valid"] = validity.get("prediction_valid", True) and ant["valid"]
            counters["draft_clipped"] += int(draft_clipped)
            counters["prediction_clipped"] += int(ant["clipped"])
            counters["prediction_invalid"] += int(not ant["valid"])
            counters[f"verdict_{verdict.lower()}"] += 1
            if reason == "idiom":
                counters["verdict_idiom"] += 1
            if last_draft is not None and draft.strip() != last_draft.strip():
                counters["draft_changed"] += 1
            last_draft = draft
            remaining = assistant.trajectory_response_length - response_tokens_used
            if verdict == _ctom.UNSATISFIED:
                if j >= max_inner:
                    counters["inner_cap_hit"] += 1
                elif remaining < len(gen) + draft_block_min:
                    counters["budget_stop"] += 1
                else:
                    # This draft stays internal; open a NEW assistant turn.
                    _append(gen)
                    j += 1
                    counters["inner_iters"] += 1
                    continue
            break
        visible = _ctom.visible_message(_ctom.DRAFT, draft=draft, task_type=task_type)
        new_last_shown = draft

    validity["invalid_count"] = sum(
        1 for k, v in validity.items() if k.endswith("_valid") and v is False
    )
    record = {
        "controller_version": _ctom.CONTROLLER_VERSION,
        "action_protocol": protocol,
        "action_offset": action_offset if categorical else None,
        "action_token_ids": (alphabet if last_shown else alphabet[:2] + [-1]) if categorical else [],
        "sampled_decision": parsed,
        "generated_tokens": generated_tokens,
        "think": think,
        "visible_tokens": len(enc(visible)),
        "decision": decision,
        "decision_raw": decision_raw,
        "coerce_reason": coerce_reason,
        "drafts": drafts_all,
        "draft": drafts_all[-1] if drafts_all else "",
        "predictions": predictions,
        "verdicts": verdicts,
        "verdict_reasons": verdict_reasons,
        "ask": ask_text if decision == _ctom.ASK else "",
        "visible": visible,
        "inner_iters": counters["inner_iters"],
        "invalid_count": validity["invalid_count"],
        "validity": dict(validity),
        "counters": counters,
    }
    return {
        "prompt": prompt,
        "response_tokens_used": response_tokens_used,
        "visible": visible,
        "record": record,
        "terminated": decision == _ctom.COMMIT,
        "last_shown": new_last_shown,
    }


def _make_result(
    sample: dict,
    messages: list[dict],
    predictions: list[str],
    error: Optional[str] = None,
    *,
    phase_validity: Optional[list[dict]] = None,
    drafts: Optional[list[str]] = None,
    bdr_rounds: Optional[list[dict]] = None,
    v7_rounds: Optional[list[dict]] = None,
    sft_v3_mode: bool = False,
    effort_max_turns: int = 8,
    effort_max_user_tokens: int = 256,
    effort_alpha: float = 0.5,
    effort_max_assistant_tokens: int = 0,
    effort_beta: float = 0.0,
    effort_token_units: str = "words",
    effort_mode: str = "bonus",
    invalid_phase_penalty: float = -1.0,
    reward_w_task: float = 0.5,
    reward_w_pred: float = 0.3,
    reward_w_effort: float = 0.2,
    effort_tokenizer=None,
) -> dict:
    """Build the result record with all rewards computed."""
    goal = sample["goal"]
    ground_truth = sample["ground_truth"]

    r_task = compute_task_reward(messages, ground_truth, goal)
    r_pred = compute_f1_prediction_reward(messages, predictions)
    if sft_v3_mode:
        effort_scores = compute_effort_reward_per_turn(
            messages,
            max_turns=effort_max_turns,
            max_user_tokens=effort_max_user_tokens,
            alpha=effort_alpha,
            max_assistant_tokens=effort_max_assistant_tokens,
            beta=effort_beta,
            tokenizer=effort_tokenizer,
            token_units=effort_token_units,
        )
        r_effort = float(np.mean(effort_scores)) if effort_scores else 0.0
    else:
        effort_scores = []
        r_effort = compute_effort_reward(
            messages,
            max_turns=effort_max_turns,
            max_user_tokens=effort_max_user_tokens,
            alpha=effort_alpha,
        )

    # Conventional weighted component total. For SFT-family modes, prediction
    # leaves the scalar reward; r_training_aligned below reconstructs the exact
    # normalized turn-wise objective used by training.
    r_total = (
        reward_w_task * r_task
        + reward_w_pred * r_pred
        + reward_w_effort * r_effort
    )
    phase_validity = phase_validity or []
    training_aligned_return = None
    effort_sum = None
    if sft_v3_mode:
        # sft_v3 prediction quality is trained by auxiliary CE, not scalar
        # R_pred. Reconstruct the exact normalized turn-wise scalar objective.
        weight_norm = reward_w_task + reward_w_effort
        weight_norm = weight_norm or 1.0
        turn_rewards = []
        effort_sum = 0.0
        for i, effort_score in enumerate(effort_scores):
            validity = phase_validity[i] if i < len(phase_validity) else {}
            valid_phase = _phase_is_valid(validity)
            if valid_phase:
                effort_term = (
                    -reward_w_effort * (1.0 - effort_score)
                    if effort_mode == "cost" else reward_w_effort * effort_score
                )
                effort_sum += effort_term / weight_norm
                turn_rewards.append(effort_term / weight_norm)
            else:
                turn_rewards.append(invalid_phase_penalty)
        if turn_rewards:
            final_validity = (
                phase_validity[len(turn_rewards) - 1]
                if len(phase_validity) >= len(turn_rewards)
                else {}
            )
            if _phase_is_valid(final_validity):
                turn_rewards[-1] += reward_w_task * r_task / weight_norm
        training_aligned_return = float(sum(turn_rewards))

    user_turns = sum(1 for m in messages if m["role"] == "user")
    assistant_turns = sum(1 for m in messages if m["role"] == "assistant")

    return {
        "index": sample["index"],
        "problem": sample["single_turn_prompt"],
        "ground_truth": ground_truth,
        "expertise": sample["expertise"],
        "goal": goal,
        "subject": sample.get("subject", ""),
        "difficulty": sample.get("difficulty", ""),
        "messages": messages,
        "predictions": predictions,
        "phase_validity": phase_validity,
        "drafts": drafts or [],
        # sft_v6 only: per-visible-turn BDR round records (draft / belief /
        # decision / ask / visible / internal_turns / invalid_count).
        "bdr_rounds": bdr_rounds or [],
        # sft_v7 only: per-visible-turn records (think / decision / drafts /
        # predictions / verdicts / ask / visible / inner_iters / counters).
        "v7_rounds": v7_rounds or [],
        "effort_scores": effort_scores,
        "r_effort_sum": effort_sum,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "r_task": r_task,
        "r_pred": r_pred,
        "r_effort": r_effort,
        "r_total": r_total,
        "r_training_aligned": training_aligned_return,
        "solved": bool(r_task == 1.0),
        "error": error,
    }


# ============================================================================
# 6. Main Evaluation Loop
# ============================================================================

async def run_evaluation(
    samples: list[dict],
    assistant,
    user_sim: UserSimulator,
    max_turns: int = 5,
    num_workers: int = 1,
    verbose: bool = False,
    q0_cache: Optional[dict] = None,
    max_context_tokens: int = 0,
    use_data_system_prompt: bool = False,
    replay_q0: bool = False,
    effort_max_turns: int = 8,
    effort_max_user_tokens: int = 256,
    effort_alpha: float = 0.5,
    effort_max_assistant_tokens: int = 0,
    effort_beta: float = 0.0,
    effort_token_units: str = "words",
    effort_mode: str = "bonus",
    invalid_phase_penalty: float = -1.0,
    reward_w_task: float = 0.5,
    reward_w_pred: float = 0.3,
    reward_w_effort: float = 0.2,
) -> list[dict]:
    """
    Run evaluation on all samples.

    num_workers=1: sequential with progress bar (sync AssistantModel).
    num_workers>1: N concurrent async workers, each independently running
                   full multi-turn rollouts (matching training-time Ray workers).
                   Requires AsyncAssistantClient.

    If q0_cache is provided, opening user messages are loaded from / saved to
    this dict (keyed by sample index) so that different eval runs use identical
    q0 messages for fair comparison.
    """
    results = []
    semaphore = asyncio.Semaphore(num_workers)
    completed = [0]  # mutable counter for progress tracking

    async def _eval_one(i: int, sample: dict) -> dict:
        async with semaphore:
            if verbose:
                print(f"\n[{i+1}/{len(samples)}] {sample['expertise']}/{sample['goal']}: "
                      f"{sample['single_turn_prompt'][:80]}...")
            result = await run_single_rollout(
                sample, assistant, user_sim,
                max_turns=max_turns, verbose=verbose,
                q0_cache=q0_cache,
                max_context_tokens=max_context_tokens,
                use_data_system_prompt=use_data_system_prompt,
                replay_q0=replay_q0,
                effort_max_turns=effort_max_turns,
                effort_max_user_tokens=effort_max_user_tokens,
                effort_alpha=effort_alpha,
                effort_max_assistant_tokens=effort_max_assistant_tokens,
                effort_beta=effort_beta,
                effort_token_units=effort_token_units,
                effort_mode=effort_mode,
                invalid_phase_penalty=invalid_phase_penalty,
                reward_w_task=reward_w_task,
                reward_w_pred=reward_w_pred,
                reward_w_effort=reward_w_effort,
            )
            if isinstance(result, dict):
                result["rep"] = sample.get("rep", 0)
            completed[0] += 1
            if not verbose and completed[0] % 10 == 0:
                print(f"  Progress: {completed[0]}/{len(samples)} samples completed")
            return result

    if num_workers <= 1:
        # Sequential — sync AssistantModel with progress bar
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            result = await _eval_one(i, sample)
            results.append(result)
    else:
        # Concurrent async workers — like training-time Ray workers
        print(f"[INFO] Launching {num_workers} concurrent workers for {len(samples)} samples...")
        tasks = [_eval_one(i, s) for i, s in enumerate(samples)]
        results = await asyncio.gather(*tasks)
        results = list(results)

    return results


# ============================================================================
# 7. Reporting
# ============================================================================

def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    if not results:
        return {"error": "no results"}

    valid = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]

    # Scoring convention (adopted 2026-09-15): a rollout that errored out did not
    # solve the problem, so it scores 0 rather than leaving the denominator.
    # Dropping errored rows biases every task metric upward in proportion to the
    # error rate, which is NOT equal across arms -- a harness-side failure (e.g.
    # the user sim exceeding its context window) hits the arm with longer
    # conversations hardest and manufactures a difference between checkpoints.
    # Reward/turn metrics stay over `valid` only: r_pred, r_effort and the turn
    # counts are undefined for a rollout that never finished, and zeroing them
    # would be an imputation rather than a scoring rule.
    def _task(r):
        return 0.0 if r["error"] is not None else float(r["r_task"])

    def _solved(r):
        return bool(r["solved"]) and r["error"] is None

    summary = {
        "total_samples": len(results),
        "successful": len(valid),
        "errors": len(errors),
        "error_handling": "errors_as_failures",
    }

    if not valid:
        return summary

    # avg@k / pass@k. With k rollouts per problem, "solve_rate" below is already
    # avg@k (the mean over all rollouts); pass@k needs grouping by problem.
    def _sampling(rows):
        by_problem = {}
        for r in rows:
            by_problem.setdefault(r["index"], []).append(_solved(r))
        k = max((len(v) for v in by_problem.values()), default=1)
        if k <= 1 or not by_problem:
            return None
        n_prob = len(by_problem)
        counts = [sum(v) for v in by_problem.values()]
        return {
            "n_problems": n_prob,
            "n_samples_per_problem": k,
            "avg_at_k": float(np.mean([c / len(v) for c, v in
                                       zip(counts, by_problem.values())])),
            "pass_at_k": sum(1 for c in counts if c > 0) / n_prob,
            "all_k_solved": sum(1 for c, v in zip(counts, by_problem.values())
                                if c == len(v)) / n_prob,
            "never_solved": sum(1 for c in counts if c == 0) / n_prob,
            # per-problem solve count histogram: how many problems were solved
            # 0/k, 1/k, ... k/k times. A spike at the ends means low diversity.
            "solve_count_hist": [sum(1 for c in counts if c == j)
                                 for j in range(k + 1)],
        }

    # Headline: every rollout counts, errored ones as failures. The
    # error-excluded figures are kept alongside so runs scored before
    # 2026-09-15 stay comparable -- they are NOT the reported metric.
    sampling = _sampling(results)
    if sampling is not None:
        summary["sampling"] = sampling
    if errors:
        sampling_excl = _sampling(valid)
        if sampling_excl is not None:
            summary["sampling_excl_errors"] = sampling_excl

    # Overall metrics
    summary["overall"] = {
        "solve_rate": float(np.mean([_task(r) for r in results])),
        "solve_rate_excl_errors": np.mean([r["r_task"] for r in valid]),
        "mean_r_pred": np.mean([r["r_pred"] for r in valid]),
        "mean_r_effort": np.mean([r["r_effort"] for r in valid]),
        "mean_r_total": np.mean([r["r_total"] for r in valid]),
        "mean_user_turns": np.mean([r["user_turns"] for r in valid]),
        "mean_assistant_turns": np.mean([r["assistant_turns"] for r in valid]),
    }
    aligned_returns = [
        r["r_training_aligned"]
        for r in valid
        if r.get("r_training_aligned") is not None
    ]
    phase_rows = [
        phase
        for r in valid
        for phase in r.get("phase_validity", [])
    ]
    if aligned_returns:
        summary["overall"]["mean_r_training_aligned"] = np.mean(aligned_returns)
    effort_sums = [r["r_effort_sum"] for r in valid if r.get("r_effort_sum") is not None]
    if effort_sums:
        summary["overall"]["mean_r_effort_sum"] = np.mean(effort_sums)
    if phase_rows:
        summary["overall"]["sft_v3_valid_phase_rate"] = np.mean([
            _phase_is_valid(p)
            for p in phase_rows
        ])
        summary["overall"]["sft_v3_answer_clip_rate"] = np.mean([
            bool(p.get("answer_clipped")) for p in phase_rows
        ])
        summary["overall"]["sft_v3_prediction_clip_rate"] = np.mean([
            bool(p.get("prediction_clipped")) for p in phase_rows
        ])

    # Breakdown by goal
    for goal in ["solving"]:
        subset = [r for r in valid if r["goal"] == goal]
        scored = [r for r in results if r["goal"] == goal]
        if subset:
            summary[f"goal_{goal}"] = {
                "count": len(scored),
                "count_valid": len(subset),
                "solve_rate": float(np.mean([_task(r) for r in scored])),
                "mean_r_pred": np.mean([r["r_pred"] for r in subset]),
                "mean_r_effort": np.mean([r["r_effort"] for r in subset]),
                "mean_user_turns": np.mean([r["user_turns"] for r in subset]),
            }

    # Breakdown by expertise
    for exp in ["expert", "novice"]:
        subset = [r for r in valid if r["expertise"] == exp]
        scored = [r for r in results if r["expertise"] == exp]
        if subset:
            summary[f"expertise_{exp}"] = {
                "count": len(scored),
                "count_valid": len(subset),
                "solve_rate": float(np.mean([_task(r) for r in scored])),
                "mean_r_pred": np.mean([r["r_pred"] for r in subset]),
                "mean_r_effort": np.mean([r["r_effort"] for r in subset]),
            }

    # Breakdown by persona (2x2)
    for exp in ["expert", "novice"]:
        for goal in ["solving"]:
            subset = [r for r in valid if r["expertise"] == exp and r["goal"] == goal]
            scored = [r for r in results
                      if r["expertise"] == exp and r["goal"] == goal]
            if subset:
                summary[f"persona_{exp}_{goal}"] = {
                    "count": len(scored),
                    "count_valid": len(subset),
                    "solve_rate": float(np.mean([_task(r) for r in scored])),
                    "mean_r_pred": np.mean([r["r_pred"] for r in subset]),
                    "mean_r_effort": np.mean([r["r_effort"] for r in subset]),
                    "mean_user_turns": np.mean([r["user_turns"] for r in subset]),
                }

    # Breakdown by subject (if available)
    subjects = set(r.get("subject", "") for r in results) - {""}
    if subjects:
        summary["by_subject"] = {}
        for subj in sorted(subjects):
            scored = [r for r in results if r.get("subject") == subj]
            if scored:
                summary["by_subject"][subj] = {
                    "count": len(scored),
                    "solve_rate": float(np.mean([_task(r) for r in scored])),
                }

    return summary


def print_summary(summary: dict):
    """Pretty-print evaluation summary."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    n_err = summary.get("errors", 0)
    print(f"Total: {summary['total_samples']}  |  "
          f"Success: {summary['successful']}  |  "
          f"Errors: {n_err}")
    if n_err:
        print(f"Scoring: errors count as failures "
              f"({n_err}/{summary['total_samples']} = "
              f"{100 * n_err / max(summary['total_samples'], 1):.1f}% of rollouts "
              f"score 0). Error-excluded figures shown for reference only.")

    if "overall" in summary:
        o = summary["overall"]
        print(f"\n{'Metric':<25} {'Value':>10}")
        print("-" * 36)
        print(f"{'Solve Rate':<25} {o['solve_rate']:>10.3f}")
        if "solve_rate_excl_errors" in o and summary.get("errors"):
            print(f"{'  (excl. errors, ref)':<25} "
                  f"{o['solve_rate_excl_errors']:>10.3f}")
        print(f"{'Mean R_pred (F1)':<25} {o['mean_r_pred']:>10.3f}")
        print(f"{'Mean R_effort':<25} {o['mean_r_effort']:>10.3f}")
        if "mean_r_training_aligned" in o:
            print(f"{'Training-aligned return':<25} {o['mean_r_training_aligned']:>10.3f}")
            print(f"{'Valid phase rate':<25} {o['sft_v3_valid_phase_rate']:>10.3f}")
            print(f"{'Answer clip rate':<25} {o['sft_v3_answer_clip_rate']:>10.3f}")
            print(f"{'Prediction clip rate':<25} {o['sft_v3_prediction_clip_rate']:>10.3f}")
        print(f"{'Mean R_total (weighted)':<25} {o['mean_r_total']:>10.3f}")
        cfg = summary.get("config", {})
        print(
            "  (weights: "
            f"task={cfg.get('reward_w_task', 0.5)}, "
            f"pred={cfg.get('reward_w_pred', 0.3)}, "
            f"effort={cfg.get('reward_w_effort', 0.2)})"
        )
        print(f"{'Mean User Turns':<25} {o['mean_user_turns']:>10.2f}")
        print(f"{'Mean Assistant Turns':<25} {o['mean_assistant_turns']:>10.2f}")
        samp = summary.get("sampling")
        if samp:
            k = samp["n_samples_per_problem"]
            print(f"\n{f'avg@{k}':<25} {samp['avg_at_k']:>10.3f}"
                  f"   over {samp['n_problems']} problems")
            print(f"{f'pass@{k}':<25} {samp['pass_at_k']:>10.3f}")
            ref = summary.get("sampling_excl_errors")
            if ref:
                print(f"{'  (excl. errors, ref)':<25} "
                      f"avg {ref['avg_at_k']:.3f} / pass {ref['pass_at_k']:.3f}"
                      f"   over {ref['n_problems']} problems")

    # 2x2 persona breakdown
    print(f"\n{'Persona':<25} {'Count':>6} {'Solve%':>8} {'R_pred':>8} {'R_effort':>8} {'Turns':>6}")
    print("-" * 62)
    for exp in ["expert", "novice"]:
        for goal in ["solving"]:
            key = f"persona_{exp}_{goal}"
            if key in summary:
                s = summary[key]
                print(f"{exp}/{goal:<19} {s['count']:>6} {s['solve_rate']:>8.3f} "
                      f"{s['mean_r_pred']:>8.3f} {s['mean_r_effort']:>8.3f} "
                      f"{s['mean_user_turns']:>6.2f}")

    # Subject breakdown
    if "by_subject" in summary:
        print(f"\n{'Subject':<30} {'Count':>6} {'Solve%':>8}")
        print("-" * 46)
        for subj, s in sorted(summary["by_subject"].items()):
            print(f"{subj:<30} {s['count']:>6} {s['solve_rate']:>8.3f}")

    print("=" * 70)


def save_results(results: list[dict], output_dir: str, summary: dict):
    """Save detailed results and summary to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # JSONL with full conversation records
    jsonl_path = os.path.join(output_dir, "eval_results.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            # Convert messages to serializable format
            record = {k: v for k, v in r.items()}
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"[INFO] Detailed results: {jsonl_path}")

    # Summary JSON
    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[INFO] Summary: {summary_path}")

    # CSV for quick analysis (without full conversation text)
    csv_records = []
    for r in results:
        csv_records.append({
            "index": r["index"],
            "rep": r.get("rep", 0),
            "problem": r["problem"][:200],
            "ground_truth": r["ground_truth"],
            "expertise": r["expertise"],
            "goal": r["goal"],
            "subject": r.get("subject", ""),
            "difficulty": r.get("difficulty", ""),
            "user_turns": r["user_turns"],
            "assistant_turns": r["assistant_turns"],
            "r_task": r["r_task"],
            "r_pred": r["r_pred"],
            "r_effort": r["r_effort"],
            "r_total": r["r_total"],
            "r_training_aligned": r.get("r_training_aligned"),
            "sft_v3_valid_phases": sum(
                _phase_is_valid(p)
                for p in r.get("phase_validity", [])
            ),
            "sft_v3_total_phases": len(r.get("phase_validity", [])),
            "solved": r["solved"],
            "error": r["error"],
        })
    csv_path = os.path.join(output_dir, "eval_results.csv")
    pd.DataFrame(csv_records).to_csv(csv_path, index=False)
    print(f"[INFO] CSV summary: {csv_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CToM Model Evaluation")

    # Model / checkpoint
    p.add_argument("--base_model_path", default="Qwen/Qwen2.5-3B-Instruct",
                    help="HuggingFace base model path")
    p.add_argument("--checkpoint_path", default=None,
                    help="Path to veRL LoRA checkpoint (e.g. checkpoints/global_step_200/actor). "
                         "If None, evaluates the base model.")
    p.add_argument("--skip_merge", action="store_true",
                    help="Skip LoRA merge (use --merged_model_dir instead)")
    p.add_argument("--merged_model_dir", default=None,
                    help="Path to already-merged model (use with --skip_merge)")

    # Data
    p.add_argument("--test_data", required=True,
                    help="Path to test parquet (e.g. data/math-l5-expert/rl_validation.parquet)")
    p.add_argument("--n_samples", type=int, default=1,
                    help="Rollouts per problem (k). k>1 enables avg@k and pass@k; "
                         "each problem is evaluated k times independently. Generation is "
                         "unseeded, so the k rollouts differ naturally at temperature>0. "
                         "At temperature 0 they will be near-identical and k>1 is pointless.")
    p.add_argument("--max_samples", type=int, default=0,
                    help="Max samples to evaluate (0 = all)")
    p.add_argument("--seed", type=int, default=42)

    # Rollout parameters
    p.add_argument("--max_turns", type=int, default=8,
                    help="Max assistant turns per conversation (matches training max_assistant_turns)")
    p.add_argument("--num_workers", type=int, default=16,
                    help="Number of concurrent rollout workers (matches training num_workers). "
                         "Requires --assistant_api_base. Set to 1 for sequential mode.")

    # Assistant — API mode (preferred, enables concurrent workers)
    p.add_argument("--assistant_api_base", default=None,
                    help="Assistant vLLM server URL(s). Comma-separated for multiple backends "
                         "(e.g. http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1). "
                         "Requests are distributed via round-robin. "
                         "When set, enables fully async concurrent rollouts.")
    p.add_argument("--assistant_model_name", default=None,
                    help="Model name for assistant API calls (must match --served-model-name "
                         "on the vLLM server). Defaults to --base_model_path.")

    # Assistant — local vLLM mode (fallback, sequential only)
    p.add_argument("--temperature", type=float, default=0.001)  # near-greedy; overridden by scripts/eval_ctom.sh
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_response_length", type=int, default=2048)
    p.add_argument(
        "--assistant_output_mode",
        choices=("standard", "sft_v3", "sft_v4", "sft_v5", "sft_v6", "sft_v7"),
        default="standard",
        help="Controller-owned generation protocol: sft_v3 = answer+prediction, "
             "sft_v4/sft_v5 = draft+prediction+revision (only the revision is "
             "shown; v4 and v5 decode identically and differ only in training), "
             "sft_v6 = BDR draft+belief+decision with internal REVISE turns "
             "(the retired BDR predecessor; --pred_answer_max_tokens is the draft cap), "
             "sft_v7 = plan+decision, draft with an anticipated user turn and "
             "internal revisions until the parsed verdict is SATISFIED "
             "(the paper's method section; --pred_answer_max_tokens is the draft "
             "cap, --prediction_max_tokens the anticipated turn).",
    )
    p.add_argument("--v7_action_protocol", choices=("legacy", "categorical_v1"), default="legacy")
    p.add_argument("--v7_think_max_tokens", type=int, default=256,
                   help="sft_v7: plan ([PLAN_START]) phase cap.")
    p.add_argument("--v7_ask_max_tokens", type=int, default=512,
                   help="sft_v7: ask phase cap.")
    p.add_argument("--v7_max_inner_revisions", type=int, default=3,
                   help="sft_v7: internal revisions allowed per user round. Must "
                        "match training's --v7_max_inner_revisions.")
    p.add_argument("--bdr_belief_max_tokens", type=int, default=256,
                   help="sft_v6: belief phase cap.")
    p.add_argument("--bdr_ask_max_tokens", type=int, default=512,
                   help="sft_v6: ask phase cap.")
    p.add_argument("--max_internal_revisions", type=int, default=1,
                   help="sft_v6: internal REVISE turns allowed per user round.")
    p.add_argument(
        "--pred_answer_max_tokens", type=int, default=2048,
        help="Answer phase cap (sft_v3) / DRAFT phase cap (sft_v4).",
    )
    p.add_argument("--prediction_max_tokens", type=int, default=384)
    p.add_argument(
        "--pred_revision_max_tokens", type=int, default=2048,
        help="sft_v4 only: cap on the revision, i.e. the visible answer.",
    )
    p.add_argument(
        "--trajectory_response_length",
        type=int,
        default=8192,
        help="Global response-space cap across user/controller/assistant tokens.",
    )
    p.add_argument(
        "--answer_gated_termination",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match the training run's answer_gated_termination setting. ON: the "
             "strict checker ends a solving episode on a correct boxed answer and "
             "the simulator is told it has no stop token. Getting this wrong does "
             "NOT crash — it silently evaluates under the other termination rule.",
    )
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=16384)

    # User simulator
    p.add_argument("--prompt_dir", required=True,
                    help="Directory containing the user-simulator prompt templates")
    p.add_argument("--user_sim_api_base", default=None,
                    help="Local vLLM server URL(s). Comma-separated for multiple backends "
                         "(e.g. http://host1:8100/v1,http://host2:8100/v1). "
                         "Requests are distributed via round-robin. "
                         "If not set, uses Together API via TOGETHER_API_KEY env var.")
    p.add_argument("--user_sim_api_key", default=None,
                    help="API key for user simulator (defaults to TOGETHER_API_KEY)")
    p.add_argument("--user_sim_novice_model", default="Qwen/Qwen3-32B-FP8")
    p.add_argument("--user_sim_expert_model", default="Qwen/Qwen3-32B-FP8")
    p.add_argument("--user_sim_temperature", type=float, default=0.7)
    p.add_argument("--user_sim_max_tokens", type=int, default=1024)
    p.add_argument(
        "--user_sim_max_model_len",
        type=int,
        default=16384,
        help="Informational record of the user-simulator server context limit.",
    )
    p.add_argument("--effort_max_turns", type=int, default=8)
    p.add_argument("--effort_max_user_tokens", type=int, default=256)
    p.add_argument("--effort_alpha", type=float, default=0.5)
    p.add_argument("--effort_max_assistant_tokens", type=int, default=0)
    p.add_argument("--effort_beta", type=float, default=0.0)
    p.add_argument("--effort_token_units", choices=("words", "tokens"), default="words")
    p.add_argument("--effort_mode", choices=("bonus", "cost"), default="bonus")
    p.add_argument("--invalid_phase_penalty", type=float, default=-1.0)
    p.add_argument("--reward_w_task", type=float, default=0.5)
    p.add_argument("--reward_w_pred", type=float, default=0.3)
    p.add_argument("--reward_w_effort", type=float, default=0.2)

    # Context truncation
    p.add_argument("--max_context_tokens", type=int, default=12000,
                    help="Max estimated tokens for conversation context. "
                         "Oldest non-system messages are dropped when exceeded. 0 = disabled.")

    # Opening message (q0) reproducibility
    p.add_argument("--q0_cache_path", default=None,
                    help="Path to a JSON file for caching q0 (opening user messages). "
                         "If the file exists, q0 messages are loaded from it to ensure "
                         "identical openings across eval runs. If it doesn't exist, "
                         "q0 messages are generated and saved to it for future runs.")
    p.add_argument("--replay_q0", action=argparse.BooleanOptionalAction, default=True,
                    help="Use single_turn_prompt verbatim as the opening user message "
                         "instead of a sim-generated q0 (default: on, matching the "
                         "training-side replay_q0=true default since 2026-07-09). Pass "
                         "--no-replay_q0 only for checkpoints trained before the fix, "
                         "or the eval q0 distribution won't match training.")

    # Output
    p.add_argument("--output_dir", default=None,
                    help="Output directory (auto-generated if not set)")
    p.add_argument("--assistant_system_prompt_from_data", action="store_true",
                   help="Use the system prompt baked into the eval parquet instead of the "
                        "default prediction-format SYSTEM_PROMPT. Use for the outcome-only "
                        "baseline (no-prediction, raw-\\boxed{} prompt).")
    p.add_argument("--verbose", action="store_true",
                    help="Print each turn during rollout")

    return p.parse_args()


async def main():
    args = parse_args()
    start_time = time.time()

    # ── Determine model path ──
    if args.assistant_api_base:
        # API mode — model is already served externally, no local loading needed
        model_path = args.assistant_model_name or args.base_model_path
        print(f"[INFO] Assistant served externally via API, model name: {model_path}")
    elif args.skip_merge:
        model_path = args.merged_model_dir or args.base_model_path
        print(f"[INFO] Using model directly: {model_path}")
    elif args.checkpoint_path:
        # Merge LoRA into base model
        merged_dir = args.merged_model_dir or os.path.join(
            os.path.dirname(args.checkpoint_path),
            "merged_for_eval",
        )
        model_path = merge_checkpoint(
            args.base_model_path, args.checkpoint_path, merged_dir,
        )
    else:
        # No checkpoint → evaluate base model
        model_path = args.base_model_path
        print(f"[INFO] No checkpoint specified — evaluating base model: {model_path}")

    # ── Load data ──
    samples = load_test_data(args.test_data, max_samples=args.max_samples, seed=args.seed)
    n_problems = len(samples)
    if args.n_samples > 1:
        # k independent rollouts per problem. "index" stays the problem identity so
        # avg@k / pass@k group on it; "rep" distinguishes the rollouts.
        samples = [dict(s, rep=r) for r in range(args.n_samples) for s in samples]
        print(f"[INFO] n_samples={args.n_samples}: {n_problems} problems -> "
              f"{len(samples)} rollouts")
    if not samples:
        print("[ERROR] No samples loaded. Check --test_data path.")
        sys.exit(1)

    # Print data distribution
    goal_counts = defaultdict(int)
    exp_counts = defaultdict(int)
    for s in samples:
        goal_counts[s["goal"]] += 1
        exp_counts[s["expertise"]] += 1
    print("[INFO] Sample distribution:")
    print(f"  Goals:     {dict(goal_counts)}")
    print(f"  Expertise: {dict(exp_counts)}")

    # ── Initialize models ──
    if args.assistant_api_base:
        # Async API mode — enables concurrent workers (like training Ray workers)
        assistant_model_name = args.assistant_model_name or args.base_model_path
        assistant = AsyncAssistantClient(
            api_base=args.assistant_api_base,
            model_name=assistant_model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_response_length,
            output_mode=args.assistant_output_mode,
            tokenizer_path=args.base_model_path,
            pred_answer_max_tokens=args.pred_answer_max_tokens,
            prediction_max_tokens=args.prediction_max_tokens,
            pred_revision_max_tokens=args.pred_revision_max_tokens,
            trajectory_response_length=args.trajectory_response_length,
            bdr_belief_max_tokens=args.bdr_belief_max_tokens,
            bdr_ask_max_tokens=args.bdr_ask_max_tokens,
            bdr_max_internal_revisions=args.max_internal_revisions,
            v7_action_protocol=args.v7_action_protocol,
            v7_think_max_tokens=args.v7_think_max_tokens,
            v7_ask_max_tokens=args.v7_ask_max_tokens,
            v7_max_inner_revisions=args.v7_max_inner_revisions,
        )
        num_workers = args.num_workers
        print(f"[INFO] Assistant API mode: {args.assistant_api_base}")
        print(f"[INFO] Model name: {assistant_model_name}, Workers: {num_workers}")
    else:
        # Local vLLM mode — synchronous, sequential only
        assistant = AssistantModel(
            model_path=model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_response_length,
        )
        num_workers = 1
        if args.num_workers > 1:
            print("[WARN] --num_workers > 1 requires --assistant_api_base. Falling back to sequential.")

    user_sim = UserSimulator(
        prompt_dir=args.prompt_dir,
        api_base=args.user_sim_api_base,
        api_key=args.user_sim_api_key,
        novice_model=args.user_sim_novice_model,
        expert_model=args.user_sim_expert_model,
        max_tokens=args.user_sim_max_tokens,
        temperature=args.user_sim_temperature,
        answer_gated_termination=args.answer_gated_termination,
    )
    print(
        f"[INFO] Answer-gated termination: "
        f"{'ON (checker ends solving episodes)' if args.answer_gated_termination else 'OFF (simulator emits the stop token)'}"
    )

    # ── Load or initialize q0 cache for reproducible opening messages ──
    q0_cache = None
    if args.q0_cache_path:
        if os.path.isfile(args.q0_cache_path):
            with open(args.q0_cache_path, "r") as f:
                q0_cache = json.load(f)
            print(f"[INFO] Loaded {len(q0_cache)} cached q0 messages from {args.q0_cache_path}")
        else:
            q0_cache = {}
            print(f"[INFO] q0 cache file not found — will generate and save to {args.q0_cache_path}")

    # ── Run evaluation ──
    results = await run_evaluation(
        samples, assistant, user_sim,
        max_turns=args.max_turns,
        num_workers=num_workers,
        verbose=args.verbose,
        q0_cache=q0_cache,
        max_context_tokens=args.max_context_tokens,
        use_data_system_prompt=args.assistant_system_prompt_from_data,
        replay_q0=args.replay_q0,
        effort_max_turns=args.effort_max_turns,
        effort_max_user_tokens=args.effort_max_user_tokens,
        effort_alpha=args.effort_alpha,
        effort_max_assistant_tokens=args.effort_max_assistant_tokens,
        effort_beta=args.effort_beta,
        effort_token_units=args.effort_token_units,
        effort_mode=args.effort_mode,
        invalid_phase_penalty=args.invalid_phase_penalty,
        reward_w_task=args.reward_w_task,
        reward_w_pred=args.reward_w_pred,
        reward_w_effort=args.reward_w_effort,
    )

    # ── Save q0 cache if we generated new entries ──
    if args.q0_cache_path and q0_cache:
        os.makedirs(os.path.dirname(args.q0_cache_path) or ".", exist_ok=True)
        with open(args.q0_cache_path, "w") as f:
            json.dump(q0_cache, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(q0_cache)} q0 messages to {args.q0_cache_path}")

    # ── Compute and display summary ──
    summary = compute_summary(results)
    summary["config"] = {
        "model_path": model_path,
        "checkpoint_path": args.checkpoint_path,
        "assistant_api_base": args.assistant_api_base,
        "num_workers": num_workers,
        "test_data": args.test_data,
        "v7_action_protocol": args.v7_action_protocol,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "max_turns": args.max_turns,
        "max_context_tokens": args.max_context_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_response_length": args.max_response_length,
        "max_model_len": args.max_model_len,
        "assistant_output_mode": args.assistant_output_mode,
        "controller_version": _ctom.CONTROLLER_VERSION if args.assistant_output_mode == "sft_v7" else None,
        "v7_think_max_tokens": args.v7_think_max_tokens,
        "v7_ask_max_tokens": args.v7_ask_max_tokens,
        "v7_max_inner_revisions": args.v7_max_inner_revisions,
        "n_samples": args.n_samples,
        "pred_answer_max_tokens": args.pred_answer_max_tokens,
        "pred_revision_max_tokens": args.pred_revision_max_tokens,
        "prediction_max_tokens": args.prediction_max_tokens,
        "trajectory_response_length": args.trajectory_response_length,
        "assistant_system_prompt_from_data": args.assistant_system_prompt_from_data,
        "replay_q0": args.replay_q0,
        "answer_gated_termination": args.answer_gated_termination,
        "user_sim_novice_model": args.user_sim_novice_model,
        "user_sim_expert_model": args.user_sim_expert_model,
        "user_sim_temperature": args.user_sim_temperature,
        "user_sim_max_tokens": args.user_sim_max_tokens,
        "user_sim_max_model_len": args.user_sim_max_model_len,
        "effort_max_turns": args.effort_max_turns,
        "effort_max_user_tokens": args.effort_max_user_tokens,
        "effort_alpha": args.effort_alpha,
        "effort_max_assistant_tokens": args.effort_max_assistant_tokens,
        "effort_beta": args.effort_beta,
        "effort_token_units": args.effort_token_units,
        "effort_mode": args.effort_mode,
        "invalid_phase_penalty": args.invalid_phase_penalty,
        "reward_w_task": args.reward_w_task,
        "reward_w_pred": args.reward_w_pred,
        "reward_w_effort": args.reward_w_effort,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    print_summary(summary)

    # ── Save results ──
    output_dir = args.output_dir or os.path.join(
        "eval_outputs",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    save_results(results, output_dir, summary)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Evaluation completed in {elapsed:.1f}s. Results in: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
