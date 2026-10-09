# Copyright 2025 The CToM Authors
# Reward function for Multi-Turn Personalized Dialogue Training.
#
# Implements three reward components from the paper:
#   R = R_task + R_pred + R_effort
#
# R_task:   per-domain task success (see the paper, Appendix A.2)
# R_pred:   Prediction accuracy (negated average prediction loss proxy)
# R_effort: User effort penalty (fewer turns + fewer user tokens = better)
import asyncio
import json
import logging
import os
import socket
import re
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
import torch
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.experimental.reward_loop.reward_manager import register
from ctom.task_coding import pass_rate as _bcb_pass_rate
from ctom.task_coding import run_tests_async as _bcb_run_tests_async
from ctom.submission import extract_final_code
from ctom import markers as _markers
from ctom.utils import SPEC_TASK_TYPES
from ctom.task_writing import document_similarity, extract_final_document
from ctom.task_math import check_math_answer
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"
UNDERSTOOD_TOKEN = "<understood>"
# Shell launchers cannot pass a real None through hydra, so they spell an unset
# optional knob "none". Numeric 0 is deliberately NOT a sentinel: voi tau=0 is
# math's real default, and silently dropping it would fall back to default_tau().
_NONE_SENTINELS = frozenset({"", "none", "null"})


def _is_none_sentinel(value) -> bool:
    """True when a launcher-supplied value means "leave this unset"."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in _NONE_SENTINELS


def _parse_pair(value, *, name: str) -> tuple[float, float]:
    """Parse a "a:b" string or a 2-sequence into a (float, float) pair."""
    parts = value.split(":") if isinstance(value, str) else list(value)
    if len(parts) != 2:
        raise ValueError(
            f"{name} must be 'base:scale', a 2-sequence, or 'none' to disable; "
            f"got {value!r}"
        )
    return float(parts[0]), float(parts[1])
# ──────────────────────────────────────────────────────────────────────
# Evaluation Logger — thread-safe JSONL writer for reward model training
# ──────────────────────────────────────────────────────────────────────
class EvalLogger:
    """
    Thread-safe JSONL logger for LLM prediction evaluations.
    Each line is a self-contained JSON record suitable for training a
    reward model:
        {
            "timestamp": "...",
            "turn_index": 2,
            "conversation_context": "[USER]: ... \\n[ASSISTANT]: ...",
            "predicted_response": "...",
            "actual_response": "...",
            "evaluator_model": "meta-llama/...",
            "evaluator_rationale": "The prediction captures ...",
            "score": 0.8,
            "f1_score": 0.65,          # included when available
            "goal": "solving",
            "data_source": "math-hard"
        }
    """
    def __init__(self, log_path: str | Path):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.info(f"EvalLogger writing to {self._path}")
    def log(self, record: dict) -> None:
        record.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(),
        )
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
# ──────────────────────────────────────────────────────────────────────
# Prediction Reward Mode
# ──────────────────────────────────────────────────────────────────────
class PredictionMode(str, Enum):
    """Controls which prediction reward variant to use."""
    F1 = "f1"              # Word-overlap F1 (original, no API call)
    LLM = "llm"            # LLM-as-judge via Together API
    COMBINED = "combined"  # Average of F1 and LLM scores
    SFT = "sft"            # Prediction moved OUT of reward into an auxiliary SFT
                           # loss on the actor (teacher-forced toward the real
                           # next user turn). Contributes 0 to the scalar reward.
    SFT_V2 = "sft_v2"      # Like SFT, but the auxiliary loss uses per-turn
                           # ON-POLICY context: each turn's target is spliced
                           # into the actual rollout tokens (sampled earlier
                           # predictions kept), instead of one fully teacher-
                           # forced sequence. Also contributes 0 to the reward.
    SFT_V3 = "sft_v3"      # Controller-owned answer/prediction phases, exact
                            # masks, explicit stop supervision, and detached
                            # prediction-NLL credit on answer tokens.
    SFT_V5 = "sft_v5"      # sft_v4's generation exactly, but the prediction is
                            # supervised against the simulator's COUNTERFACTUAL
                            # reaction to the draft instead of the real next user
                            # turn. In v4 the real turn was a reaction to the
                            # revision while the supervised context ends at the
                            # draft, so conditioning and target disagreed; here
                            # they agree, and the target is diagnostic (what the
                            # user would object to) rather than optimistic.
    SFT_V6 = "sft_v6"      # BDR (belief / decide / revise): draft -> belief ->
                            # decision (COMMIT|SHOW|ASK|REVISE); REVISE opens an
                            # internal turn the simulator never sees. The belief
                            # is unscored; asking is paid by value-of-information
                            # (reward_composition=voi). the retired BDR predecessor.
    SFT_V7 = "sft_v7"      # decide {DRAFT|ASK|COMMIT}; DRAFT = draft -> policy-
                            # written anticipated user turn -> parsed verdict ->
                            # revise as a NEW assistant turn (cap); ASK shows the
                            # prediction as a confirm-or-correct assumption.
                            # Aux NLL on every prediction slot, counterfactual
                            # targets. Reward = sft_v5's turn-wise task + effort
                            # cost. the paper's method section.
    SFT_V4 = "sft_v4"      # Predict-then-revise: draft -> prediction -> revision,
                            # all three phases in the PPO loss. Only the revision
                            # is shown to the user, so the prediction finally sits
                            # UPSTREAM of the text that gets scored (in sft_v3 it
                            # was generated after the answer and could not affect
                            # it). Same reward shape as SFT_V3 otherwise.


# Modes where prediction leaves the scalar reward for an auxiliary actor loss.
SFT_MODES = (
    PredictionMode.SFT,
    PredictionMode.SFT_V2,
    PredictionMode.SFT_V3,
    PredictionMode.SFT_V4,
    PredictionMode.SFT_V5,
    PredictionMode.SFT_V6,
    PredictionMode.SFT_V7,
)
# Modes where the controller owns the phase delimiters, so a turn that fails to
# produce every phase is penalized with invalid_phase_penalty instead of scored.
PHASE_GATED_MODES = (
    PredictionMode.SFT_V3,
    PredictionMode.SFT_V4,
    PredictionMode.SFT_V5,
    PredictionMode.SFT_V6,
    PredictionMode.SFT_V7,
)
# Every site that BUILDS a result dict must cover these, or the composition
# raises KeyError mid-run (see _empty_result_dict).


def _empty_result_dict() -> dict:
    """Gradient-neutral result for a rollout that produced no messages.

    Must carry EVERY key any composition branch indexes — not just the CToM
    pair. A missing key
    directly, so omitting them here killed a run at the first empty rollout
    surfaces as a KeyError mid-step. The values are irrelevant to the gradient
    (`skipped=True` neutralizes the sample); their PRESENCE is what matters,
    and it also keeps reward_extra_info's schema constant across the batch
    concat.
    """
    return {
        "task": torch.tensor(0.0, dtype=torch.float32),
        "prediction": torch.tensor(0.0, dtype=torch.float32),
        "effort": torch.tensor(0.0, dtype=torch.float32),
        "skipped": torch.tensor(True, dtype=torch.bool),
    }


# The SAME failure one level up. verl's agent_loop._postprocess builds the batch
# with
#     reward_extra_keys = list(reward_extra_infos[0].keys())
#     non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])
# i.e. it takes the key list from ROLLOUT 0 and then indexes every other rollout
# with it. A key that one rollout has and another lacks is a hard KeyError in
# the middle of generate_sequences: the conversation
# path emitted it, the empty-generation path did not, and whichever rollout
# landed at index 0 decided whether the step survived.
#
# So every return path fills THIS dict. Defaults live here; call sites override.
REWARD_EXTRA_INFO_SCHEMA = {
    "reward/task": 0.0,
    "reward/prediction": 0.0,
    "reward/effort": 0.0,
    "reward/effort_sum": 0.0,
    # CollabLLM-composition components: 0.0 under the CToM composition, which
    # never computes them. Present regardless — see above.
    "reward/total": 0.0,
    "reward/num_scored_turns": 0.0,
    "reward/turn_reward_mean": 0.0,
    "reward/skipped": False,
    "reward/empty_generation": False,
    "acc": 0.0,
    # BDR / VOI composition telemetry (0.0 on every other path).
    **{key: 0.0 for key in _markers.VOI_STAT_KEYS},
}


def _reward_extra_info(**overrides) -> dict:
    """Build reward_extra_info with the full canonical key set.

    Unknown keys are rejected rather than merged: a key added at one call site
    only is precisely the bug this exists to prevent, and it would not surface
    until a batch happened to mix the two paths.
    """
    unknown = sorted(set(overrides) - set(REWARD_EXTRA_INFO_SCHEMA))
    if unknown:
        raise KeyError(
            f"reward_extra_info keys not in REWARD_EXTRA_INFO_SCHEMA: {unknown}. "
            "Add them to the schema so every return path emits them."
        )
    info = dict(REWARD_EXTRA_INFO_SCHEMA)
    info.update(overrides)
    return info
# ──────────────────────────────────────────────────────────────────────
# API Key Rotator — thread-safe round-robin across multiple keys
# ──────────────────────────────────────────────────────────────────────
class KeyRotator:
    """Thread-safe round-robin rotation across multiple API keys.
    Automatically discovers keys from environment variables named
    TOGETHER_API_KEY, TOGETHER_API_KEY_1, TOGETHER_API_KEY_2, ...
    Duplicate values are deduplicated while preserving insertion order.
    """
    def __init__(self, keys: list[str]):
        assert keys, "At least one API key is required"
        self._keys = keys
        self._idx = 0
        self._lock = threading.Lock()
    def next(self) -> str:
        with self._lock:
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
            return key
    def __len__(self) -> int:
        return len(self._keys)
    @classmethod
    def from_env(cls, explicit_key: Optional[str] = None) -> Optional["KeyRotator"]:
        """Build a rotator from an explicit key and/or env vars.
        Scans TOGETHER_API_KEY, TOGETHER_API_KEY_1 .. TOGETHER_API_KEY_10.
        Returns None if no keys are found.
        """
        keys: list[str] = []
        seen: set[str] = set()
        def _add(k: str):
            k = k.strip()
            if k and k not in seen:
                keys.append(k)
                seen.add(k)
        if explicit_key:
            _add(explicit_key)
        # Check numbered env vars first (preferred ordering)
        for i in range(1, 11):
            val = os.getenv(f"TOGETHER_API_KEY_{i}", "")
            _add(val)
        # Fallback: the unnumbered var
        _add(os.getenv("TOGETHER_API_KEY", ""))
        return cls(keys) if keys else None
# ──────────────────────────────────────────────────────────────────────
# LLM Evaluator Prompt
# ──────────────────────────────────────────────────────────────────────
DEFAULT_LLM_EVALUATOR_PROMPT = """\
You are an expert evaluator assessing how well a **predicted user response** matches the **actual user response** in a multi-turn dialogue.
## Context
An AI assistant is conversing with a user. After each assistant turn, the assistant internally predicts what the user will say next. Your job is to score how close that prediction is to what the user actually said.
## Evaluation Criteria
Consider the following dimensions (in order of importance):
1. **Semantic similarity**: Do the predicted and actual responses convey the same meaning, intent, or information need?
2. **Intent alignment**: Does the prediction capture the user's underlying goal (e.g., asking for clarification, providing an answer, expressing confusion)?
3. **Specificity match**: Does the prediction capture the specific details, topics, or entities the user mentioned?
4. **Tone/style consistency**: Does the prediction reflect a similar tone (e.g., confused, confident, frustrated)?
## Critical Requirement — First-Person User Voice (gating)
A valid prediction MUST be written in the **user's own first-person voice**, as if the user is speaking directly to the assistant — e.g. "I think this step is wrong because…", "Why do we divide here?", "Can you show me another example?".
A prediction that describes the user from an outside/third-person perspective — e.g. "The user will ask for clarification", "They would want a simpler explanation", "The user is likely to say…" — is INVALID as a next-user-turn prediction, no matter how semantically accurate its content is.
Any such third-person/observer prediction MUST receive **SCORE: 0.0**. Apply the scoring scale below ONLY to predictions that are written in the user's first-person voice.
## Scoring Scale
- **1.0**: Near-perfect match — same meaning, intent, and key details
- **0.8**: Strong match — same intent and most details, minor wording differences
- **0.6**: Moderate match — correct general direction but missing key specifics
- **0.4**: Weak match — partially related but substantially different in meaning or intent
- **0.2**: Poor match — only superficially related (e.g., same topic but different intent)
- **0.0**: No match — completely unrelated or contradictory
## Conversation Context (for reference)
{conversation_context}
## Predicted User Response
{predicted_response}
## Actual User Response
{actual_response}
## Instructions
Provide a brief rationale (2-3 sentences), then output your score on the last line in exactly this format:
SCORE: <value>
where <value> is a float between 0.0 and 1.0."""
# ──────────────────────────────────────────────────────────────────────
# Helper: extract role/content from message (dict or object)
# ──────────────────────────────────────────────────────────────────────
def _get_role(msg) -> str:
    return msg.role if hasattr(msg, "role") else msg.get("role", "")
def _get_content(msg) -> str:
    return msg.content if hasattr(msg, "content") else msg.get("content", "")
def _extract_user_response(text: str) -> str:
    """Extract the 'response' field from simulator JSON, or return text as-is.
    
    The user simulator returns structured JSON like:
        ```json
        {"current_status": "...", "thought": "...", "response": "..."}
        ```
    This extracts just the 'response' field for clean comparison.
    """
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed["response"]
    except (json.JSONDecodeError, TypeError):
        pass
    return text
def _extract_prediction_text(pred) -> str:
    """Extract just the prediction string from possibly-structured assistant output.
    
    The assistant outputs JSON like:
        {"answer": "...", "prediction": "..."}
    This extracts just the 'prediction' field.
    """
    if isinstance(pred, dict):
        return pred.get("prediction", "")
    if isinstance(pred, str):
        try:
            parsed = json.loads(pred)
            if isinstance(parsed, dict):
                return parsed.get("prediction", "")
        except (json.JSONDecodeError, TypeError):
            pass
    return pred if isinstance(pred, str) else ""
def _format_conversation_context(messages: list, up_to_turn: int) -> str:
    """Format conversation history up to a given turn for the evaluator prompt."""
    lines = []
    turn_count = 0
    for msg in messages:
        role = _get_role(msg)
        content = _get_content(msg)
        if role == "user":
            turn_count += 1
        if turn_count > up_to_turn:
            break
        if role != "system":
            lines.append(f"[{role.upper()}]: {content}")
    return "\n".join(lines) if lines else "(no prior context)"
# ──────────────────────────────────────────────────────────────────────
# Individual Reward Components
# ──────────────────────────────────────────────────────────────────────
def compute_task_reward(
    messages: list,
    ground_truth: str,
    goal: str,
) -> float:
    """
    R_task (Eq. 2):
    - Solving: 1 if the final boxed answer in ANY assistant turn is
      mathematically equivalent to the ground truth, else 0
    - Learning: 1 if the user signaled <understood> or [[TERMINATE CHAT]], else 0
    """
    user_messages = [m for m in messages if _get_role(m) == "user"]
    assistant_messages = [m for m in messages if _get_role(m) == "assistant"]
    has_terminate = any(
        TERMINATION_SIGNAL in _get_content(m) for m in user_messages
    )

    if goal == "solving":
        if not assistant_messages:
            return 0.0
        # Check every assistant turn for the correct answer
        if any(check_math_answer(_get_content(m), ground_truth) for m in assistant_messages):
            return 1.0
        return 0.0
    return 0.0


def compute_length_reward(messages: list, target_words: int = 3000) -> float:
    """Plumbing sanity-check reward: normalized total assistant response length.

    r = min(1, total_assistant_words / target_words). Trivially learnable — if
    the RL loop is healthy, mean response length (and reward/task) must climb
    within a few steps. Pick target_words comfortably above the policy's
    initial mean length: once every rollout in a GRPO group hits the cap the
    group advantage is 0 and the signal disappears.
    """
    total_words = sum(
        len(_get_content(m).split())
        for m in messages
        if _get_role(m) == "assistant"
    )
    return min(1.0, total_words / max(1, target_words))



def compute_writing_task_reward(
    messages: list,
    ground_truth: str,
    interaction_kwargs: dict,
) -> float:
    """CollabLLM MediumDocEdit R_task: similarity of the submitted document to
    the target article the human simulator was holding.

    ROUGE-L F1 by default. The floor is ~0.11 rather than 0 (two unrelated
    articles still share stop words), so read a score as "how much of the target
    made it in", not as a hit rate -- see ctom/task_writing.py for the
    calibration this default came from.
    """
    document = extract_final_document(messages)
    return document_similarity(
        document,
        ground_truth,
        metric=str(interaction_kwargs.get("writing_metric", "")),
    )


def compute_bigcodebench_task_reward(
    messages: list,
    ground_truth: str,
    interaction_kwargs: dict,
) -> float:
    """CollabLLM BigCodeBench R_task: fraction of the task's unittest methods the
    submitted code passes -- their "runnable code->pass_rate" metric.

    The suite travels in interaction_kwargs["test_source"] (put there by
    data_prep/build_coding.py); `ground_truth` is the reference solution and
    is NOT used for scoring, only for diagnostics -- BigCodeBench has no
    reference OUTPUT to diff against.

    Execution happens in a separate interpreter; see ctom/task_coding.py
    for why, and for the entry-point aliasing that stands in for CollabLLM's LLM
    extraction pass.
    """
    test_source = str(interaction_kwargs.get("test_source", "") or "")
    if not test_source:
        logger.error("BigCodeBench row carries no test_source; assigning task reward 0")
        return 0.0
    candidate = extract_final_code(messages)
    return _bcb_pass_rate(candidate, test_source)


async def compute_task_reward_for_example_async(
    messages: list,
    ground_truth: str,
    goal: str,
    interaction_kwargs: Optional[dict] = None,
    task_reward_mode: str = "math",
    length_target_words: int = 3000,
) -> float:
    """Async wrapper: identical to the sync version for every domain but one.

    BigCodeBench's R_task is a BLOCKING subprocess (a whole unittest suite, ~0.7s
    median and up to CTOM_BCB_TIMEOUT). Both reward paths that call this are
    `async def`, so running it inline would hold the event loop for the duration
    and serialize the scoring of every rollout in the step — the reward manager's
    concurrency would silently become 1. Here the grading wait happens on
    task_coding's dedicated worker pool, which both frees the coordinator
    and caps how many suites run at once (CTOM_BCB_CONCURRENCY).

    Every other domain scores in-process in microseconds and stays on the direct
    call, so no existing path changes behaviour or ordering.
    """
    interaction_kwargs = interaction_kwargs or {}
    if interaction_kwargs.get("task_type") == "bigcodebench" and task_reward_mode != "length":
        test_source = str(interaction_kwargs.get("test_source", "") or "")
        if not test_source:
            logger.error("BigCodeBench row carries no test_source; assigning task reward 0")
            return 0.0
        result = await _bcb_run_tests_async(extract_final_code(messages), test_source)
        return float(result["rate"])
    return compute_task_reward_for_example(
        messages, ground_truth, goal, interaction_kwargs,
        task_reward_mode, length_target_words,
    )


def compute_task_reward_for_example(
    messages: list,
    ground_truth: str,
    goal: str,
    interaction_kwargs: Optional[dict] = None,
    task_reward_mode: str = "math",
    length_target_words: int = 3000,
) -> float:
    """Route task scoring without changing the existing math reward path."""
    if task_reward_mode == "length":
        return compute_length_reward(messages, length_target_words)
    interaction_kwargs = interaction_kwargs or {}
    task_type = interaction_kwargs.get("task_type")
    if task_type == "writing":
        return compute_writing_task_reward(messages, ground_truth, interaction_kwargs)
    if task_type == "bigcodebench":
        return compute_bigcodebench_task_reward(messages, ground_truth, interaction_kwargs)
    return compute_task_reward(messages, ground_truth, goal)


def _compute_f1_prediction_reward(
    messages: list,
    predictions: list[str],
) -> float:
    """
    Original word-overlap F1 proxy for prediction reward.
    """
    if not predictions or len(predictions) == 0:
        return 0.0
    user_messages = []
    for msg in messages:
        if _get_role(msg) == "user":
            user_messages.append(_get_content(msg))
    # FIX: extract clean response text from simulator JSON
    actual_responses = [_extract_user_response(m) for m in user_messages[1:]]
    num_pairs = min(len(predictions), len(actual_responses))
    if num_pairs == 0:
        return 0.0
    total_similarity = 0.0
    for i in range(num_pairs):
        pred = predictions[i]
        actual = actual_responses[i]
        if not pred or not actual:
            continue
        pred_words = set(pred.lower().split())
        actual_words = set(actual.lower().split())
        if len(pred_words) == 0 or len(actual_words) == 0:
            continue
        intersection = pred_words & actual_words
        precision = len(intersection) / len(pred_words)
        recall = len(intersection) / len(actual_words)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        total_similarity += f1
    return total_similarity / num_pairs
def _f1_pair(pred: str, actual: str) -> float:
    """Word-overlap F1 between a single prediction and a single actual response."""
    if not pred or not actual:
        return 0.0
    pred_words = set(pred.lower().split())
    actual_words = set(actual.lower().split())
    if not pred_words or not actual_words:
        return 0.0
    inter = pred_words & actual_words
    precision = len(inter) / len(pred_words)
    recall = len(inter) / len(actual_words)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
def _compute_f1_prediction_reward_per_pair(messages: list, predictions: list[str]) -> list[float]:
    """Per-turn variant of the F1 prediction reward.

    Returns one F1 score per (prediction[i], actual_response[i]) pair, where
    actual_response[i] is the user turn that followed assistant turn i. The
    conversation-level F1 reward is just the mean of this list.
    """
    if not predictions:
        return []
    user_messages = [_get_content(m) for m in messages if _get_role(m) == "user"]
    actual_responses = [_extract_user_response(m) for m in user_messages[1:]]
    num_pairs = min(len(predictions), len(actual_responses))
    return [_f1_pair(predictions[i], actual_responses[i]) for i in range(num_pairs)]
async def compute_prediction_reward_per_turn(
    messages: list,
    predictions: list[str],
    mode: str | PredictionMode = PredictionMode.F1,
    llm_kwargs: Optional[dict] = None,
    eval_logger: Optional[EvalLogger] = None,
    eval_metadata: Optional[dict] = None,
) -> list[Optional[float]]:
    """Per-turn R_pred: one score per (prediction, actual-next-user-turn) pair.

    This is the turn-wise counterpart of compute_prediction_reward. Instead of
    averaging the per-pair scores into a single conversation reward, it returns
    the raw list so each turn's prediction reward can be placed on that turn's
    tokens.  Entries may be None for LLM/combined pairs whose API calls all
    failed (caller should treat those as 0 / gradient-neutral).
    """
    mode = PredictionMode(mode)
    llm_kwargs = llm_kwargs or {}
    # Number of scored turns = #(prediction, actual-response) pairs.
    user_messages = [_get_content(m) for m in messages if _get_role(m) == "user"]
    num_pairs = min(len(predictions), max(0, len(user_messages) - 1))
    if mode in SFT_MODES:
        # Prediction handled by the auxiliary SFT loss; contributes 0 to reward.
        return [0.0] * num_pairs
    if mode == PredictionMode.F1:
        return list(_compute_f1_prediction_reward_per_pair(messages, predictions))
    elif mode == PredictionMode.LLM:
        return await _compute_llm_prediction_reward(
            messages, predictions,
            eval_logger=eval_logger, eval_metadata=eval_metadata,
            return_per_pair=True, **llm_kwargs,
        )
    elif mode == PredictionMode.COMBINED:
        f1_scores = _compute_f1_prediction_reward_per_pair(messages, predictions)
        llm_scores = await _compute_llm_prediction_reward(
            messages, predictions,
            eval_logger=eval_logger, eval_metadata=eval_metadata,
            return_per_pair=True, **llm_kwargs,
        )
        out: list[Optional[float]] = []
        for i in range(min(len(f1_scores), len(llm_scores))):
            if llm_scores[i] is None:
                out.append(None)  # API failure for this pair
            else:
                out.append((f1_scores[i] + llm_scores[i]) / 2.0)
        return out
    else:
        raise ValueError(f"Unknown prediction mode: {mode}")
def _effort_length(text: str, tokenizer=None, token_units: str = "words") -> int:
    """Length of `text` for the effort terms, in words or real tokens.

    Until 2026-09-05 both effort terms counted whitespace words. Words are not
    a stable unit across roles: measured on run 967309, the simulator's replies
    ran 1.30 tokens/word while the policy's LaTeX-heavy drafts ran 1.85, so a
    single threshold meant two different budgets depending on who was talking,
    and neither was comparable to the generation caps (`pred_answer_max_tokens`
    and friends), which are in real tokens.

    `token_units="words"` is the default so every launcher that has not opted in
    keeps its historical thresholds. Falls back to words if the tokenizer is
    missing or raises, so the reward can never fail on a length count.
    """
    if token_units != "tokens" or tokenizer is None:
        return len(text.split())
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:  # tokenizer unavailable in this worker / bad input
        return len(text.split())


def compute_effort_reward_per_turn(
    messages: list,
    max_turns: int,
    max_user_tokens: int,
    alpha: float = 0.5,
    max_assistant_tokens: int = 0,
    beta: float = 0.0,
    *,
    tokenizer=None,
    token_units: str = "words",
    assistant_extra_texts: list[list[str]] | None = None,
) -> list[float]:
    """Per-turn R_effort: one effort score per assistant turn.

    The conversation-level effort (compute_effort_reward) rewards using fewer
    turns and fewer user tokens overall.  The per-turn form makes that signal
    local so it can be credited to each turn's tokens:

        effort_i = alpha   * max(0, 1 - (i+1)/N)          # later turns worth less
                 + (1-a-b) * max(0, 1 - tau_user_i/T_user) # shorter elicited user reply
                 + beta    * max(0, 1 - tau_asst_i/T_asst)  # shorter assistant turn

    where tau_user_i is the length of the user response elicited by assistant
    turn i (i.e. the (i+1)-th user message), tau_asst_i the length of assistant
    turn i. With beta=0 (default) the assistant-token term is off.

    `assistant_extra_texts[i]` holds text the policy GENERATED for turn i but did
    not send -- in sft_v7 the superseded drafts of the inner revision loop. Their
    lengths are added to tau_asst_i, so a round that burns three revisions to
    arrive at the same paragraph costs more than one that gets there directly.
    Without it only the shown message is charged and the revision loop is free,
    which is how the inner=3 arms came to spend 2,948 revision slots to change
    288 of them (an earlier default). Omit it, or pass None,
    and the historical shown-text-only behaviour is unchanged.
    """
    assistant_messages = [_get_content(m) for m in messages if _get_role(m) == "assistant"]
    user_messages = [_get_content(m) for m in messages if _get_role(m) == "user"]
    actual_responses = [_extract_user_response(m) for m in user_messages[1:]]
    user_weight = 1.0 - alpha - beta
    out: list[float] = []
    for i in range(len(assistant_messages)):
        turn_eff = max(0.0, 1.0 - (i + 1) / max_turns) if max_turns > 0 else 0.0
        tau_user = (
            _effort_length(actual_responses[i], tokenizer, token_units)
            if i < len(actual_responses) else 0
        )
        user_eff = max(0.0, 1.0 - tau_user / max_user_tokens) if max_user_tokens > 0 else 0.0
        tau_asst = _effort_length(assistant_messages[i], tokenizer, token_units)
        if assistant_extra_texts is not None and i < len(assistant_extra_texts):
            tau_asst += sum(
                _effort_length(t, tokenizer, token_units)
                for t in (assistant_extra_texts[i] or ())
            )
        asst_eff = max(0.0, 1.0 - tau_asst / max_assistant_tokens) if max_assistant_tokens > 0 else 0.0
        out.append(alpha * turn_eff + user_weight * user_eff + beta * asst_eff)
    return out


def unsent_round_texts(v7_rounds: list) -> list[list[str]]:
    """Per assistant turn, the drafts the policy generated but did not send.

    One entry per v7 round, aligned with assistant turns (a v7 round IS an
    assistant turn). On a DRAFT round the inner loop's last draft becomes the
    visible message, so only `drafts[:-1]` are unsent; on ASK and COMMIT rounds
    the visible text is not a draft at all (an ASK round carries an ask plus its
    assumption, and both carry `drafts == []` in practice), so every draft
    present is unsent. Checking `drafts[-1] == visible` rather than trusting the
    decision label keeps the two cases from double-counting the shown text.
    """
    out: list[list[str]] = []
    for rd in (v7_rounds or []):
        drafts = [d for d in (rd.get("drafts") or []) if d]
        visible = rd.get("visible") or ""
        out.append(drafts[:-1] if (drafts and drafts[-1] == visible) else drafts)
    return out
async def _compute_llm_prediction_reward(
    messages: list,
    predictions: list[str],
    together_model: str | list[str] = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    together_api_key: Optional[str] = None,
    evaluator_prompt_template: Optional[str] = None,
    max_concurrent: int = 8,
    num_retries: int = 4,
    eval_logger: Optional[EvalLogger] = None,
    eval_metadata: Optional[dict] = None,
    return_per_pair: bool = False,
) -> Optional[float] | list[Optional[float]]:
    """
    LLM-as-judge prediction reward via Together API.
    
    Sends each (prediction, actual) pair to an external LLM for semantic
    similarity scoring on a 0-1 scale.  Uses round-robin key rotation
    across all available TOGETHER_API_KEY* env vars to distribute rate
    limit pressure.
    
    Args:
        messages: Full conversation history.
        predictions: List of predicted user responses from the assistant.
        together_model: Together API model identifier for the evaluator, or a
            list of identifiers. When a list is given, scoring requests are
            round-robined across the models to spread per-model user rate
            limits (in addition to the per-key rotation below).
        together_api_key: Together API key (falls back to TOGETHER_API_KEY* env vars).
        evaluator_prompt_template: Custom prompt template (uses default if None).
        max_concurrent: Max concurrent API calls (for rate limiting).
        num_retries: Number of retries with exponential backoff per pair.
        eval_logger: Optional EvalLogger instance for saving evaluations to JSONL.
        eval_metadata: Optional dict of extra fields to include in each log record
                       (e.g. goal, data_source, ground_truth).
    
    Returns:
        Average score in [0, 1] on success, or None if ALL pairs failed
        due to API errors (signaling the data point should be skipped).
    """
    import together
    if not predictions or len(predictions) == 0:
        return [] if return_per_pair else 0.0
    user_messages = []
    for msg in messages:
        if _get_role(msg) == "user":
            user_messages.append(_get_content(msg))
    # FIX: extract clean response text from simulator JSON
    actual_responses = [_extract_user_response(m) for m in user_messages[1:]]
    num_pairs = min(len(predictions), len(actual_responses))
    if num_pairs == 0:
        return [] if return_per_pair else 0.0
    # ── Build key rotator from explicit key + env vars ──
    key_rotator = KeyRotator.from_env(explicit_key=together_api_key)
    if key_rotator is None:
        logger.warning("No Together API key found. Falling back to F1 prediction reward.")
        if return_per_pair:
            return _compute_f1_prediction_reward_per_pair(messages, predictions)
        return _compute_f1_prediction_reward(messages, predictions)
    # ── Build model rotator: round-robin across one or more judge models ──
    # Spreads load so no single Together model hits its per-user rate limit.
    model_list = [together_model] if isinstance(together_model, str) else list(together_model)
    model_list = [m for m in model_list if m]  # drop empty/None entries
    if not model_list:
        model_list = ["meta-llama/Llama-3.3-70B-Instruct-Turbo"]
    model_rotator = KeyRotator(model_list)  # generic thread-safe round-robin over strings
    logger.info(
        f"LLM prediction scoring with {len(key_rotator)} API key(s), "
        f"{len(model_rotator)} model(s) [{', '.join(model_list)}], "
        f"max_concurrent={max_concurrent}"
    )
    prompt_template = evaluator_prompt_template or DEFAULT_LLM_EVALUATOR_PROMPT
    semaphore = asyncio.Semaphore(max_concurrent)
    metadata = eval_metadata or {}
    # logger.warning(f"predictions: {predictions}")
    # logger.warning(f"actual_responses: {actual_responses}")
    async def _score_single_pair(idx: int) -> Optional[float]:
        """Returns score on success, None on API failure (to be skipped)."""
        pred = predictions[idx]
        actual = actual_responses[idx]
        if not pred or not actual:
            return 0.0  # Empty input is a valid 0, not an API failure
        # Build conversation context up to the turn before this prediction
        # predictions[i] predicts user turn i+1, so context is up to user turn i
        conversation_context = _format_conversation_context(messages, up_to_turn=idx + 1)
        prompt = prompt_template.format(
            conversation_context=conversation_context,
            predicted_response=pred,
            actual_response=actual,
        )
        # logger.warning(f"prompt: {prompt}")
        # ── Retry with exponential backoff ──
        for attempt in range(num_retries):
            async with semaphore:
                try:
                    # Pick next key + model in rotation — spreads load across
                    # accounts and across judge models (per-model rate limits)
                    client = together.AsyncTogether(api_key=key_rotator.next())
                    model = model_rotator.next()
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                        temperature=0.1,
                    )
                    output = response.choices[0].message.content.strip()
                    # Parse "SCORE: <value>" from the last line
                    score_match = re.search(r"SCORE:\s*([\d.]+)", output, re.IGNORECASE)
                    if score_match:
                        score = float(score_match.group(1))
                        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                    else:
                        logger.warning(
                            f"Could not parse score from LLM output for pair {idx}: {output}"
                        )
                        score = 0.0
                    # Extract rationale (everything before the SCORE: line)
                    rationale = re.sub(
                        r"SCORE:\s*[\d.]+\s*$", "", output, flags=re.IGNORECASE
                    ).strip()
                    # ── Log evaluation record ──
                    if eval_logger is not None:
                        # Compute F1 for this pair as an extra signal
                        pred_words = set(pred.lower().split())
                        actual_words = set(actual.lower().split())
                        f1 = 0.0
                        if pred_words and actual_words:
                            inter = pred_words & actual_words
                            p = len(inter) / len(pred_words)
                            r = len(inter) / len(actual_words)
                            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                        record = {
                            "turn_index": idx,
                            "conversation_context": conversation_context,
                            "predicted_response": pred,
                            "actual_response": actual,
                            "evaluator_model": model,
                            "evaluator_prompt": prompt,
                            "evaluator_rationale": rationale,
                            "score": score,
                            "f1_score": round(f1, 4),
                            **metadata,
                        }
                        eval_logger.log(record)
                    return score
                except Exception as e:
                    if attempt < num_retries - 1:
                        backoff = min(2 ** attempt, 60)
                        logger.warning(
                            f"Attempt {attempt + 1}/{num_retries} failed for pair {idx}: {e}. "
                            f"Retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(
                            f"All {num_retries} attempts failed for prediction pair {idx}: {e}. "
                            f"Returning None (will be skipped in aggregation)."
                        )
                        return None  # Signal: API failure, skip this pair
        return None  # Should not reach here, but defensive
    # Run all scoring tasks concurrently (bounded by semaphore)
    raw_scores = await asyncio.gather(*[_score_single_pair(i) for i in range(num_pairs)])
    # Per-turn caller wants the raw per-pair scores (None = API failure for that
    # pair) rather than a single conversation-level average.
    if return_per_pair:
        return list(raw_scores)
    # Filter out None (API failures) — only average successfully scored pairs
    valid_scores = [s for s in raw_scores if s is not None]
    num_failed = len(raw_scores) - len(valid_scores)
    if num_failed > 0:
        logger.warning(
            f"LLM prediction scoring: {num_failed}/{len(raw_scores)} pairs failed "
            f"(API errors), excluded from average."
        )
    if len(valid_scores) == 0:
        # ALL pairs failed — signal upstream to skip this data point entirely
        logger.error(
            f"All {len(raw_scores)} LLM prediction pairs failed. "
            f"Returning None to signal data point should be skipped."
        )
        return None
    return sum(valid_scores) / len(valid_scores)
async def compute_prediction_reward(
    messages: list,
    predictions: list[str],
    mode: str | PredictionMode = PredictionMode.F1,
    llm_kwargs: Optional[dict] = None,
    eval_logger: Optional[EvalLogger] = None,
    eval_metadata: Optional[dict] = None,
) -> Optional[float]:
    """
    R_pred (Eq. 3): Prediction accuracy reward.
    Args:
        messages: Full conversation history.
        predictions: List of predicted user responses from the assistant.
        mode: Which prediction reward variant to use:
            - "f1":       Word-overlap F1 (fast, no API call)
            - "llm":      LLM-as-judge via Together API
            - "combined": Average of F1 and LLM scores
        llm_kwargs: Keyword arguments forwarded to the LLM evaluator:
            - together_model (str): Model ID for Together API
            - together_api_key (str): API key (or set TOGETHER_API_KEY env var)
            - evaluator_prompt_template (str): Custom evaluator prompt
            - max_concurrent (int): Max concurrent API calls
            - num_retries (int): Retries per pair with exponential backoff
        eval_logger: Optional EvalLogger for saving LLM evaluations to JSONL.
        eval_metadata: Optional dict of extra fields for each log record.
    Returns:
        Score in [0, 1] on success, or None if LLM scoring failed entirely
        (signaling the data point should be skipped).
    """
    mode = PredictionMode(mode)
    llm_kwargs = llm_kwargs or {}
    if mode in SFT_MODES:
        # Prediction is handled by the auxiliary SFT loss on the actor, not by
        # the reward. Contribute nothing to the scalar (and make no API call).
        return 0.0
    if mode == PredictionMode.F1:
        return _compute_f1_prediction_reward(messages, predictions)
    elif mode == PredictionMode.LLM:
        return await _compute_llm_prediction_reward(
            messages, predictions,
            eval_logger=eval_logger, eval_metadata=eval_metadata,
            **llm_kwargs,
        )
    elif mode == PredictionMode.COMBINED:
        f1_score = _compute_f1_prediction_reward(messages, predictions)
        llm_score = await _compute_llm_prediction_reward(
            messages, predictions,
            eval_logger=eval_logger, eval_metadata=eval_metadata,
            **llm_kwargs,
        )
        if llm_score is None:
            # LLM failed entirely — can't compute combined, signal skip
            return None
        return (f1_score + llm_score) / 2.0
    else:
        raise ValueError(f"Unknown prediction mode: {mode}")
def compute_effort_reward(
    messages: list,
    max_turns: int,
    max_user_tokens: int,
    alpha: float = 0.5,
    max_assistant_tokens: int = 0,
    beta: float = 0.0,
    skip_first_user_turn: bool = False,
    *,
    tokenizer=None,
    token_units: str = "words",
) -> float:
    """
    R_effort (Eq. 4, extended):
    R_effort = alpha * (1 - n/N)
             + (1-alpha-beta) * (1 - tau_user/T_user)
             + beta * (1 - tau_asst/T_asst)

    When beta=0 and max_assistant_tokens=0 (defaults), this reduces to the
    original two-term formula.

    Args:
        messages: Full conversation history.
        max_turns: Maximum number of user turns (N).
        max_user_tokens: Maximum user token budget (T_user).
        alpha: Weight for turn efficiency term.
        max_assistant_tokens: Maximum assistant token budget (T_asst). 0 disables.
        beta: Weight for assistant token efficiency term.
        skip_first_user_turn: Exclude the first user message from n and
            tau_user. Set for spec tasks where q0 is the dataset's
            problem description — a fixed input the policy cannot influence,
            whose length would otherwise inject per-sample reward noise.
            (The per-turn variant already skips q0 by construction.)
    """
    user_messages = [m for m in messages if _get_role(m) == "user"]
    if skip_first_user_turn and user_messages:
        user_messages = user_messages[1:]
    assistant_messages = [m for m in messages if _get_role(m) == "assistant"]
    n = len(user_messages)
    tau_user = sum(
        _effort_length(_get_content(msg), tokenizer, token_units) for msg in user_messages
    )
    tau_asst = sum(
        _effort_length(_get_content(msg), tokenizer, token_units)
        for msg in assistant_messages
    )
    turn_efficiency = max(0.0, 1.0 - n / max_turns) if max_turns > 0 else 0.0
    user_token_efficiency = max(0.0, 1.0 - tau_user / max_user_tokens) if max_user_tokens > 0 else 0.0
    asst_token_efficiency = max(0.0, 1.0 - tau_asst / max_assistant_tokens) if max_assistant_tokens > 0 else 0.0
    logger.debug(f"turn_efficiency={turn_efficiency:.3f} (n={n}, N={max_turns})")
    logger.debug(f"user_token_efficiency={user_token_efficiency:.3f} (tau={tau_user}, T={max_user_tokens})")
    logger.debug(f"asst_token_efficiency={asst_token_efficiency:.3f} (tau={tau_asst}, T={max_assistant_tokens})")
    user_token_weight = 1.0 - alpha - beta
    return alpha * turn_efficiency + user_token_weight * user_token_efficiency + beta * asst_token_efficiency



# ──────────────────────────────────────────────────────────────────────
# Combined reward function (called during rollout)
# ──────────────────────────────────────────────────────────────────────
async def ctom_reward_func(
    data_source: str,
    messages: list,
    ground_truth: str,
    goal: str,
    predictions: list[str],
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Compute the combined CToM reward: R = R_task + R_pred + R_effort
    Returns dict with keys "task", "prediction", "effort", and optionally
    "skipped" (bool tensor) when LLM prediction scoring failed entirely
    due to API errors — signaling this data point should be made
    gradient-neutral in downstream aggregation.
    Args:
        data_source: Dataset identifier
        messages: Full conversation history
        ground_truth: Reference answer
        goal: "solving"
        predictions: List of predicted user responses from the assistant
        **kwargs: max_turns, max_user_tokens, alpha, prediction_mode, llm_kwargs
    """
    max_turns = kwargs.get("max_turns", 10)
    max_user_tokens = kwargs.get("max_user_tokens", 500)
    alpha = kwargs.get("alpha", 0.5)
    max_assistant_tokens = kwargs.get("max_assistant_tokens", 0)
    beta = kwargs.get("beta", 0.0)
    prediction_mode = kwargs.get("prediction_mode", PredictionMode.F1)
    llm_kwargs = kwargs.get("llm_kwargs", {})
    eval_logger = kwargs.get("eval_logger", None)
    interaction_kwargs = kwargs.get("interaction_kwargs", {})
    # Build metadata for eval log records
    eval_metadata = {
        "goal": goal,
        "ground_truth": ground_truth,
        "data_source": str(data_source),
        "num_turns": sum(1 for m in messages if _get_role(m) == "user"),
    }
    r_task = await compute_task_reward_for_example_async(
        messages, ground_truth, goal, interaction_kwargs,
        task_reward_mode=kwargs.get("task_reward_mode", "math"),
        length_target_words=kwargs.get("length_target_words", 3000),
    )
    r_pred = await compute_prediction_reward(
        messages, predictions, mode=prediction_mode, llm_kwargs=llm_kwargs,
        eval_logger=eval_logger, eval_metadata=eval_metadata,
    )
    # A replayed q0 (every hidden-spec family always; math with the
    # interaction's replay_q0, default on) is the task statement, not user
    # effort — detect it by comparing the first user message to
    # single_turn_prompt and exclude it.
    first_user_content = next(
        (_get_content(m) for m in messages if _get_role(m) == "user"), None
    )
    q0_replayed = interaction_kwargs.get("task_type") in SPEC_TASK_TYPES or (
        first_user_content is not None
        and first_user_content.strip()
        == str(interaction_kwargs.get("single_turn_prompt", "")).strip()
    )
    r_effort = compute_effort_reward(
        messages, max_turns, max_user_tokens, alpha, max_assistant_tokens, beta,
        skip_first_user_turn=q0_replayed,
    )
    skipped = r_pred is None
    if skipped:
        logger.error(
            f"Reward component unavailable (API failure) for data_source={data_source}, "
            f"goal={goal} (prediction unavailable). "
            f"Marking as skipped."
            f"Messages:{messages}"
        )
        r_pred = 0.0  # Placeholder — will be handled by caller
    return {
        "task": torch.tensor(r_task, dtype=torch.float32),
        "prediction": torch.tensor(r_pred, dtype=torch.float32),
        "effort": torch.tensor(r_effort, dtype=torch.float32),
        "skipped": torch.tensor(skipped, dtype=torch.bool),
    }
@register("ctom")
class CToMRewardManager(RewardManagerBase):
    """
    Reward Manager for CToM multi-turn collaborative training.
    Combines three reward components:
        R = w_task * R_task + w_pred * R_pred + w_effort * R_effort
    Config for prediction mode (in reward_kwargs):
        prediction_mode: "f1" | "llm" | "combined"
        llm_kwargs:
            together_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo"
                # or a list to round-robin judges across per-model rate limits:
                # ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "zai-org/GLM-5.2", "openai/gpt-oss-120b"]
            together_api_key: null  # or set TOGETHER_API_KEY env var
            evaluator_prompt_template: null  # uses default if not set
            max_concurrent: 8
            num_retries: 4
        eval_log_path: "logs/llm_eval_{job_id}.jsonl"  # null to disable
    """
    def __init__(
        self,
        config,
        tokenizer: PreTrainedTokenizer,
        reward_fn_key: str = "data_source",
        compute_score: Optional[Callable] = None,
        normalize_by_data_source: bool = False,
        reward_router_address=None,
        reward_model_tokenizer=None,
        **kwargs
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.num_examine = kwargs.get("num_examine", 1)
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        # print("\n\nreward_kwargs:", config)
        reward_kwargs = config.reward.get("reward_kwargs", {})
        if not reward_kwargs:
            reward_kwargs = config.reward.reward_model.get("reward_kwargs", {})
        print("\n\nreward_kwargs:", reward_kwargs)
        self.reward_weights = reward_kwargs['reward_weights']
        self.effort_kwargs = reward_kwargs['effort_kwargs']
        self.max_turns = self.effort_kwargs.get("max_turns", 10)
        self.max_user_tokens = self.effort_kwargs.get("max_user_tokens", 500)
        self.alpha = self.effort_kwargs.get("alpha", 0.5)
        self.max_assistant_tokens = self.effort_kwargs.get("max_assistant_tokens", 0)
        self.beta = self.effort_kwargs.get("beta", 0.0)
        # Units for the two length terms. "words" is the historical default and
        # is kept so pre-2026-09-05 launchers reproduce exactly; "tokens" uses
        # the policy tokenizer, which is the only unit comparable to the
        # generation caps. See _effort_length().
        self.effort_token_units = str(
            self.effort_kwargs.get("token_units", "words")
        ).lower()
        if self.effort_token_units not in ("words", "tokens"):
            raise ValueError(
                "effort_kwargs.token_units must be 'words' or 'tokens', "
                f"got {self.effort_token_units!r}"
            )
        # user_weight = 1 - alpha - beta is the weight on the elicited user
        # reply. Negative means longer user replies would be REWARDED, which
        # silently inverts the whole effort term, so refuse it here rather than
        # discover it in a trained policy.
        if self.alpha + self.beta > 1.0 + 1e-9:
            raise ValueError(
                f"effort_kwargs alpha + beta must be <= 1.0 (got alpha={self.alpha}, "
                f"beta={self.beta}); the user-reply weight 1-alpha-beta would go negative"
            )
        if self.beta > 0.0 and not self.max_assistant_tokens:
            raise ValueError(
                "effort_kwargs.beta > 0 requires max_assistant_tokens > 0; "
                "otherwise the assistant-length term is identically 0 and beta "
                "only shrinks the user-reply weight"
            )
        # User policy (2026-09-21): only user-visible assistant text contributes
        # to efficiency. Reject stale training recipes; the pure effort helper
        # still accepts extra text for explicitly labelled historical audits.
        # Parsed defensively: the launcher passes this through hydra as an
        # unquoted true/false, which OmegaConf resolves to a real bool -- but a
        # quoted value, or an env-var default that survives as text, would arrive
        # as the STRING "false", and bool("false") is True. That would silently
        # turn the charge ON for every run that never asked for it.
        _crt = self.effort_kwargs.get("count_revision_tokens", False)
        if isinstance(_crt, str):
            _crt = _crt.strip().lower() in ("1", "true", "yes", "on")
        self.count_revision_tokens = bool(_crt)
        if self.count_revision_tokens:
            raise ValueError(
                "Revision text must be excluded from assistant efficiency: "
                "set effort_kwargs.count_revision_tokens=false. "
                "Revision-inclusive checkpoints require a fresh corrected training run."
            )
        # How the per-turn effort score enters the summed turn reward.
        #   "bonus" (default, historical): r_i += w_effort * effort_i.  Because
        #       effort_i >= 0 always (both its terms are max(0, .)) and the turn
        #       rewards are SUMMED into the return, this is a survival bonus:
        #       every extra turn adds reward, so a policy is paid to keep the
        #       conversation alive.  It is only safe when the achievable effort
        #       sum stays small next to the task term -- true at max_turns=4
        #       (effort-sum : task = 0.65 : 1) and false at max_turns=10
        #       (2.89 : 1).
        #   "cost": r_i -= w_effort * (1 - effort_i).  Every turn subtracts, a
        #       solve adds w_task * R_task once, so solve-early > solve-late >
        #       stall regardless of the turn budget.
        self.effort_mode = str(
            reward_kwargs.get("effort_mode", "bonus")
        ).lower()
        if self.effort_mode not in ("bonus", "cost"):
            raise ValueError(
                f"effort_mode must be 'bonus' or 'cost', got {self.effort_mode!r}"
            )
        # Prediction mode configuration
        self.prediction_mode = PredictionMode(
            reward_kwargs.get("prediction_mode", "f1")
        )
        self.invalid_phase_penalty = float(reward_kwargs.get("invalid_phase_penalty", -1.0))
        # Reward granularity: how the scalar reward is distributed over response
        # tokens.
        #   "conversation" (default): one scalar R = w_task*R_task + w_pred*R_pred
        #       + w_effort*R_effort placed on the LAST response token (original).
        #   "turn": per-turn rewards placed on EACH assistant turn's last token —
        #       prediction and effort are credited per turn, task stays terminal
        #       (added to the final turn). GAE then propagates each turn's reward
        #       back through that turn's tokens.
        self.reward_granularity = str(
            reward_kwargs.get("reward_granularity", "conversation")
        ).lower()
        if self.reward_granularity not in ("conversation", "turn"):
            raise ValueError(
                f"reward_granularity must be 'conversation' or 'turn', "
                f"got {self.reward_granularity!r}"
            )
        # The revision lengths are only read on the turn-wise path, which is the
        # only one that sees v7_rounds. Refuse the combination rather than let the
        # flag be silently inert -- that failure mode has cost this project whole
        # runs before (the aux-coef schedule, the aux-loss localizer skip).
        if self.count_revision_tokens and self.reward_granularity != "turn":
            raise ValueError(
                "effort_kwargs.count_revision_tokens requires "
                "reward_granularity='turn'; on the conversation path the "
                "per-round drafts are never read and the flag would do nothing"
            )
        # Where the per-turn rewards SIT, holding their total fixed:
        #   "turn"     — on each turn's last assistant token (default).
        #   "terminal" — summed onto the final turn boundary.
        # Isolates credit placement from reward magnitude, which reward_granularity
        # cannot do (conversation mode also switches effort from a per-turn SUM to
        # a mean, so it changes the total). Inert under GRPO by construction.
        self.reward_placement = str(
            reward_kwargs.get("reward_placement", "turn")
        ).lower()
        if self.reward_placement not in ("turn", "terminal"):
            raise ValueError(
                f"reward_placement must be 'turn' or 'terminal', "
                f"got {self.reward_placement!r}"
            )
        if self.prediction_mode == PredictionMode.SFT_V3 and self.reward_granularity != "turn":
            raise ValueError(
                "sft_v3 requires reward_granularity='turn' so prediction NLL "
                "and invalid-phase penalties are credited to the matching answer"
            )
        if self.effort_mode == "cost" and self.reward_granularity != "turn":
            # Conversation granularity contributes a single effort term, so
            # -(1 - effort) is just effort shifted by a constant: a no-op for
            # GRPO (group-normalized) and absorbed by the critic under GAE.
            # Reject rather than silently doing nothing.
            raise ValueError(
                "effort_mode='cost' only changes behaviour under "
                "reward_granularity='turn'; with 'conversation' the single "
                "effort term is merely offset by a constant. Use "
                "reward_granularity='turn' or effort_mode='bonus'."
            )
        # Task-reward mode: "math" (default) scores answer correctness; "length"
        # replaces R_task with the normalized-response-length sanity check
        # (compute_length_reward) to verify the RL plumbing can optimize a
        # trivially learnable signal. Everything else in the pipeline is
        # unchanged, so a flat reward curve in length mode indicts the training
        # loop itself rather than the task reward.
        self.task_reward_mode = str(
            reward_kwargs.get("task_reward_mode", "math")
        ).lower()
        if self.task_reward_mode not in ("math", "length"):
            raise ValueError(
                f"task_reward_mode must be 'math' or 'length', "
                f"got {self.task_reward_mode!r}"
            )
        self.length_target_words = int(reward_kwargs.get("length_target_words", 3000))
        print(
            f"[CToMRewardManager] task_reward_mode={self.task_reward_mode}"
            + (
                f" (length_target_words={self.length_target_words})"
                if self.task_reward_mode == "length"
                else ""
            )
        )
        self.llm_kwargs = reward_kwargs.get("llm_kwargs", {})
        # ── Reward composition ──────────────────────────────────────────────
        #   "ctom"      (default): the existing w_task/w_pred/w_effort blend,
        #                renormalized in SFT modes. Unchanged behaviour.
        self.reward_composition = str(
            reward_kwargs.get("reward_composition", "ctom")
        ).lower()
        if self.reward_composition not in ("ctom", "voi"):
            raise ValueError(
                f"reward_composition must be 'ctom' or 'voi', "
                f"got {self.reward_composition!r}"
            )
        # ── voi: BDR value-of-information reward (the retired BDR predecessor) ──
        # Per real user turn: +alpha*delta when the reply improved the answer and
        # the counterfactual self-revision could NOT have; -beta when it could
        # (the "waited for the user" cell); -gamma for a wasted turn the model
        # had the fix for; -eps per user turn; -pi per invalid phase; terminal
        # task reward. No effort term, no prediction term, no aux loss.
        self.voi_kwargs = dict(reward_kwargs.get("voi_kwargs", {}) or {})
        if self.reward_composition == "voi":
            if self.prediction_mode != PredictionMode.SFT_V6:
                raise ValueError(
                    "reward_composition='voi' requires prediction_mode='sft_v6' "
                    f"(got {self.prediction_mode.value!r}); the VOI gate reads the "
                    "BDR round records only that mode produces."
                )
            if self.reward_granularity != "turn":
                raise ValueError("reward_composition='voi' requires reward_granularity='turn'.")
            # task_rescale: "base:scale", a 2-sequence, or a none-sentinel
            # (disabled). The sentinel test MUST come before the ":" split --
            # "none" is a non-empty string, so splitting it raises and kills
            # every RewardLoopWorker at construction (run 964216).
            rescale = self.voi_kwargs.get("task_rescale", None)
            self.voi_kwargs["task_rescale"] = (
                None if _is_none_sentinel(rescale)
                else _parse_pair(rescale, name="voi_kwargs.task_rescale")
            )
            self.voi_kwargs.setdefault("pi", -float(self.invalid_phase_penalty))
            print(f"[CToMRewardManager] reward_composition=voi kwargs={self.voi_kwargs}")
        elif self.prediction_mode == PredictionMode.SFT_V6:
            raise ValueError(
                "prediction_mode='sft_v6' requires reward_composition='voi'; the "
                "CToM composition has no term that pays for a user turn."
            )
        # Evaluation logger for reward model training data. Only the LLM-judge
        # modes produce eval records; f1 has none and sft routes prediction to
        # the actor's auxiliary loss (no reward-side evaluation).
        eval_log_path = reward_kwargs.get("eval_log_path", None)
        if eval_log_path and self.prediction_mode in (PredictionMode.LLM, PredictionMode.COMBINED):
            self.eval_logger = EvalLogger(eval_log_path)
        else:
            self.eval_logger = None
        # Skipped/empty sample logger for debugging — always enabled.
        # Derives path from eval_log_path (sibling file) or falls back to cwd.
        if eval_log_path:
            skip_log_path = str(Path(eval_log_path).parent / "skipped_samples.jsonl")
        else:
            skip_log_path = "skipped_samples.jsonl"
        self.skip_logger = EvalLogger(skip_log_path)
        print(f"[CToMRewardManager] Skipped sample log: {skip_log_path}")
        # ── BDR phase trace ────────────────────────────────────────────────
        # The draft and belief are generated, scored and DISCARDED: the
        # simulator only ever sees the visible message, and nothing else
        # persists them. On smoke 964600 that made it impossible to review why
        # a turn chose ASK over COMMIT -- the reasoning behind the decision was
        # gone. This writes one JSON line per scored episode carrying every
        # phase's text alongside the reward decomposition, so a run can be
        # read turn by turn afterwards. Off unless a path is resolvable.
        self.bdr_trace_logger = None
        if self.reward_composition == "voi":
            # Config first (reaches every Ray worker), then env, then a
            # sibling of the eval log. An env var set on the driver does not
            # reach worker nodes started by a separate srun, so the launcher
            # passes reward_kwargs.bdr_trace_path.
            trace_path = str(reward_kwargs.get("bdr_trace_path", "") or "") \
                or os.environ.get("CTOM_BDR_TRACE", "") or (
                str(Path(eval_log_path).parent / "bdr_trace.jsonl") if eval_log_path else ""
            )
            if trace_path and str(trace_path).lower() not in ("0", "off", "none"):
                # ONE FILE PER WORKER. Reward workers run in separate processes
                # across nodes; a shared append target would interleave records
                # far larger than the atomic-write size and corrupt lines.
                base = Path(trace_path)
                tag = f"{socket.gethostname()}.{os.getpid()}"
                trace_path = str(base.with_name(f"{base.stem}.{tag}{base.suffix or '.jsonl'}"))
                self.bdr_trace_logger = EvalLogger(trace_path)
                print(f"[CToMRewardManager] BDR phase trace: {trace_path}")
        self.normalize_by_data_source = normalize_by_data_source
        logger.info(
            f"CToMRewardManager initialized with prediction_mode={self.prediction_mode.value}, "
            f"reward_granularity={self.reward_granularity}, "
            f"eval_logging={'enabled' if self.eval_logger else 'disabled'}"
        )
        print(f"[CToMRewardManager] reward_granularity={self.reward_granularity}")
    def _log_skipped_sample(
        self,
        reason: str,
        data_source: str,
        goal: str,
        ground_truth: str,
        interaction_kwargs: dict,
        tool_extra_fields: dict,
        message_lst: list,
        predictions_lst: list,
        reward_score: float,
        extra_detail: str = "",
    ):
        """Dump full diagnostic info for a skipped/empty sample to JSONL."""
        # Format messages as readable text for easy inspection
        formatted_msgs = []
        for m in message_lst:
            role = _get_role(m)
            content = _get_content(m)
            formatted_msgs.append({"role": role, "content": content})

        self.skip_logger.log({
            "reason": reason,
            "reward_score": reward_score,
            "data_source": data_source,
            "goal": goal,
            "expertise": interaction_kwargs.get("expertise", ""),
            "ground_truth": ground_truth,
            "single_turn_prompt": interaction_kwargs.get("single_turn_prompt", ""),
            "connection_failed": tool_extra_fields.get("connection_failed", False),
            "empty_generation": tool_extra_fields.get("empty_generation", False),
            "n_messages": len(message_lst),
            "n_predictions": len(predictions_lst),
            "messages": formatted_msgs,
            "predictions": predictions_lst,
            "extra_detail": extra_detail,
        })

    @staticmethod
    def _dense_from_scalar(score: float, valid_len: int, total_len: int) -> torch.Tensor:
        """Conversation-wise placement: the single scalar reward on the last
        valid response token (matches verl's default rm_scores placement)."""
        v = torch.zeros(total_len, dtype=torch.float32)
        idx = int(valid_len) - 1
        idx = max(0, min(idx, total_len - 1))
        v[idx] = float(score)
        return v

    @staticmethod
    def _dense_from_turns(
        turn_rewards: list[float],
        turn_end_indices: list[int],
        total_len: int,
        placement: str = "turn",
    ) -> torch.Tensor:
        """Place the per-turn rewards in response-token space.

        placement="turn"     — each turn's reward on that turn's last assistant
                               token (the historical behaviour).
        placement="terminal" — the SAME rewards summed onto the last surviving
                               turn boundary.

        The two differ only in WHERE the credit sits; the total is identical by
        construction, so a turn-vs-terminal pair isolates credit placement with
        no change to the reward's magnitude or composition. Note that under GRPO
        they are provably equivalent — core_algos sums the dense tensor to a
        scalar and broadcasts it — so this knob only does anything under a
        position-sensitive estimator (GAE, reinforce_plus_plus).

        Indices truncated away (>= total_len) are dropped in both modes, so the
        totals stay equal even when the response was clipped.
        """
        v = torch.zeros(total_len, dtype=torch.float32)
        kept = [
            (float(r), int(idx))
            for r, idx in zip(turn_rewards, turn_end_indices)
            if idx is not None and 0 <= int(idx) < total_len
        ]
        if not kept:
            return v
        if placement == "terminal":
            v[max(idx for _, idx in kept)] += sum(r for r, _ in kept)
            return v
        for r, idx in kept:
            v[idx] += r
        return v

    async def _run_single_turnwise(
        self,
        *,
        message_lst: list,
        predictions_lst: list,
        ground_truth: str,
        goal: str,
        data_source,
        interaction_kwargs: dict,
        tool_extra_fields: dict,
        turn_end_indices: list[int],
        response_len: int,
    ) -> dict[str, Any]:
        """Turn-wise reward: prediction + effort credited per assistant turn,
        task added to the final turn. Builds a dense per-token rm_scores vector
        so GAE propagates each turn's reward through that turn's tokens."""
        w = self.reward_weights
        eval_metadata = {
            "goal": goal,
            "ground_truth": ground_truth,
            "data_source": str(data_source),
            "num_turns": sum(1 for m in message_lst if _get_role(m) == "user"),
        }
        # ── Per-turn components ──
        pred_scores = await compute_prediction_reward_per_turn(
            message_lst, predictions_lst, mode=self.prediction_mode,
            llm_kwargs=self.llm_kwargs, eval_logger=self.eval_logger,
            eval_metadata=eval_metadata,
        )
        effort_scores = compute_effort_reward_per_turn(
            message_lst, self.max_turns, self.max_user_tokens, self.alpha,
            self.max_assistant_tokens, self.beta,
            tokenizer=self.tokenizer, token_units=self.effort_token_units,
            assistant_extra_texts=(
                unsent_round_texts(tool_extra_fields.get("v7_rounds", []) or [])
                if self.count_revision_tokens else None
            ),
        )
        r_task = await compute_task_reward_for_example_async(
            message_lst, ground_truth, goal, interaction_kwargs,
            task_reward_mode=self.task_reward_mode,
            length_target_words=self.length_target_words,
        )

        # Skip only when LLM/combined scoring failed for EVERY pair (API outage).
        valid_pred = [s for s in pred_scores if s is not None]
        skipped = (
            self.prediction_mode in (PredictionMode.LLM, PredictionMode.COMBINED)
            and len(pred_scores) > 0
            and len(valid_pred) == 0
        )

        # ── Assemble per-turn reward, one entry per recorded assistant turn ──
        n_turns = len(turn_end_indices)
        is_sft = self.prediction_mode in SFT_MODES
        is_v3 = self.prediction_mode in PHASE_GATED_MODES
        phase_validity = list(tool_extra_fields.get("phase_validity", []) or [])
        weight_norm = (w["task"] + w["effort"]) if is_sft else 1.0
        weight_norm = weight_norm or 1.0
        turn_rewards: list[float] = []
        valid_turns: list[bool] = []
        effort_terms: list[float] = []
        for i in range(n_turns):
            validity = phase_validity[i] if i < len(phase_validity) else {}
            # Every phase the controller recorded must be valid. Reading the
            # *_valid keys off the dict rather than naming them keeps this in
            # step with the mode's phase count automatically: sft_v3 emits
            # answer/prediction, sft_v4 adds revision. A dict with no *_valid
            # keys at all (missing entry) stays invalid, as before.
            valid_flags = [v for k, v in validity.items() if k.endswith("_valid")]
            phase_valid = bool(valid_flags) and all(valid_flags)
            valid_turns.append((not is_v3) or phase_valid)
            if is_v3 and not phase_valid:
                # Scale before the common SFT normalization so the configured
                # penalty remains exactly invalid_phase_penalty afterward.
                turn_rewards.append(self.invalid_phase_penalty * weight_norm)
                continue
            p = pred_scores[i] if (i < len(pred_scores) and pred_scores[i] is not None) else 0.0
            e = effort_scores[i] if i < len(effort_scores) else 0.0
            # bonus: pay for being early (>=0, so summing rewards length).
            # cost:  charge for the turn taken (<=0, so summing penalizes length).
            e_term = (
                -w["effort"] * (1.0 - e)
                if self.effort_mode == "cost"
                else w["effort"] * e
            )
            effort_terms.append(e_term)
            r = e_term if is_sft else (w["prediction"] * p + e_term)
            turn_rewards.append(r)
        # An invalid/clipped terminal phase earns no task-success credit.
        if n_turns > 0 and valid_turns[-1]:
            turn_rewards[-1] += w["task"] * r_task
        if is_sft:
            turn_rewards = [r / weight_norm for r in turn_rewards]

        dense = self._dense_from_turns(
            turn_rewards,
            turn_end_indices,
            response_len,
            getattr(self, "reward_placement", "turn"),
        )
        score = float(dense.sum().item())

        # ── Logging / metrics (aggregates derived from the per-turn values) ──
        mean_pred = (sum(valid_pred) / len(valid_pred)) if valid_pred else 0.0
        mean_effort = (sum(effort_scores) / len(effort_scores)) if effort_scores else 0.0
        # How much of `reward/total` came from effort, on the same scale as the
        # score. Read against the task contribution (w_task * r_task / norm) to
        # see whether talking or solving dominates the return.
        effort_sum = sum(effort_terms) / (weight_norm if is_sft else 1.0)
        if r_task == 1.0:
            print(f"Finish task: {goal}")
        print(
            f"CToM Score (turn-wise): {score:.4f}"
            + (" [SKIPPED - API failure]" if skipped else "")
        )
        print(
            f"  Per-turn: n={n_turns}, pred={[None if (i>=len(pred_scores)) else pred_scores[i] for i in range(n_turns)]}, "
            f"effort={[round(effort_scores[i], 3) for i in range(min(n_turns, len(effort_scores)))]}, "
            f"task(terminal)={r_task}"
        )
        if skipped:
            self._log_skipped_sample(
                reason="api_failure",
                data_source=data_source,
                goal=goal,
                ground_truth=ground_truth,
                interaction_kwargs=interaction_kwargs,
                tool_extra_fields=tool_extra_fields,
                message_lst=message_lst,
                predictions_lst=predictions_lst,
                reward_score=score,
                extra_detail=f"turn-wise, prediction_mode={self.prediction_mode.value}, task={r_task}",
            )
        reward_extra_info = _reward_extra_info(
            **{
                "reward/task": float(r_task),
                "reward/prediction": float(mean_pred),
                "reward/effort": float(mean_effort),
                "reward/effort_sum": float(effort_sum),
                "reward/total": score,
                "reward/num_scored_turns": float(n_turns),
                "reward/turn_reward_mean": (score / n_turns) if n_turns > 0 else 0.0,
                "reward/skipped": bool(skipped),
                "reward/empty_generation": False,
                "acc": float(r_task),
            }
        )
        return {
            "reward_score": torch.tensor(score, dtype=torch.float32),
            "reward_extra_info": reward_extra_info,
            "token_level_scores": dense,
        }

    async def _run_single_voi(
        self,
        *,
        message_lst: list,
        ground_truth: str,
        goal: str,
        data_source,
        interaction_kwargs: dict,
        tool_extra_fields: dict,
        turn_end_indices: list[int],
        response_len: int,
        valid_response_length,
    ) -> dict[str, Any]:
        """BDR value-of-information reward (the retired BDR predecessor).

        Reads the per-visible-turn round records and the counterfactual
        self-revision drafts the agent loop attached, scores every draft with the
        domain scorer, and places one reward per visible turn on that turn's last
        policy token (terminal task reward on the last one). All arithmetic lives
        in ctom.markers.compute_voi so the eval harness can reproduce it offline.
        """
        rounds = list(tool_extra_fields.get("bdr_rounds", []) or [])
        cf_drafts = list(tool_extra_fields.get("cf_drafts", []) or [])
        internal = set(int(i) for i in (tool_extra_fields.get("internal_msg_indices", []) or []))
        task_type = str(interaction_kwargs.get("task_type") or "math")
        # The real user reply after visible turn t: the first non-internal user
        # message after that turn's message index (telemetry: assumption confirmed).
        replies: list[Optional[str]] = []
        for rec in rounds:
            start = int(rec.get("visible_msg_index", -1))
            reply = None
            if start >= 0:
                for j in range(start + 1, len(message_lst)):
                    if j in internal:
                        continue
                    if _get_role(message_lst[j]) == "user":
                        reply = _get_content(message_lst[j])
                        break
            replies.append(reply)
        kw = {k: v for k, v in self.voi_kwargs.items()
              if k in ("alpha", "beta", "gamma", "eps", "pi", "tau", "task_rescale")}
        if _is_none_sentinel(kw.get("tau")):
            kw.pop("tau", None)
        result = _markers.compute_voi(
            rounds, cf_drafts, task_type=task_type, ground_truth=ground_truth,
            interaction_kwargs=interaction_kwargs, replies=replies, **kw,
        )
        turn_rewards = list(result["turn_rewards"])
        if not rounds or not turn_end_indices:
            # No BDR round was recorded (degenerate rollout): score 0, log it.
            self._log_skipped_sample(
                reason="voi_no_rounds", data_source=data_source, goal=goal,
                ground_truth=ground_truth, interaction_kwargs=interaction_kwargs,
                tool_extra_fields=tool_extra_fields, message_lst=message_lst,
                predictions_lst=[], reward_score=0.0,
                extra_detail=f"rounds={len(rounds)} boundaries={len(turn_end_indices)}",
            )
            dense = self._dense_from_scalar(0.0, valid_response_length, response_len)
            return {
                "reward_score": torch.tensor(0.0, dtype=torch.float32),
                "reward_extra_info": _reward_extra_info(**{"reward/skipped": True}),
                "token_level_scores": dense,
            }
        n = min(len(turn_rewards), len(turn_end_indices))
        if len(turn_rewards) != len(turn_end_indices):
            logger.warning(
                f"[voi] {len(turn_rewards)} round rewards vs {len(turn_end_indices)} boundaries; "
                f"truncating to {n}"
            )
        dense = self._dense_from_turns(
            turn_rewards[:n], list(turn_end_indices)[:n], response_len,
            getattr(self, "reward_placement", "turn"),
        )
        score = float(dense.sum().item())
        r_task_raw = float(result["r_task_raw"])
        if r_task_raw >= 1.0:
            print(f"Finish task: {goal}")
        stats = result["stats"]
        print(
            f"CToM Score (voi): {score:.4f}  task={result['r_task']:.3f} raw={r_task_raw:.3f} "
            f"turns={[round(r, 3) for r in turn_rewards]} decisions="
            f"{[r.get('decision') for r in rounds]} cells="
            f"{[round(stats[f'voi/cell_{c}'], 2) for c in _markers.VOI_CELLS]} g={stats['voi/g_mean']:.2f}"
        )
        extra = {
            "reward/task": float(result["r_task"]),
            "reward/total": score,
            "reward/num_scored_turns": float(len(rounds)),
            "reward/turn_reward_mean": score / len(rounds),
            "reward/skipped": False,
            "reward/empty_generation": False,
            "acc": r_task_raw,
        }
        extra.update({k: float(v) for k, v in stats.items()})
        if self.bdr_trace_logger is not None:
            # Full per-turn record: every phase's text plus what it earned.
            self.bdr_trace_logger.log({
                "task_type": task_type,
                "ground_truth": ground_truth,
                "problem": str(interaction_kwargs.get("single_turn_prompt", "")),
                "expertise": interaction_kwargs.get("expertise", ""),
                "score": score,
                "r_task": float(result["r_task"]),
                "r_task_raw": r_task_raw,
                "b": [float(x) for x in result["b"]],
                "turn_rewards": [float(x) for x in turn_rewards],
                "cells": {c: float(stats[f"voi/cell_{c}"]) for c in _markers.VOI_CELLS},
                "g_mean": float(stats["voi/g_mean"]),
                # Nothing is truncated: the point of the trace is that every
                # phase the policy generated can be read back afterwards.
                # "turns" carries the internal REVISE turns too, so an ASK's
                # withheld draft and the belief behind it are both recoverable.
                "rounds": [
                    {
                        "draft": r.get("draft", ""),
                        "first_draft": r.get("first_draft", ""),
                        "belief": r.get("belief", ""),
                        "decision": r.get("decision", ""),
                        "decision_raw": r.get("decision_raw", ""),
                        "coerce_reason": r.get("coerce_reason", ""),
                        "ask": r.get("ask", ""),
                        "visible": r.get("visible", ""),
                        "internal_turns": int(r.get("internal_turns", 0) or 0),
                        "invalid_count": int(r.get("invalid_count", 0) or 0),
                        "turns": r.get("turns", []),
                    }
                    for r in rounds
                ],
                "cf_drafts": [list(cf or []) for cf in cf_drafts],
                "cf_traces": [list(ct or []) for ct in
                              (tool_extra_fields.get("cf_traces", []) or [])],
                "replies": list(replies),
            })
        return {
            "reward_score": torch.tensor(score, dtype=torch.float32),
            "reward_extra_info": _reward_extra_info(**extra),
            "token_level_scores": dense,
        }

    async def run_single(
        self, data: DataProto
    ) -> torch.Tensor | dict[str, Any]:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        interaction_kwargs = data_item.non_tensor_batch["interaction_kwargs"]
        tool_extra_fields = data_item.non_tensor_batch["tool_extra_fields"]
        ground_truth = interaction_kwargs["ground_truth"]
        data_source = data_item.non_tensor_batch["data_source"]
        message_lst = tool_extra_fields.get("messages", {}).get("messages", [])
        predictions_lst = tool_extra_fields.get("predictions", [])
        goal = interaction_kwargs['goal']
        # Per-token reward placement: full (padded) response width, and the
        # last-token index of each assistant turn (turn-wise reward mode).
        response_len = int(data_item.batch["responses"].shape[-1])
        turn_end_indices = list(tool_extra_fields.get("turn_end_indices", []) or [])
        # FIX: unwrap nested lists FIRST, then extract predictionf text
        if len(message_lst) == 1 and isinstance(message_lst[0], list):
            message_lst = message_lst[0]
        if len(predictions_lst) == 1 and isinstance(predictions_lst[0], list):
            predictions_lst = predictions_lst[0]
        # FIX: extract clean prediction text from assistant JSON output
        predictions_lst = [_extract_prediction_text(p) for p in predictions_lst]
        # logger.warning(f"messages: {message_lst}")
        # logger.warning(f"predictions_lst: {predictions_lst}")
        # ── Detect empty / degenerate generations and apply harsh penalty ──
        # Covers: (1) explicit empty_generation flag from agent loop (dummy response path),
        # (2) no assistant messages at all, (3) assistant messages with empty content.
        #
        # NOTE: must read the explicit flag FIRST.  When message_lst is empty (e.g.
        # connection failure at q0), the `and message_lst` guard below would silently
        # skip all checks, leaving empty_generation_flag=False and scoring 0.0 instead
        # of the intended -1.0 penalty.
        empty_generation_flag = bool(tool_extra_fields.get("empty_generation", False))
        if not empty_generation_flag and message_lst:
            assistant_msgs = [m for m in message_lst if _get_role(m) == "assistant"]
            empty_generation_flag = (
                len(assistant_msgs) == 0
                or any(not _get_content(m).strip() for m in assistant_msgs)
            )
        if empty_generation_flag:
            # Determine why it's empty
            explicit_flag = bool(tool_extra_fields.get("empty_generation", False))
            conn_failed = bool(tool_extra_fields.get("connection_failed", False))
            # An empty rollout from a user-sim infra failure (connection_failed) is
            # NOT the model's fault — make it gradient-neutral (skipped) rather than
            # applying the -1.0 penalty, which would inject false negative signal
            # during sim/proxy outages. Model-side degenerate empties still get -1.0.
            empty_penalty = 0.0 if conn_failed else -1.0
            has_assistant = any(_get_role(m) == "assistant" for m in message_lst) if message_lst else False
            detail_parts = []
            if conn_failed:
                detail_parts.append("connection_failed")
            if explicit_flag:
                detail_parts.append("explicit_empty_generation_flag")
            if not message_lst:
                detail_parts.append("no_messages")
            elif not has_assistant:
                detail_parts.append("no_assistant_messages")
            else:
                empty_asst = [_get_content(m)[:100] for m in message_lst if _get_role(m) == "assistant" and not _get_content(m).strip()]
                if empty_asst:
                    detail_parts.append(f"empty_assistant_content({len(empty_asst)})")
            self._log_skipped_sample(
                reason="empty_generation",
                data_source=data_source,
                goal=goal,
                ground_truth=ground_truth,
                interaction_kwargs=interaction_kwargs,
                tool_extra_fields=tool_extra_fields,
                message_lst=message_lst,
                predictions_lst=predictions_lst,
                reward_score=empty_penalty,
                extra_detail=" | ".join(detail_parts),
            )
            _label = "CONN-FAILED neutral (skipped)" if conn_failed else "EMPTY GENERATION PENALTY"
            print(f"CToM Score: {empty_penalty} [{_label}] detail={' | '.join(detail_parts)}")
            return {
                "reward_score": torch.tensor(empty_penalty, dtype=torch.float32),
                "reward_extra_info": _reward_extra_info(
                    **{
                        "reward/total": empty_penalty,
                        "reward/turn_reward_mean": float(empty_penalty),
                        "reward/skipped": conn_failed,
                        "reward/empty_generation": True,
                    }
                ),
                "token_level_scores": self._dense_from_scalar(
                    empty_penalty, valid_response_length, response_len
                ),
            }
        # ── BDR / VOI dispatch ──
        if self.reward_composition == "voi" and message_lst:
            return await self._run_single_voi(
                message_lst=message_lst,
                ground_truth=ground_truth,
                goal=goal,
                data_source=data_source,
                interaction_kwargs=interaction_kwargs,
                tool_extra_fields=tool_extra_fields,
                turn_end_indices=turn_end_indices,
                response_len=response_len,
                valid_response_length=valid_response_length,
            )
        # ── Turn-wise reward dispatch ──
        # When enabled and per-turn token boundaries were recorded by the agent
        # loop, score each assistant turn separately and place the rewards on
        # their respective tokens. Degenerate rollouts (no recorded turns) fall
        # through to the conversation-wise path below.
        if self.reward_granularity == "turn" and message_lst and turn_end_indices:
            return await self._run_single_turnwise(
                message_lst=message_lst,
                predictions_lst=predictions_lst,
                ground_truth=ground_truth,
                goal=goal,
                data_source=data_source,
                interaction_kwargs=interaction_kwargs,
                tool_extra_fields=tool_extra_fields,
                turn_end_indices=turn_end_indices,
                response_len=response_len,
            )
        if message_lst:
            result = await ctom_reward_func(
                data_source=data_item,
                messages=message_lst,
                ground_truth=ground_truth,
                goal=goal,
                predictions=predictions_lst,
                max_turns=self.max_turns,
                max_user_tokens=self.max_user_tokens,
                alpha=self.alpha,
                max_assistant_tokens=self.max_assistant_tokens,
                beta=self.beta,
                prediction_mode=self.prediction_mode,
                llm_kwargs=self.llm_kwargs,
                eval_logger=self.eval_logger,
                interaction_kwargs=interaction_kwargs,
                task_reward_mode=self.task_reward_mode,
                length_target_words=self.length_target_words,
                reward_composition=self.reward_composition,
            )
        else:
            self._log_skipped_sample(
                reason="no_messages",
                data_source=data_source,
                goal=goal,
                ground_truth=ground_truth,
                interaction_kwargs=interaction_kwargs,
                tool_extra_fields=tool_extra_fields,
                message_lst=message_lst,
                predictions_lst=predictions_lst,
                reward_score=0.0,
                extra_detail=f"connection_failed={tool_extra_fields.get('connection_failed', False)}",
            )
            print("No messages, skip this sample")
            result = _empty_result_dict()
        reward_extra_info = _reward_extra_info()
        score: float
        if isinstance(result, dict):
            skipped = result.get("skipped", torch.tensor(False)).item()
            if self.prediction_mode in SFT_MODES:
                # ── SFT mode: prediction is DISENTANGLED from the scalar reward ──
                # The prediction objective is optimized by an auxiliary SFT loss in
                # the actor update (teacher-forced toward the real next user turn),
                # NOT by reward shaping.  The scalar reward is therefore only
                #     R = w_task * R_task + w_effort * R_effort
                # renormalized by (w_task + w_effort) so the reward scale is
                # invariant to whatever w_pred is set to.  `reward/prediction` is
                # still reported below (==0 here) for a consistent metric schema.
                components = ["task", "effort"]
                weight_norm = sum(self.reward_weights[c] for c in components) or 1.0
                score = sum(
                    result[comp].item() * self.reward_weights[comp]
                    for comp in components
                ) / weight_norm
            else:
                # f1 / llm / combined: prediction stays part of the scalar reward
                components = ["task", "prediction", "effort"]
                score = sum(
                    result[comp].item() * self.reward_weights[comp]
                    for comp in components
                )
            score = torch.tensor(score, dtype=torch.float32)
            if skipped:
                self._log_skipped_sample(
                    reason="api_failure",
                    data_source=data_source,
                    goal=goal,
                    ground_truth=ground_truth,
                    interaction_kwargs=interaction_kwargs,
                    tool_extra_fields=tool_extra_fields,
                    message_lst=message_lst,
                    predictions_lst=predictions_lst,
                    reward_score=score.item(),
                    extra_detail=f"prediction_mode={self.prediction_mode.value}, "
                                 f"task={result['task'].item()}, effort={result['effort'].item()}",
                )
            if result["task"].item() == 1.0:
                print(f"Finish task: {goal}")
            print(f"CToM Score: {score.item()}" + (" [SKIPPED - API failure]" if skipped else ""))
            print(f"  Components: { {k: v.item() for k, v in result.items() if k != 'skipped'} }")
            reward_extra_info["reward/task"] = result["task"].item()
            reward_extra_info["acc"] = result["task"].item()
            reward_extra_info["reward/prediction"] = result["prediction"].item()
            reward_extra_info["reward/effort"] = result["effort"].item()
            # Conversation granularity contributes a single effort term, so the
            # "sum" is that one weighted term (effort_mode='cost' is rejected
            # for this path in __init__).
            reward_extra_info["reward/effort_sum"] = (
                result["effort"].item()
                * self.reward_weights["effort"]
                / ((self.reward_weights["task"] + self.reward_weights["effort"])
                   if self.prediction_mode in SFT_MODES else 1.0)
            )
            reward_extra_info["reward/total"] = score.item()
            # Conversation-wise == one scored "turn" (the whole trajectory), so
            # the per-turn reward equals the total. Keeps the reward_extra_info
            # schema identical to the turn-wise path for batch concatenation.
            reward_extra_info["reward/num_scored_turns"] = 1.0
            reward_extra_info["reward/turn_reward_mean"] = score.item()
            reward_extra_info["reward/skipped"] = skipped
            reward_extra_info["reward/empty_generation"] = False
        else:
            score = result
            reward_extra_info["acc"] = float(score)
            reward_extra_info["reward/num_scored_turns"] = 1.0
            reward_extra_info["reward/turn_reward_mean"] = float(score)
            reward_extra_info["reward/empty_generation"] = False
        reward = score
        # Conversation-wise placement: scalar reward on the last valid token.
        token_level_scores = self._dense_from_scalar(
            float(reward.item()) if torch.is_tensor(reward) else float(reward),
            valid_response_length,
            response_len,
        )
        return {
            "reward_score": reward,
            "reward_extra_info": reward_extra_info,
            "token_level_scores": token_level_scores,
        }
