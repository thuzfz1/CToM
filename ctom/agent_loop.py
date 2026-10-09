# Copyright 2025 The CToM Authors
# Agent loop for personalized multi-turn dialogue training.
# Fully self-contained — no collabllm dependencies.
#
# Key difference from CollabLLM:
#   The assistant generates BOTH an answer (at) AND a predicted next-turn user response (et)
#   at each turn. The prediction et is later compared against the actual user response qt+1
#   to compute a DPO-style prediction loss.

import asyncio
import json
import logging
import os
import socket
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import torch

from ctom.utils import (
    SPEC_TASK_TYPES,
    extract_sim_response,
    is_valid_messages,
    parse_assistant_response,
)
from ctom.prediction_loss import (
    ANSWER_END_TAG,
    ANSWER_START_TAG,
    PREDICTION_END_TAG,
    PREDICTION_START_TAG,
    REVISION_END_TAG,
    REVISION_START_TAG,
    locate_prediction_value_token_span,
)
from ctom import markers as _markers
from ctom import controller as _ctom
from ctom.submission import (
    answer_is_final,
    compose_revision,
    forced_submit_marker,
)

from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.profiler import simple_timer
from verl.workers.rollout.schemas import Message

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# How many times to retry q0 generation when the user sim connection fails
Q0_MAX_RETRIES = int(os.getenv("CTOM_Q0_MAX_RETRIES", "3"))
# Base delay (seconds) between q0 retries — actual delay is base * 2^attempt
Q0_RETRY_BASE_DELAY = float(os.getenv("CTOM_Q0_RETRY_BASE_DELAY", "5.0"))


def parse_answer_and_prediction(content: str) -> tuple[str, str]:
    """Wrapper around JSON parser for backward compatibility."""
    parsed = parse_assistant_response(content)
    return parsed["answer"], parsed["prediction"]


def _is_submit_task(agent_data: AgentData) -> bool:
    """Task families that end when the agent submits, with no further user turn.

    coding (submits a program) and writing (submits a document).
    Both replay q0 and hold a hidden specification; see utils.SPEC_TASK_TYPES.
    """
    return agent_data.interaction_kwargs.get("task_type") in SPEC_TASK_TYPES


def _is_submit_final(agent_data: AgentData, answer: str) -> bool:
    """Whether this episode just submitted its artifact behind the answer marker."""
    return _is_submit_task(agent_data) and answer_is_final(answer)


def _ensure_tensor(value, dtype=torch.long):
    """Convert lists to tensors defensively."""
    if isinstance(value, list):
        return torch.tensor(value, dtype=dtype)
    return value


def _clean_last_user_message(agent_data: AgentData) -> None:
    """Clean raw simulator JSON from the last user message in-place.

    The base ToolAgentLoop._handle_interacting_state may append the raw
    simulator output (which is structured JSON with current_status, thought,
    response fields) instead of the clean response string.  This helper
    extracts just the 'response' field so the assistant sees clean user text
    and the stored conversation is clean for reward computation.
    """
    if not agent_data.messages:
        return
    last_msg = agent_data.messages[-1]
    if not isinstance(last_msg, dict):
        return
    if last_msg.get("role") != "user":
        return
    content = last_msg.get("content", "")
    if not content:
        return
    cleaned = extract_sim_response(content)
    if cleaned != content:
        logger.debug(
            f"Cleaned raw simulator JSON from user message "
            f"(original len={len(content)}, cleaned len={len(cleaned)})"
        )
        last_msg["content"] = cleaned


def _make_empty_output(agent_data, tokenizer, eos_token_id=None, connection_failed=False):
    """
    Build a minimal AgentLoopOutput when no useful generation happened.

    Two distinct causes are tracked via extra_fields:
    * connection_failed=True  – infrastructure issue (vLLM server unreachable).
      The trainer MUST filter these out entirely before rollout correction;
      response_mask is all-zeros which would crash compute_rollout_correction.
    * connection_failed=False  – model-side degenerate output (empty response).
      These could carry a penalty reward but still need filtering since there
      are no real logprobs to compute importance weights on.

    In both cases response_mask=[0] — the veRL trainer must drop the sample.
    """
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", 0) or 0

    prompt_ids = _ensure_tensor(agent_data.prompt_ids)

    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=torch.tensor([eos_token_id], dtype=torch.long),
        response_mask=torch.tensor([0], dtype=torch.long),
        multi_modal_data={},
        # Must be a real (zero) tensor matching response_ids, NOT None: when
        # rollout log-probs are collected, verl's agent_loop._postprocess does
        # torch.cat([o.response_logprobs for o in outputs]) over ALL outputs
        # before any filtering — a None here crashes that cat (pred_f1), and an
        # absent key makes the batch tensordict union incompatible (baseline).
        # The token stays gradient-neutral via response_mask=[0].
        response_logprobs=torch.zeros(1, dtype=torch.float32),
        num_turns=0,
        metrics=agent_data.metrics,
        extra_fields={
            "turn_scores": agent_data.turn_scores if hasattr(agent_data, "turn_scores") else [],
            "messages": {"messages": []},
            "predictions": [],
            "turn_end_indices": list(getattr(agent_data, "turn_end_indices", [])),
            "pred_value_spans": list(getattr(agent_data, "pred_value_spans", [])),
            "prediction_context_ids": list(getattr(agent_data, "prediction_context_ids", [])),
            "phase_validity": list(getattr(agent_data, "phase_validity", [])),
            "draft_texts": list(getattr(agent_data, "draft_texts", [])),
            "counterfactual_targets": list(getattr(agent_data, "counterfactual_targets", [])),
            "bdr_rounds": list(getattr(agent_data, "bdr_rounds", [])),
            "internal_msg_indices": list(getattr(agent_data, "internal_msg_indices", [])),
            "cf_drafts": list(getattr(agent_data, "cf_drafts", [])),
            "cf_traces": list(getattr(agent_data, "cf_traces", [])),
            "v7_rounds": list(getattr(agent_data, "v7_rounds", [])),
            "empty_generation": True,
            "connection_failed": connection_failed,
        },
    )


# ── sft_v7 phase trace ───────────────────────────────────────────────────────
# `v7_rounds` carries every PLAN, decision, internal draft, anticipated user
# reply and verdict. Until 2026-09-06 it was attached to extra_fields and then
# dropped -- nothing downstream reads it -- so run 969590 finished with no
# record of WHY any round drafted, asked or revised, and the discarded drafts
# were only recoverable indirectly from the counterfactual simulator probes.
# One JSONL per rollout worker (separate processes on separate nodes; a shared
# append target interleaves records far larger than the atomic write size).
# Path: $CTOM_V7_TRACE, else beside the run's conversation log, else
# logs/run_$SLURM_JOB_ID/. Set CTOM_V7_TRACE=off to disable.
_V7_TRACE_STATE: dict = {}


def _v7_trace_path() -> Optional[Path]:
    """Resolve this worker's trace file once. Never raises."""
    if "path" in _V7_TRACE_STATE:
        return _V7_TRACE_STATE["path"]
    _V7_TRACE_STATE["path"] = None
    try:
        raw = str(os.environ.get("CTOM_V7_TRACE", "") or "")
        if raw.lower() in ("0", "off", "none", "false"):
            return None
        if not raw:
            conv = str(os.environ.get("CTOM_CONVERSATION_LOG", "") or "")
            job = str(os.environ.get("SLURM_JOB_ID", "") or "")
            if conv:
                raw = str(Path(conv).with_name("v7_trace.jsonl"))
            elif job:
                # Anchored on the package, not the worker's cwd: rollout workers
                # are started by a separate srun and need not share it.
                raw = str(Path(__file__).resolve().parents[1] / "logs" / f"run_{job}" / "v7_trace.jsonl")
            else:
                return None
        base = Path(raw)
        tag = f"{socket.gethostname()}.{os.getpid()}"
        path = base.with_name(f"{base.stem}.{tag}{base.suffix or '.jsonl'}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _V7_TRACE_STATE["path"] = path
        logger.info(f"[CToMAgentLoop] sft_v7 phase trace -> {path}")
    except Exception as exc:  # noqa: BLE001 - telemetry must never kill a rollout
        logger.warning(f"[CToMAgentLoop] sft_v7 phase trace disabled: {exc}")
    return _V7_TRACE_STATE["path"]


def _v7_trace_write(agent_data) -> None:
    """Append one record per episode: every phase text plus its validity."""
    try:
        rounds = list(getattr(agent_data, "v7_rounds", []) or [])
        if not rounds:
            return
        path = _v7_trace_path()
        if path is None:
            return
        ik = dict(getattr(agent_data, "interaction_kwargs", {}) or {})
        record = {
            "controller_version": _ctom.CONTROLLER_VERSION,
            "action_protocol": rounds[0].get("action_protocol", "legacy"),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "ground_truth": ik.get("ground_truth"),
            "request_id": str(getattr(agent_data, "request_id", "")),
            "problem": str(ik.get("single_turn_prompt", "")),
            "expertise": ik.get("expertise", ""),
            "task_type": ik.get("task_type", "math"),
            "user_turns": getattr(agent_data, "user_turns", None),
            "assistant_turns": getattr(agent_data, "assistant_turns", None),
            "phase_validity": list(getattr(agent_data, "phase_validity", []) or []),
            "counterfactual_targets": list(getattr(agent_data, "counterfactual_targets", []) or []),
            "rounds": rounds,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 - telemetry must never kill a rollout
        logger.warning(f"[CToMAgentLoop] could not write the sft_v7 phase trace: {exc}")


class _DetachedState:
    """Minimal token-state container for the sft_v6 counterfactual branch.

    Carries exactly the attributes _sample_controlled_phase /
    _append_controller_tokens and the round generator touch, starting from a
    snapshot of the real context, so the branch can generate without touching
    the rollout's AgentData. response_mask starts EMPTY on purpose: the branch
    gets a fresh budget regardless of what the real round consumed (design doc
    §3.7 — otherwise burning the internal cap and then asking would make the
    gate vacuous).
    """

    def __init__(self, agent_data: AgentData, prompt_ids: list[int]):
        self.request_id = agent_data.request_id
        self.image_data = agent_data.image_data
        self.video_data = agent_data.video_data
        self.prompt_ids = list(prompt_ids)
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.metrics: dict[str, Any] = {}
        self.extra_fields: dict[str, Any] = {}


class CToMAgentLoop(ToolAgentLoop):
    """
    Agent loop for CToM multi-turn collaborative training.

    At each turn, the assistant outputs:
        - at: the answer to the user's question
        - et: a prediction of what the user will say next

    The prediction et is stored alongside the conversation for later reward computation
    (DPO-style prediction loss).
    """

    @property
    def _pred_sft_enabled(self) -> bool:
        """True when the auxiliary predict-next-user-turn SFT loss is active
        (actor.pred_sft_coef > 0). In that case the assistant's prediction is
        learned by the SFT loss, so its tokens are masked out of the PPO loss."""
        return float(
            OmegaConf.select(self.config, "actor_rollout_ref.actor.pred_sft_coef", default=0.0)
        ) > 0.0

    @property
    def _prediction_mode(self) -> str:
        try:
            reward_kwargs = self.config.reward.get("reward_kwargs", {}) or {}
            if not reward_kwargs:
                reward_kwargs = self.config.reward.reward_model.get("reward_kwargs", {}) or {}
            return str(reward_kwargs.get("prediction_mode", "")).lower()
        except Exception:
            return ""

    @property
    def _is_controller_phased(self) -> bool:
        """True when this class owns the phase delimiters and drives generation
        itself, so the base-class single-shot generate path and its post-hoc
        prediction-span localization must both be bypassed."""
        return self._is_sft_v7

    def _append_controller_tokens(self, agent_data: AgentData, text: str) -> bool:
        """Append fixed-format tokens with PPO mask zero."""
        # q0 is appended before the first model call, when the base loop leaves
        # response_logprobs empty. Backfill user positions so the final vector
        # stays response-space aligned.
        if len(agent_data.response_logprobs) < len(agent_data.response_mask):
            missing = len(agent_data.response_mask) - len(agent_data.response_logprobs)
            agent_data.response_logprobs += [0.0] * missing
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        remaining = self.response_length - len(agent_data.response_mask)
        complete = len(token_ids) <= remaining
        token_ids = list(token_ids[: max(remaining, 0)])
        agent_data.prompt_ids = list(agent_data.prompt_ids) + token_ids
        agent_data.response_mask += [0] * len(token_ids)
        agent_data.response_logprobs += [0.0] * len(token_ids)
        return complete

    @staticmethod
    def _phase_sampling_params(
        sampling_params: dict[str, Any], max_tokens: int, stop_text: str
    ) -> dict[str, Any]:
        params = dict(sampling_params)
        params["max_tokens"] = int(max_tokens)
        existing = params.get("stop")
        if existing is None:
            stops = []
        elif isinstance(existing, str):
            stops = [existing]
        else:
            stops = list(existing)
        if stop_text not in stops:
            stops.append(stop_text)
        params["stop"] = stops
        return params

    async def _sample_controlled_phase(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        *,
        max_tokens: int,
        stop_text: str,
        ppo_mask: int,
    ) -> tuple[list[int], str, bool]:
        """Sample one bounded phase and append it with an explicit PPO mask."""
        if max_tokens <= 0:
            return [], "", True
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=list(agent_data.prompt_ids),
                sampling_params=self._phase_sampling_params(
                    sampling_params, max_tokens, stop_text
                ),
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        token_ids = list(output.token_ids)
        if hasattr(agent_data, "v7_generated_tokens"):
            agent_data.v7_generated_tokens += len(token_ids)
        agent_data.response_ids = token_ids
        agent_data.prompt_ids = list(agent_data.prompt_ids) + token_ids
        agent_data.response_mask += [int(ppo_mask)] * len(token_ids)
        if output.log_probs is None:
            agent_data.response_logprobs += [0.0] * len(token_ids)
        else:
            agent_data.response_logprobs += list(output.log_probs)

        num_preempted = output.num_preempted if output.num_preempted is not None else -1
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = num_preempted
        elif num_preempted > 0:
            agent_data.metrics["num_preempted"] += num_preempted
        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        elif output.extra_fields.get("max_global_steps") is not None:
            agent_data.extra_fields["max_global_steps"] = output.extra_fields["max_global_steps"]

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        # vLLM trims a matched stop string from the detokenized text but NOT from
        # token_ids, and we decode token_ids ourselves — so a phase that ended on
        # its stop tag carries that tag in `text`. Measured at 60.8% of turns in
        # math run 921945 before this strip.
        #
        # It is not cosmetic: `text` becomes the message the user simulator sees
        # and the counterfactual context. Some domains append [REVISION_END]
        # after the submitted function, so the code handed to check_correctness
        # does not parse. Token accounting is deliberately left alone — those
        # tokens really were generated and must keep carrying policy gradient.
        if stop_text and text.endswith(stop_text):
            text = text[: -len(stop_text)].rstrip()
        # Backends collapse both stop and length into stop_reason=completed. A
        # phase that consumes its full hard cap is conservatively marked clipped.
        clipped = len(token_ids) >= max_tokens
        return token_ids, text, clipped

    async def _generate_phased_turn(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        """Run one controller-owned round."""
        return await self._run_v7_round(agent_data, sampling_params)

    async def _internal_message_ids(self) -> list[int]:
        """Token ids of the controller's [INTERNAL] user-role message, rendered
        through the SAME chat-template path a real user turn uses, so the
        internal turn is well-formed for the model. Cached: it is constant."""
        cached = getattr(self, "_internal_ids_cache", None)
        if cached is not None:
            return list(cached)
        ids = await self.apply_chat_template(
            [{"role": "user", "content": _markers.INTERNAL_MESSAGE}],
            remove_system_prompt=True,
        )
        self._internal_ids_cache = list(ids)
        return list(ids)

    def _turn_min_tokens(self) -> int:
        """Smallest number of response tokens one more round can occupy: the
        plan and decision scaffolding plus a one-token draft. Used to refuse an
        internal turn that would leave no room for the round that follows it."""
        _len = lambda t: len(self.tokenizer(t, add_special_tokens=False)["input_ids"])  # noqa: E731
        return (
            _len(_ctom.THINK_START_TAG + "\n")
            + _len("\n" + _ctom.THINK_END_TAG + "\n" + _ctom.DECISION_TAG + " ")
            + _ctom.DECISION_MAX_TOKENS
            + _len(_ctom.DRAFT_START_TAG + "\n")
            + _len("\n" + _ctom.DRAFT_END_TAG)
            + 3
        )

    async def _append_internal_message(self, state, *, record_message: bool) -> bool:
        """Append the [INTERNAL] controller message (ppo_mask=0, user_turns
        untouched). Returns False, appending nothing, when the dialogue budget
        cannot hold the message plus one minimal turn after it."""
        ids = await self._internal_message_ids()
        remaining = self.response_length - len(state.response_mask)
        if len(ids) + self._turn_min_tokens() > remaining:
            return False
        if record_message:
            state.messages.append({"role": "user", "content": _markers.INTERNAL_MESSAGE})
            state.internal_msg_indices.append(len(state.messages) - 1)
        if len(state.response_logprobs) < len(state.response_mask):
            state.response_logprobs += [0.0] * (len(state.response_mask) - len(state.response_logprobs))
        state.prompt_ids = list(state.prompt_ids) + list(ids)
        state.response_mask += [0] * len(ids)
        state.response_logprobs += [0.0] * len(ids)
        return True

    def _bump(self, agent_data, key: str, value: float = 1.0) -> None:
        agent_data.metrics[key] = agent_data.metrics.get(key, 0.0) + float(value)

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """The simulator is rendered from the SIM-VISIBLE transcript (internal
        turns dropped). The plain baseline mode is byte-identical to the base
        class."""
        if not self._is_sft_v7:
            return await super()._handle_interacting_state(agent_data)
        visible = self._v7_sim_view(agent_data)
        should_terminate, response, reward, _metrics = await agent_data.interaction.generate_response(
            agent_data.request_id, visible, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1
        add_messages = [{"role": "user", "content": response}]
        agent_data.messages.extend(add_messages)
        if reward is not None:
            agent_data.turn_scores.append(reward)
        response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        return AgentState.TERMINATED if should_terminate else AgentState.GENERATING

    # ══════════════════════════════════════════════════════════════════════
    # sft_v7: decide / draft-predict-revise / ask-with-assumption
    # (the paper's method section). Each inner revision is a NEW assistant
    # turn; the prediction is written by the policy INSIDE a user turn that
    # begins with [ANTICIPATED]; the verdict is parsed from that text.
    # ══════════════════════════════════════════════════════════════════════
    @property
    def _is_sft_v7(self) -> bool:
        return self._prediction_mode == "sft_v7"

    def _v7_cfg(self) -> dict[str, Any]:
        sel = lambda key, default: OmegaConf.select(  # noqa: E731
            self.config, f"actor_rollout_ref.actor.{key}", default=default
        )
        draft_cap = sel("bdr_draft_max_tokens", 0) or sel("pred_answer_max_tokens", 2048)
        mode = str(sel("v7_verdict_mode", "parse")).lower()
        if mode not in _ctom.VERDICT_MODES:
            raise ValueError(
                f"v7_verdict_mode={mode!r} is not implemented; choose from {_ctom.VERDICT_MODES}"
            )
        return {
            "action_protocol": _ctom.validate_action_protocol(str(sel("v7_action_protocol", "legacy"))),
            "think_cap": int(sel("v7_think_max_tokens", 256)),
            "draft_cap": int(draft_cap),
            "pred_cap": int(sel("pred_prediction_max_tokens", 384)),
            "ask_cap": int(sel("v7_ask_max_tokens", 512)),
            "max_inner": int(sel("v7_max_inner_revisions", 3)),
            "verdict_mode": mode,
        }

    def _v7_append_ids(self, state, ids: list[int]) -> bool:
        """Append fixed token ids (chat-template header / footer) with ppo_mask 0."""
        if len(state.response_logprobs) < len(state.response_mask):
            state.response_logprobs += [0.0] * (len(state.response_mask) - len(state.response_logprobs))
        remaining = self.response_length - len(state.response_mask)
        complete = len(ids) <= remaining
        ids = list(ids[: max(remaining, 0)])
        state.prompt_ids = list(state.prompt_ids) + ids
        state.response_mask += [0] * len(ids)
        state.response_logprobs += [0.0] * len(ids)
        return complete

    def _v7_generation_prompt_ids(self) -> list[int]:
        """Token ids the chat template appends to open an assistant turn."""
        tok = self.tokenizer
        msgs = [{"role": "user", "content": "x"}]
        s0 = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        s1 = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if not s1.startswith(s0):
            raise RuntimeError("sft_v7: the chat template's generation prompt is not a suffix")
        return list(tok(s1[len(s0):], add_special_tokens=False)["input_ids"])

    async def _v7_user_slot_ids(self) -> tuple[list[int], list[int], list[int]]:
        """(header, close, gen_prompt) around a policy-written user turn. On the
        Qwen3 template: header = "<|im_start|>user\\n[ANTICIPATED]\\n", close =
        "<|im_end|>\\n", gen_prompt = "<|im_start|>assistant\\n". Derived from
        the template at runtime (common prefix / suffix of two renders) so it
        does not hard-code a layout. Cached."""
        cached = getattr(self, "_v7_slot_ids_cache", None)
        if cached is not None:
            return cached

        def _msg(s):
            return [{"role": "user", "content": f"{_ctom.ANTICIPATED_MARKER}\n{s}"}]

        a = await self.apply_chat_template(_msg("ZZZZ"), remove_system_prompt=True)
        b = await self.apply_chat_template(_msg("QQQQ"), remove_system_prompt=True)
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
                "sft_v7: could not split the user-turn footer into close + generation prompt "
                f"(suffix={suffix}, gen={gen})"
            )
        close = suffix[: len(suffix) - len(gen)]
        self._v7_slot_ids_cache = (header, close, list(gen))
        return self._v7_slot_ids_cache

    def _v7_eos_ids(self) -> set[int]:
        ids = set()
        for attr in ("eos_token_id", "pad_token_id"):
            v = getattr(self.tokenizer, attr, None)
            if isinstance(v, int) and v >= 0:
                ids.add(v)
        return ids

    def _v7_pop_trailing_eos(self, state, start: int) -> int:
        """The model may close the user turn itself; drop those generated end
        tokens (after `start` only) so the controller closes it exactly once."""
        eos = self._v7_eos_ids()
        popped = 0
        while len(state.response_mask) > start and state.prompt_ids and state.prompt_ids[-1] in eos:
            state.prompt_ids.pop()
            state.response_mask.pop()
            if state.response_logprobs:
                state.response_logprobs.pop()
            popped += 1
        return popped

    def _reward_messages(self, agent_data) -> list[Any]:
        """The transcript the reward manager scores: the simulator's view for
        sft_v7 (no anticipated turns, no internal drafts), else the raw list."""
        if self._is_sft_v7:
            return self._v7_sim_view(agent_data)
        return agent_data.messages

    def _v7_sim_view(self, agent_data) -> list[Any]:
        return _ctom.sim_view(
            agent_data.messages,
            getattr(agent_data, "v7_hidden", ()),
            getattr(agent_data, "v7_visible", {}),
        )

    async def _v7_anticipate(self, agent_data, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """One policy-written user turn: header (mask 0) -> context snapshot ->
        sampled prediction (mask 1) -> "\\n[PREDICTION_END]" + close (mask 0).
        The caller appends the generation prompt if another assistant turn
        follows. Returns {text, clipped, valid, slot}."""
        cfg = self._v7_cfg()
        header, close, _gen = await self._v7_user_slot_ids()
        end_tag = "\n" + PREDICTION_END_TAG
        end_len = len(self.tokenizer(end_tag, add_special_tokens=False)["input_ids"])
        header_ok = self._v7_append_ids(agent_data, header)
        agent_data.prediction_context_ids.append(list(agent_data.prompt_ids))
        start = len(agent_data.response_mask)
        remaining = self.response_length - len(agent_data.response_mask)
        budget = min(cfg["pred_cap"], max(0, remaining - end_len - len(close) - 1))
        _ids, text, clipped = await self._sample_controlled_phase(
            agent_data, sampling_params, max_tokens=budget,
            stop_text=PREDICTION_END_TAG, ppo_mask=1,
        )
        popped = self._v7_pop_trailing_eos(agent_data, start)
        if popped:
            self._bump(agent_data, "v7_prediction_eos_popped", popped)
        end = len(agent_data.response_mask)
        if not hasattr(agent_data, "pred_value_spans"):
            agent_data.pred_value_spans = []
        agent_data.pred_value_spans.append([start, end])
        tag_ok = self._append_controller_tokens(agent_data, end_tag)
        close_ok = self._v7_append_ids(agent_data, close)
        agent_data.messages.append({"role": "user", "content": _ctom.anticipated_user_content(text)})
        agent_data.v7_hidden.add(len(agent_data.messages) - 1)
        valid = bool(text.strip()) and not clipped and header_ok and tag_ok and close_ok
        if clipped:
            self._bump(agent_data, "v7_prediction_clipped")
        if not valid:
            self._bump(agent_data, "v7_prediction_invalid")
        return {
            "text": text,
            "clipped": clipped,
            "valid": valid,
            "slot": len(agent_data.prediction_context_ids) - 1,
        }

    async def _run_v7_round(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """One user round: think -> decision -> (draft/anticipate/verdict loop |
        ask/anticipate | commit) -> one visible message. Returns INTERACTING when
        the simulator should reply, TERMINATED on COMMIT or a cap."""
        cfg = self._v7_cfg()
        categorical = cfg["action_protocol"] == "categorical_v1"
        alphabet = _ctom.action_token_ids(lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"]) if categorical else []
        task_type = str(agent_data.interaction_kwargs.get("task_type") or "math")
        _len = lambda t: len(self.tokenizer(t, add_special_tokens=False)["input_ids"])  # noqa: E731
        header, close, gen = await self._v7_user_slot_ids()
        think_open = _ctom.THINK_START_TAG + "\n"
        decision_open = "\n" + _ctom.THINK_END_TAG + "\n" + _ctom.DECISION_TAG + " "
        draft_open = _ctom.DRAFT_START_TAG + "\n"
        draft_close = "\n" + _ctom.DRAFT_END_TAG
        ask_open = _ctom.ASK_START_TAG + "\n"
        ask_close = "\n" + _ctom.ASK_END_TAG
        pred_tail = len(header) + _len("\n" + PREDICTION_END_TAG) + len(close) + 4
        draft_block_min = _len(draft_open) + _len(draft_close) + pred_tail + 8
        min_turn = (
            _len(think_open) + _len(decision_open) + (1 if categorical else _ctom.DECISION_MAX_TOKENS) + 2 + draft_block_min
        )
        if self.response_length - len(agent_data.response_mask) < min_turn:
            self._bump(agent_data, "v7_turn_skipped_no_budget")
            return AgentState.TERMINATED

        agent_data.v7_generated_tokens = 0
        shown_before = str(getattr(agent_data, "v7_last_shown", "") or "")
        validity: dict[str, Any] = {}

        # ── think ──
        ok_think = self._append_controller_tokens(agent_data, think_open)
        remaining = self.response_length - len(agent_data.response_mask)
        think_budget = min(cfg["think_cap"], max(0, remaining - (min_turn - _len(think_open))))
        _, think, think_clipped = await self._sample_controlled_phase(
            agent_data, sampling_params, max_tokens=think_budget,
            stop_text=_ctom.THINK_END_TAG, ppo_mask=1,
        )
        validity["think_valid"] = bool(think.strip()) and not think_clipped and ok_think
        if think_clipped:
            self._bump(agent_data, "v7_think_clipped")

        # ── decision ──
        ok_dec = self._append_controller_tokens(agent_data, decision_open)
        decision_offset = len(agent_data.response_mask)
        allowed_ids = alphabet if shown_before else alphabet[:2]
        decision_params = _ctom.categorical_sampling_params(sampling_params, allowed_ids) if categorical else sampling_params
        decision_ids, decision_raw, _ = await self._sample_controlled_phase(
            agent_data, decision_params, max_tokens=1 if categorical else _ctom.DECISION_MAX_TOKENS,
            stop_text="\n", ppo_mask=1,
        )
        parsed = _ctom.parse_action_token(decision_ids, alphabet, bool(shown_before)) if categorical else _ctom.parse_decision(decision_raw)
        decision, coerce_reason = _ctom.coerce_decision(parsed, has_shown_draft=bool(shown_before))
        nl_ok = self._append_controller_tokens(agent_data, "\n")
        validity["decision_valid"] = parsed is not None and ok_dec and nl_ok
        if coerce_reason == "invalid":
            self._bump(agent_data, "v7_decision_invalid")
        elif coerce_reason == "commit_without_draft":
            self._bump(agent_data, "v7_commit_without_draft")

        rec: dict[str, Any] = {
            "action_protocol": cfg["action_protocol"],
            "action_offset": decision_offset if categorical else None,
            "action_token_ids": (alphabet if shown_before else alphabet[:2] + [-1]) if categorical else [],
            "sampled_decision": parsed,
            "think": think, "decision": decision, "decision_raw": decision_raw,
            "coerce_reason": coerce_reason, "drafts": [], "predictions": [],
            "verdicts": [], "verdict_reasons": [], "ask": "", "visible": "",
            "inner_iters": 0, "slots": [],
        }
        round_index = len(agent_data.v7_rounds)
        visible = ""

        if decision == _ctom.COMMIT:
            visible = _ctom.visible_message(_ctom.COMMIT, last_shown=shown_before, task_type=task_type)
            agent_data.messages.append({"role": "assistant", "content": visible})

        if decision == _ctom.ASK:
            ok_open = self._append_controller_tokens(agent_data, ask_open)
            remaining = self.response_length - len(agent_data.response_mask)
            ask_budget = min(cfg["ask_cap"], max(0, remaining - _len(ask_close) - (0 if categorical else pred_tail) - 1))
            _, ask, ask_clipped = await self._sample_controlled_phase(
                agent_data, sampling_params, max_tokens=ask_budget,
                stop_text=_ctom.ASK_END_TAG, ppo_mask=1,
            )
            ok_close = self._append_controller_tokens(agent_data, ask_close)
            if ask_clipped:
                self._bump(agent_data, "v7_ask_clipped")
            if not ask.strip():
                # Nothing to send: fall through to a draft rather than an empty message.
                self._bump(agent_data, "v7_empty_ask")
                validity["ask_valid"] = False
                decision = _ctom.DRAFT
                rec["decision"] = _ctom.DRAFT
                rec["coerce_reason"] = "empty_ask"
            elif categorical:
                visible = ask.strip()
                agent_data.messages.append({"role": "assistant", "content": visible})
                rec["ask"] = visible
                validity["ask_valid"] = (not ask_clipped) and ok_open and ok_close
            else:
                hist = self._v7_sim_view(agent_data)
                agent_data.messages.append({"role": "assistant", "content": ask})
                q_index = len(agent_data.messages) - 1
                ant = await self._v7_anticipate(agent_data, sampling_params)
                visible = _ctom.visible_message(
                    _ctom.ASK, ask=ask, prediction=ant["text"], task_type=task_type
                )
                agent_data.v7_visible[q_index] = visible
                agent_data.v7_slots.append({
                    "kind": "ask", "round": round_index, "history": hist,
                    "text": ask, "verdict": None, "slot": ant["slot"],
                })
                rec["ask"] = ask
                rec["predictions"].append(ant["text"])
                rec["slots"].append(ant["slot"])
                validity["ask_valid"] = (not ask_clipped) and ok_open and ok_close
                validity["prediction_valid"] = ant["valid"]

        if decision == _ctom.DRAFT:
            j = 0
            last_draft: Optional[str] = None
            draft = ""
            d_index = -1
            while True:
                ok_open = self._append_controller_tokens(agent_data, draft_open)
                remaining = self.response_length - len(agent_data.response_mask)
                draft_budget = min(cfg["draft_cap"], max(0, remaining - _len(draft_close) - pred_tail - 1))
                _, draft, draft_clipped = await self._sample_controlled_phase(
                    agent_data, sampling_params, max_tokens=draft_budget,
                    stop_text=_markers.DRAFT_END_TAG, ppo_mask=1,
                )
                ok_close = self._append_controller_tokens(agent_data, draft_close)
                hist = self._v7_sim_view(agent_data)
                agent_data.messages.append({"role": "assistant", "content": draft})
                d_index = len(agent_data.messages) - 1
                ant = await self._v7_anticipate(agent_data, sampling_params)
                # `draft=` enables the format-only rule: a predicted reply that
                # only asks for a box the draft already has is an acceptance,
                # not a reason to rewrite the draft (ctom/controller.py).
                verdict, reason = _ctom.parse_verdict(ant["text"], task_type, draft=draft)
                rec["drafts"].append(draft)
                rec["predictions"].append(ant["text"])
                rec["verdicts"].append(verdict)
                rec["verdict_reasons"].append(reason)
                rec["slots"].append(ant["slot"])
                agent_data.v7_slots.append({
                    "kind": "draft", "round": round_index, "history": hist,
                    "text": draft, "verdict": verdict, "slot": ant["slot"],
                })
                draft_ok = bool(draft.strip()) and not draft_clipped and ok_open and ok_close
                validity["draft_valid"] = validity.get("draft_valid", True) and draft_ok
                validity["prediction_valid"] = validity.get("prediction_valid", True) and ant["valid"]
                if draft_clipped:
                    self._bump(agent_data, "v7_draft_clipped")
                self._bump(agent_data, f"v7_verdict_{verdict.lower()}")
                if reason == "idiom":
                    self._bump(agent_data, "v7_verdict_idiom")
                if last_draft is not None and draft.strip() != last_draft.strip():
                    self._bump(agent_data, "v7_draft_changed")
                last_draft = draft
                remaining = self.response_length - len(agent_data.response_mask)
                need_more = len(gen) + draft_block_min
                if verdict == _ctom.UNSATISFIED:
                    if j >= cfg["max_inner"]:
                        self._bump(agent_data, "v7_inner_cap_hit")
                    elif remaining < need_more:
                        # Benign: the round COMPLETES and this draft is shown; only
                        # the extra revision is dropped, exactly as at the cap. It
                        # is NOT the truncation the round-start guard above counts,
                        # and sharing that counter made a 6-round writing episode
                        # that merely ran out of revision room look like a budget
                        # misconfiguration to the smoke gate.
                        self._bump(agent_data, "v7_inner_stopped_no_budget")
                    else:
                        # This draft stays internal; open a NEW assistant turn.
                        agent_data.v7_hidden.add(d_index)
                        self._v7_append_ids(agent_data, gen)
                        j += 1
                        self._bump(agent_data, "v7_inner_iters")
                        continue
                break
            visible = _ctom.visible_message(_ctom.DRAFT, draft=draft, task_type=task_type)
            agent_data.v7_visible[d_index] = visible
            agent_data.v7_last_shown = draft
            rec["inner_iters"] = j

        # ── finalize the round ──
        rec["visible"] = visible
        rec["visible_tokens"] = _len(visible)
        rec["generated_tokens"] = agent_data.v7_generated_tokens
        self._bump(agent_data, "v7_visible_tokens", rec["visible_tokens"])
        self._bump(agent_data, "v7_generated_tokens", rec["generated_tokens"])
        agent_data.assistant_turns += 1
        boundary = None
        for k in range(len(agent_data.response_mask) - 1, -1, -1):
            if agent_data.response_mask[k]:
                boundary = k
                break
        agent_data.turn_end_indices.append(boundary)
        validity["invalid_count"] = sum(
            1 for k, v in validity.items() if k.endswith("_valid") and v is False
        )
        agent_data.phase_validity.append(validity)
        if not hasattr(agent_data, "draft_texts"):
            agent_data.draft_texts = []
        agent_data.draft_texts.append(rec["drafts"][-1] if rec["drafts"] else shown_before)
        rec["boundary_prompt_len"] = len(agent_data.prompt_ids)
        rec["invalid_count"] = validity["invalid_count"]
        agent_data.v7_rounds.append(rec)
        self._bump(agent_data, "v7_rounds")
        self._bump(agent_data, f"v7_decision_{decision.lower()}")
        if validity["invalid_count"]:
            self._bump(agent_data, "v7_invalid_phases", validity["invalid_count"])

        if decision == _ctom.COMMIT:
            return AgentState.TERMINATED
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED
        return AgentState.INTERACTING

    async def _collect_v7_counterfactual_targets(self, agent_data: AgentData) -> None:
        """One simulator label per prediction slot: what the user WOULD say to
        that draft (or to the bare question). Terminate-only replies become the
        canonical satisfied utterance (design §2.1). Also fills the verdict-vs-
        simulator 2x2 (v7_gap_*)."""
        slots = list(getattr(agent_data, "v7_slots", []))
        n_ctx = len(getattr(agent_data, "prediction_context_ids", []))
        targets: list[Optional[str]] = [None] * n_ctx
        agent_data.counterfactual_targets = targets
        interaction = agent_data.interaction
        if interaction is None or not slots:
            return
        task_type = str(agent_data.interaction_kwargs.get("task_type") or "math")

        async def _one(slot):
            history = _ctom.cf_history(slot["history"], slot["text"])
            try:
                _, response, _, _ = await interaction.generate_response(
                    agent_data.request_id, history, counterfactual=True
                )
            except Exception as exc:  # noqa: BLE001 - a label must never kill a rollout
                logger.warning(f"[CToMAgentLoop] v7 counterfactual sim call failed: {exc}")
                return slot, None, False
            cleaned = extract_sim_response(response or "")
            text, substituted = _ctom.substitute_terminate_target(cleaned, task_type)
            if not text.strip() and slot["kind"] == "draft":
                # Under the answer gate the simulator is told never to terminate,
                # so on a CORRECT draft it has nothing to say and returns "" (117
                # of the first 302 sim calls on smoke 967233). That is the
                # satisfied case, the same one the terminate-only rule covers.
                text, substituted = _ctom.TERMINATE_SUBSTITUTE.get(
                    task_type, _ctom.TERMINATE_SUBSTITUTE["math"]), True
            return slot, (text if text.strip() else None), substituted

        # `pred_sft_skip_terminate` stops being inert here (2026-09-06). A
        # terminate-only / empty counterfactual reply is the answer gate's way of
        # saying "this draft is correct", and substituting the canonical
        # utterance made that constant ~20% of the auxiliary supervision. The
        # policy duly collapsed onto it: at step 180, 96.1% of run 969590's
        # SATISFIED predictions were the exact string "That makes sense,
        # thanks." and the inner loop exited on it 219 times out of 228, while
        # the aux-off control never emitted it once. Since the same text is also
        # the loop's exit test, the auxiliary objective was teaching the policy
        # the password to its own gate. Those slots now carry NO target.
        #
        # The substituted text is still computed and still labels the 2x2 and
        # the cf_ok counter -- dropping it from the *loss* must not blind the
        # telemetry, and a dropped slot is not a failed simulator call.
        skip_terminate = bool(OmegaConf.select(
            self.config, "actor_rollout_ref.actor.pred_sft_skip_terminate", default=True
        ))
        got = subs = dropped = 0
        for slot, text, substituted in await asyncio.gather(*(_one(s) for s in slots)):
            k = int(slot["slot"])
            if 0 <= k < n_ctx:
                targets[k] = None if (substituted and skip_terminate) else text
            if text is None:
                continue
            got += 1
            subs += int(substituted)
            dropped += int(substituted and skip_terminate)
            if slot["kind"] == "draft":
                cf_push = _ctom.is_pushback(text, task_type)
                model_unsat = slot["verdict"] == _ctom.UNSATISFIED
                cell = "tp" if (model_unsat and cf_push) else "fp" if model_unsat else "fn" if cf_push else "tn"
                self._bump(agent_data, f"v7_gap_{cell}")
            elif slot["kind"] == "ask" and _markers.user_confirmed(text):
                self._bump(agent_data, "v7_ask_confirmed")
        self._bump(agent_data, "v7_cf_ok", got)
        self._bump(agent_data, "v7_cf_missing", len(slots) - got)
        if subs:
            self._bump(agent_data, "v7_cf_terminate_substituted", subs)
        if dropped:
            # Supervised slots = cf_ok - this. Not "missing": the call succeeded.
            self._bump(agent_data, "v7_cf_dropped_satisfied", dropped)


    def _mask_prediction_tokens_out_of_ppo(self, agent_data: AgentData) -> None:
        """Zero the PPO response_mask over the most-recently-generated assistant
        turn's prediction-value tokens, so PPO does not optimize the prediction
        (it is learned only by the auxiliary SFT loss). Fail-open: if the span
        can't be localized, the tokens are left in PPO unchanged.

        Must be called right after an assistant generation, before the next user
        turn is appended, so the generated block is the tail of response_mask.
        """
        if self._is_controller_phased:
            return
        if not self._pred_sft_enabled:
            return
        # Record this turn's prediction-value span (response-token space) for
        # the sft_v2 on-policy-context builder. Exactly ONE entry is appended
        # per call — [-1, -1] when localization fails — so the list stays in
        # lockstep with the predictions list (same call sites).
        if not hasattr(agent_data, "pred_value_spans"):
            agent_data.pred_value_spans = []
        resp_ids = list(agent_data.response_ids) if agent_data.response_ids else []
        block_len = len(resp_ids)
        if block_len == 0 or len(agent_data.response_mask) < block_len:
            agent_data.pred_value_spans.append([-1, -1])
            return
        span = locate_prediction_value_token_span(resp_ids, self.tokenizer)
        if span is None:
            agent_data.pred_value_spans.append([-1, -1])
            return
        s, e = span
        base = len(agent_data.response_mask) - block_len
        agent_data.pred_value_spans.append([base + s, base + e])
        n_masked = 0
        for k in range(base + s, base + e):
            if agent_data.response_mask[k]:
                agent_data.response_mask[k] = 0
                n_masked += 1
        logger.debug(
            f"[CToMAgentLoop] masked {n_masked} prediction tokens out of PPO "
            f"(turn block_len={block_len}, span={span})"
        )

    def _record_turn_boundary(self, agent_data: AgentData) -> None:
        """Record the response-space index of the LAST PPO token of the assistant
        turn that was just generated, for turn-wise reward placement.

        Must be called right after an assistant generation (and after its
        prediction tokens are masked out of PPO, in sft mode), before the next
        user turn is appended — so the freshly generated block is the tail of
        response_mask and the last response_mask==1 entry is this turn's last
        trainable token. The per-turn reward will be placed on that token so GAE
        credits it back through the turn. Kept in lockstep with the predictions
        list (one boundary per scored assistant turn).
        """
        if self._is_controller_phased:
            return
        if not hasattr(agent_data, "turn_end_indices"):
            agent_data.turn_end_indices = []
        mask = agent_data.response_mask
        idx = None
        for k in range(len(mask) - 1, -1, -1):
            if mask[k]:
                idx = k
                break
        if idx is None:
            # No trainable token in this turn (fully masked / empty). Skip so the
            # boundary list stays aligned with turns that actually have tokens.
            return
        agent_data.turn_end_indices.append(idx)

    async def _get_final_user_response(self, agent_data: AgentData, request_id: str) -> str:
        """
        Generate one final user simulator response after the assistant terminates.

        This is needed to:
        1. Get termination signals ([[TERMINATE CHAT]], <understood>) for R_task
        2. Get the actual user response to compare against the assistant's
           last prediction for R_pred

        Only triggers when the last message is from the assistant (i.e., the
        assistant terminated, not the user).  Does NOT modify token-level data
        (prompt_ids, response_ids, response_mask) — only appends to
        agent_data.messages for reward computation.
        """
        interaction = agent_data.interaction
        if interaction is None:
            return ""

        last_visible = (
            self._v7_sim_view(agent_data)[-1] if (self._is_sft_v7 and agent_data.messages) else
            (agent_data.messages[-1] if agent_data.messages else {})
        )
        if not agent_data.messages or last_visible.get("role") != "assistant":
            return ""

        try:
            if self._is_sft_v7:
                sim_messages = self._v7_sim_view(agent_data)
            else:
                sim_messages = agent_data.messages
            should_terminate, response, reward, extra_info = await interaction.generate_response(
                request_id, sim_messages
            )
            if response and response.strip():
                cleaned = extract_sim_response(response)
                agent_data.messages.append({"role": "user", "content": cleaned})
                logger.info(
                    f"[CToMAgentLoop] Generated final user response "
                    f"(len={len(cleaned)}, terminate={should_terminate}) "
                    f"for request_id={request_id}"
                )
                return cleaned
            return ""
        except Exception as e:
            logger.error(f"[CToMAgentLoop] Failed to generate final user response: {e}")
            return ""

    async def _generate_q0_with_retry(
        self,
        agent_data: AgentData,
        interaction,
        request_id: str,
        interaction_kwargs: dict,
    ) -> AgentState:
        """
        Generate the initial user query q0 with retry on connection failure.

        When the user simulator's vLLM server is temporarily overloaded or
        unreachable, this retries the q0 generation up to Q0_MAX_RETRIES
        times with exponential backoff.  This is safe because at q0 the
        conversation is empty — there is no state to corrupt by retrying.

        Mid-conversation retries are NOT safe (the dialogue would be in an
        inconsistent state), so this is intentionally q0-only.

        Returns the AgentState after the (possibly retried) q0 generation.
        """
        for attempt in range(Q0_MAX_RETRIES + 1):  # 0..Q0_MAX_RETRIES inclusive
            status = await self._handle_interacting_state(agent_data)
            _clean_last_user_message(agent_data)

            # ── Success: q0 was generated ──
            if status != AgentState.TERMINATED:
                # Verify the user message is actually non-empty
                last_msg = agent_data.messages[-1] if agent_data.messages else {}
                if (
                    last_msg.get("role") == "user"
                    and last_msg.get("content", "").strip()
                ):
                    if attempt > 0:
                        logger.info(
                            f"q0 generation succeeded on retry {attempt} "
                            f"(request_id={request_id})"
                        )
                    return status

            # ── Failure: check if it was a connection issue ──
            conn_failed = (
                interaction is not None
                and hasattr(interaction, "is_connection_failed")
                and interaction.is_connection_failed(request_id)
            )

            if not conn_failed:
                # Not a connection error (e.g. natural termination or empty
                # response from the model). Don't retry — return as-is.
                return status

            while (
                    agent_data.messages
                    and agent_data.messages[-1].get("role") == "user"
                ):
                    agent_data.messages.pop()
                    
            delay = Q0_RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"q0 connection failed (attempt {attempt + 1}/{Q0_MAX_RETRIES + 1}), "
                f"retrying in {delay:.1f}s... (request_id={request_id})"
            )
            await asyncio.sleep(delay)


    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        if self._is_sft_v7:
            messages = _ctom.action_prompt_messages(messages, self._v7_cfg()["action_protocol"])
        image_data = deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        video_data = deepcopy(kwargs.get("multi_modal_data", {}).get("video", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found. "
                    f"Available: {list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            video_data=video_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        # Per-turn reward placement: last-token index (response space) of each
        # scored assistant turn. Stays in lockstep with the predictions list.
        agent_data.turn_end_indices = []
        agent_data.prediction_context_ids = []
        agent_data.phase_validity = []
        agent_data.draft_texts = []
        agent_data.assistant_msg_indices = []
        agent_data.counterfactual_targets = []
        agent_data.internal_msg_indices = []
        # sft_v7 bookkeeping — see _run_v7_round.
        agent_data.pred_value_spans = []
        agent_data.v7_rounds = []
        agent_data.v7_slots = []
        agent_data.v7_hidden = set()
        agent_data.v7_visible = {}
        agent_data.v7_last_shown = ""

        # ── Algorithm 1, Phase 1: Multi-Turn Rollout ──
        #
        # Step 1: q0 <- Mu(rho, p)  — User simulator generates the initial question.
        #   The prompt only contains [system], so the last message role is "system".
        #   _handle_interacting_state calls the user simulator which sees empty chat
        #   history and generates q0 based on persona and problem.
        #
        # Step 2: (a0, e0) <- Ma(rho, q0) — Assistant answers and predicts.
        #   After q0 is appended, the assistant generates its first response.

        # Step 1: Generate q0 from user simulator (with retry on connection failure)
        interacting_status = await self._generate_q0_with_retry(
            agent_data, interaction, request_id, interaction_kwargs
        )

        if interacting_status == AgentState.TERMINATED:
            # Distinguish infrastructure failure from natural termination.
            conn_failed = (
                interaction is not None
                and hasattr(interaction, "is_connection_failed")
                and interaction.is_connection_failed(request_id)
            )
            if conn_failed:
                logger.error("User sim connection failed at q0 after all retries — returning skippable output.")
            else:
                logger.error("User sim terminated at q0 — returning empty output.")
            return _make_empty_output(agent_data, self.tokenizer, connection_failed=conn_failed)

        # ── Guard: check q0 was actually appended and is non-empty ──
        last_msg = agent_data.messages[-1] if agent_data.messages else {}
        if last_msg.get("role") != "user" or not last_msg.get("content", "").strip():
            logger.error("User simulator returned empty q0. Returning empty output.")
            return _make_empty_output(agent_data, self.tokenizer)

        # Step 2: Generate (a0, e0) from assistant
        await self._handle_pending_state(agent_data, sampling_params)

        # Snapshot state before generation for diagnostics
        pre_gen_mask_len = len(agent_data.response_mask)
        pre_gen_n_messages = len(agent_data.messages)

        if self._is_controller_phased:
            status = await self._generate_phased_turn(agent_data, sampling_params)
        else:
            status = await self._handle_generating_state(agent_data, sampling_params)

        post_gen_mask_len = len(agent_data.response_mask)
        assistant_tokens_generated = post_gen_mask_len - pre_gen_mask_len
        has_assistant_msg = (
            len(agent_data.messages) > pre_gen_n_messages
            and agent_data.messages[-1].get("role") == "assistant"
        )

        # ── Workaround for veRL base class bug ──
        # _handle_generating_state returns TERMINATED *before* appending the
        # assistant message to agent_data.messages when response_mask hits the
        # response_length limit.  The tokens ARE in prompt_ids/response_mask,
        # but the decoded text is missing from messages.  Decode it ourselves.
        # sft_v7 rounds legitimately end on a policy-written USER turn, so the
        # missing-assistant heuristic below does not apply to them.
        if not has_assistant_msg and assistant_tokens_generated > 0 and not self._is_sft_v7:
            decoded = self.tokenizer.decode(
                agent_data.response_ids, skip_special_tokens=True
            )
            agent_data.messages.append({"role": "assistant", "content": decoded})
            has_assistant_msg = True
            print(
                f"[CToMAgentLoop] Patched missing assistant message "
                f"(base class early-return on response_length). "
                f"decoded_len={len(decoded)}",
                flush=True,
            )

        assistant_content = agent_data.messages[-1].get("content", "") if has_assistant_msg else ""

        # if status == AgentState.TERMINATED:
            # print(
            #     f"\n[CToMAgentLoop] Assistant TERMINATED at first turn:"
            #     f"\n  status={status}"
            #     f"\n  response_mask_len: {pre_gen_mask_len} -> {post_gen_mask_len} "
            #     f"(response_length limit={self.response_length})"
            #     f"\n  assistant_tokens_generated={assistant_tokens_generated}"
            #     f"\n  has_assistant_msg={has_assistant_msg}"
            #     f"\n  assistant_content_len={len(assistant_content)}"
            #     f"\n  assistant_content_preview={assistant_content[:200]!r}"
            #     f"\n  n_messages={len(agent_data.messages)}"
            #     f"\n  roles={[m.get('role') for m in agent_data.messages]}",
            #     flush=True,
            # )

        # ── Retry if assistant produced empty/degenerate output ──
        # This can happen when:
        #   - vLLM returns 0 tokens (server hiccup)
        #   - Model immediately emits EOS
        #   - Response budget exhausted by long user message (prompt_ids + q0
        #     tokens already fill response_mask close to response_length)
        # Retry is safe here: the conversation state is just [system, q0],
        # and we can re-run _handle_pending_state + _handle_generating_state.
        if (
            not self._is_controller_phased
            and status == AgentState.TERMINATED
            and assistant_tokens_generated <= 1
        ):
            max_retries = 2
            for retry in range(max_retries):
                print(
                    f"[CToMAgentLoop] Retrying assistant generation "
                    f"(attempt {retry + 2}/{max_retries + 1}), "
                    f"assistant_tokens={assistant_tokens_generated}",
                    flush=True,
                )
                # Roll back: remove the empty/minimal assistant message and
                # reset response state to pre-generation snapshot
                while (
                    agent_data.messages
                    and agent_data.messages[-1].get("role") == "assistant"
                ):
                    agent_data.messages.pop()
                agent_data.prompt_ids = agent_data.prompt_ids[
                    : len(agent_data.prompt_ids) - assistant_tokens_generated
                ]
                agent_data.response_mask = agent_data.response_mask[
                    : pre_gen_mask_len
                ]
                if agent_data.response_logprobs:
                    agent_data.response_logprobs = agent_data.response_logprobs[
                        : pre_gen_mask_len
                    ]
                agent_data.assistant_turns -= 1

                await self._handle_pending_state(agent_data, sampling_params)
                status = await self._handle_generating_state(agent_data, sampling_params)

                new_mask_len = len(agent_data.response_mask)
                assistant_tokens_generated = new_mask_len - pre_gen_mask_len
                has_assistant_msg = (
                    len(agent_data.messages) > pre_gen_n_messages
                    and agent_data.messages[-1].get("role") == "assistant"
                )

                # Same base class early-return workaround for retries
                if not has_assistant_msg and assistant_tokens_generated > 0:
                    decoded = self.tokenizer.decode(
                        agent_data.response_ids, skip_special_tokens=True
                    )
                    agent_data.messages.append({"role": "assistant", "content": decoded})
                    has_assistant_msg = True

                assistant_content = (
                    agent_data.messages[-1].get("content", "")
                    if has_assistant_msg else ""
                )

                if assistant_tokens_generated > 1:
                    print(
                        f"[CToMAgentLoop] Retry {retry + 2} succeeded: "
                        f"assistant_tokens={assistant_tokens_generated}",
                        flush=True,
                    )
                    break
            else:
                print(
                    f"[CToMAgentLoop] All {max_retries + 1} assistant generation "
                    f"attempts produced ≤1 token. Proceeding with what we have.",
                    flush=True,
                )

        # Disentangle the a0 prediction tokens from PPO (sft mode). Done here,
        # after retries settle and before any user turn is appended, so the
        # generated block is still the tail of response_mask.
        self._mask_prediction_tokens_out_of_ppo(agent_data)
        # Record a0's last trainable token for turn-wise reward placement (turn 0).
        self._record_turn_boundary(agent_data)

        initial_answer, _ = parse_answer_and_prediction(assistant_content)
        submit_final = _is_submit_final(agent_data, initial_answer)
        if status == AgentState.TERMINATED or submit_final:
            num_repeats = 0
        else:
            num_repeats = self.config.actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts

        # Storage for predictions and actual user responses across rollouts
        if num_repeats == 0:
            interaction_requests = []
        else:
            interaction_requests = [agent_data]

        # ── Capture the initial conversation (q0, a0) even when TERMINATED ──
        # When the assistant's first response terminates the conversation
        # (e.g. solved in one turn, or hit EOS), num_repeats=0 so the loop
        # below is skipped entirely.  But the messages and prediction from
        # the (q0, a0, e0) exchange already exist in agent_data — we must
        # capture them here, otherwise messages_lst stays [] and the reward
        # manager sees an empty conversation.
        messages_lst = []
        predictions_lst = []  # predictions_lst[rollout_idx] = [e0, e1, ...]

        if num_repeats == 0 and agent_data.messages:
            # Extract prediction from a0 before storing
            initial_predictions = []
            if (
                agent_data.messages[-1].get("role") == "assistant"
                and agent_data.messages[-1].get("content", "").strip()
            ):
                answer, prediction = parse_answer_and_prediction(
                    agent_data.messages[-1]["content"]
                )
                initial_predictions.append(prediction)
                agent_data.messages[-1]["content"] = answer

            # Generate one final user response so the reward function can:
            # 1. Detect [[TERMINATE CHAT]] / <understood> signals for R_task
            # 2. Compare prediction e0 against actual user response for R_pred
            # A submit-task episode (coding, writing) ends at
            # submission. There is no next human turn, so the final prediction is
            # intentionally left unscored.
            if not _is_submit_task(agent_data):
                await self._get_final_user_response(agent_data, request_id)

            # This branch is the ONLY exit that skips _run_interaction_loop, so
            # without this call a one-turn episode contributes zero counterfactual
            # targets and therefore zero auxiliary supervision. It is reachable on
            # whenever the first revision already carries
            # "I WANT TO ANSWER:" (submit_final above), which is precisely the
            # behaviour a task-reward-maximizing policy drifts toward -- so the
            # objective would quietly starve exactly as the run started working.
            await self._collect_v7_counterfactual_targets(agent_data)

            messages_lst.append([Message(**msg) for msg in self._reward_messages(agent_data)])
            predictions_lst.append(initial_predictions)

        for _agent_data in interaction_requests:
            if not is_valid_messages(_agent_data.messages[-1]):
                break

            rollout_predictions = []

            # Parse prediction from the initial assistant response (a0, e0)
            if _agent_data.messages[-1]["role"] == "assistant":
                answer, prediction = parse_answer_and_prediction(
                    _agent_data.messages[-1]["content"]
                )
                rollout_predictions.append(prediction)
                # Strip prediction from the message shown to user simulator
                _agent_data.messages[-1]["content"] = answer

            # Run remaining turns: q1 -> (a1, e1) -> q2 -> (a2, e2) -> ...
            await self._run_interaction_loop(_agent_data, sampling_params, rollout_predictions)

            # Label each prediction slot with what the user WOULD have
            # said to that turn's draft. Done here, after the rollout, because
            # nothing in generation consumes it — it is a training label, so the
            # calls can all be issued concurrently instead of adding latency to
            # every turn.
            await self._collect_v7_counterfactual_targets(_agent_data)

            messages_lst.append([Message(**msg) for msg in self._reward_messages(_agent_data)])
            predictions_lst.append(rollout_predictions)

        # ── Defensive: ensure prompt_ids / response_mask are tensors ──
        agent_data.prompt_ids = _ensure_tensor(agent_data.prompt_ids)
        agent_data.response_mask = _ensure_tensor(agent_data.response_mask)

        # Handle the edge case where response_mask is empty (no generation happened)
        if len(agent_data.response_mask) == 0:
            logger.error("Empty response_mask — assistant produced no tokens. Returning empty output.")
            return _make_empty_output(agent_data, self.tokenizer)

        # ── Defensive: ensure prompt_ids / response_mask are tensors ──
        # When the assistant produces empty or degenerate output, these may
        # remain as plain Python lists, causing downstream crashes in
        # _agent_loop_postprocess: AttributeError: 'list' object has no attribute 'dim'
        if isinstance(agent_data.prompt_ids, list):
            agent_data.prompt_ids = torch.tensor(agent_data.prompt_ids, dtype=torch.long)
        if isinstance(agent_data.response_mask, list):
            agent_data.response_mask = torch.tensor(agent_data.response_mask, dtype=torch.long)

        # Handle the edge case where response_mask is empty (no generation happened)
        if len(agent_data.response_mask) == 0:
            logger.error("Empty response_mask after tensor conversion — returning empty output.")
            return _make_empty_output(agent_data, self.tokenizer)

        # Persist the v7 phase trace before these rounds are dropped downstream.
        if self._is_sft_v7:
            _v7_trace_write(agent_data)

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs is not None
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={
                "turn_scores": agent_data.turn_scores,
                "messages": {"messages": messages_lst},
                "predictions": predictions_lst,  # store predictions for reward computation
                "turn_end_indices": list(getattr(agent_data, "turn_end_indices", [])),  # per-turn reward placement
                "pred_value_spans": list(getattr(agent_data, "pred_value_spans", [])),  # sft_v2 on-policy context
                "prediction_context_ids": list(getattr(agent_data, "prediction_context_ids", [])),
                "phase_validity": list(getattr(agent_data, "phase_validity", [])),
                # sft_v4 only: the internal draft each revision was written from.
                # Carried for offline analysis (how much does the revision change?)
                # and deliberately kept OUT of messages so the sim never sees it.
                "draft_texts": list(getattr(agent_data, "draft_texts", [])),
                # sft_v5 only: per-turn counterfactual reaction to the draft,
                # the teacher-forced target for the prediction phase.
                "counterfactual_targets": list(getattr(agent_data, "counterfactual_targets", [])),
                # sft_v6 only: one record per visible turn (draft/belief/decision/
                # ask/visible text), the message indices the simulator never saw,
                # and the counterfactual self-revision drafts the VOI gate scores.
                "bdr_rounds": list(getattr(agent_data, "bdr_rounds", [])),
                "internal_msg_indices": list(getattr(agent_data, "internal_msg_indices", [])),
                "cf_drafts": list(getattr(agent_data, "cf_drafts", [])),
                "cf_traces": list(getattr(agent_data, "cf_traces", [])),
                # sft_v7 only: one record per round (think/decision/drafts/predictions/verdicts).
                "v7_rounds": list(getattr(agent_data, "v7_rounds", [])),
                "empty_generation": False,  # must be present in ALL outputs for DataProto.concat
                "connection_failed": False,  # must be present in ALL outputs for DataProto.concat
            },
        )
        return output

    async def _run_interaction_loop(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        rollout_predictions: list[str],
    ):
        """
        Run the multi-turn interaction loop.

        At each turn:
        1. User simulator generates qt (via INTERACTING state)
        2. Assistant generates (at, et) (via GENERATING state)
        3. Parse and store et, strip prediction from message before next user turn.

        If the user simulator returns an empty response (connection failure),
        the loop terminates gracefully with whatever turns were completed.
        """
        state = AgentState.INTERACTING

        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                pre_msgs = len(agent_data.messages)
                if self._is_controller_phased:
                    state = await self._generate_phased_turn(agent_data, sampling_params)
                else:
                    state = await self._handle_generating_state(agent_data, sampling_params)

                # Workaround: base class early-returns on response_length before
                # appending the assistant message. Decode it ourselves if missing.
                if (
                    len(agent_data.messages) == pre_msgs
                    and hasattr(agent_data, "response_ids")
                    and agent_data.response_ids
                ):
                    decoded = self.tokenizer.decode(
                        agent_data.response_ids, skip_special_tokens=True
                    )
                    agent_data.messages.append({"role": "assistant", "content": decoded})

                # After generation, parse the assistant's output to extract prediction
                if agent_data.messages[-1].get("role") == "assistant":
                    content = agent_data.messages[-1].get("content", "").strip()
                    if content:
                        answer, prediction = parse_answer_and_prediction(content)
                        rollout_predictions.append(prediction)
                        # Disentangle this turn's prediction tokens from PPO
                        # (sft mode) before the message text is stripped and
                        # before the next user turn is appended.
                        self._mask_prediction_tokens_out_of_ppo(agent_data)
                        # Record this turn's last trainable token (turn-wise
                        # reward), in lockstep with the prediction just appended.
                        self._record_turn_boundary(agent_data)
                        # Strip prediction from the message shown to user simulator
                        agent_data.messages[-1]["content"] = answer
                        if _is_submit_final(agent_data, answer):
                            state = AgentState.TERMINATED

            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)

                # FIX: clean raw simulator JSON from the user message
                # The base class may append the raw simulator output (structured
                # JSON with current_status/thought/response) instead of the clean
                # response string. Extract just the 'response' field.
                _clean_last_user_message(agent_data)

                # ── Guard: if user simulator returned empty content, stop the loop ──
                # This happens when the connection to the vLLM server fails and all
                # retries are exhausted. We keep whatever turns we have so far rather
                # than crashing the entire rollout.
                if state != AgentState.TERMINATED and agent_data.messages:
                    last_msg = agent_data.messages[-1]
                    if last_msg.get("role") == "user" and not last_msg.get("content", "").strip():
                        logger.error(
                            "User simulator returned empty response mid-dialogue. "
                            "Terminating loop with partial conversation."
                        )
                        # Remove the empty user message so conversation stays clean
                        agent_data.messages.pop()
                        state = AgentState.TERMINATED
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # After the loop ends, if the last message is from the assistant
        # (i.e., the assistant terminated, not the user), generate one final
        # user response for reward signal detection and prediction evaluation.
        if not _is_submit_task(agent_data):
            await self._get_final_user_response(agent_data, agent_data.request_id)
