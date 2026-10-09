# Copyright 2025 The CToM Authors
# Personalized user simulator for multi-turn dialogue training.
# Fully self-contained — no collabllm dependencies.

import asyncio
import copy
import datetime
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from ctom.utils import SPEC_TASK_TYPES, remove_think_block, truncate_messages
from verl.interactions.base import BaseInteraction
from verl.utils.rollout_trace import rollout_trace_op


def _answer_is_correct(text: str, ground_truth: str) -> bool:
    """Strict-checker verdict for the answer-gated termination path.

    Imported lazily and fail-closed: math_verify pulls in sympy, and a checker
    error must never end an episode early (that would hand out a spurious
    terminal state), so any failure reports "not correct" and the dialogue
    continues exactly as it would have without the gate.
    """
    if not text or not ground_truth:
        return False
    try:
        from ctom.task_math import check_math_answer

        return bool(check_math_answer(text, ground_truth))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[CToMUserSimulator] answer gate check failed ({exc}); not terminating")
        return False

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ── JSON parse failure log ──
_JSON_FAIL_LOG = os.getenv("CTOM_JSON_FAIL_LOG", "json_parse_failures.jsonl")

# ── Global atomic counter for deterministic backend assignment ──
_BACKEND_COUNTER_LOCK = threading.Lock()
_BACKEND_COUNTER = [0]


def _next_backend_index(num_backends: int) -> int:
    """Deterministic round-robin assignment across all CToMUserSimulator instances."""
    with _BACKEND_COUNTER_LOCK:
        idx = _BACKEND_COUNTER[0] % num_backends
        _BACKEND_COUNTER[0] += 1
        return idx


def _log_json_parse_failure(
    instance_id: str,
    raw_response: str,
    error_msg: str,
    expertise: str = "",
    goal: str = "",
    turn_count: int = -1,
):
    """Append a structured record to the JSONL failure log for post-hoc debugging."""
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "instance_id": instance_id,
        "expertise": expertise,
        "goal": goal,
        "turn_count": turn_count,
        "error": error_msg,
        "raw_response": raw_response,
    }
    try:
        with open(_JSON_FAIL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[CToMUserSimulator] Could not write to {_JSON_FAIL_LOG}: {e}")


# ── Conversation logger (replaces proxy logging when proxy is skipped) ──
_CONVERSATION_LOG = os.getenv("CTOM_CONVERSATION_LOG", "")


class _ConversationLogger:
    """Thread-safe JSONL writer for conversation logging, matching the proxy log format."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        logger.info(f"[CToMUserSimulator] Conversation logging enabled → {path}")

    def log(self, record: dict) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            logger.warning(f"[CToMUserSimulator] Could not write conversation log: {e}")


TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"
UNDERSTOOD_TOKEN = "<understood>"


NOVICE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
EXPERT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

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


def _load_prompt_template(path: str) -> str:
    """Load a prompt template from a text file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return p.read_text(encoding="utf-8")


def _load_prompt_template_optional(path: str) -> Optional[str]:
    """Load a prompt template if the file exists, else return None (no error)."""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.is_file() else None


def _strip_think_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks from a raw string.

    This is a string-safe version — unlike remove_think_block() from
    ctom.utils which expects a message dict, this operates directly
    on plain text.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _escape_control_chars_in_json_strings(json_str: str) -> str:
    """
    Escape literal control characters (newline, tab, carriage return, etc.)
    that appear inside JSON string values.

    LLMs frequently output JSON with raw newlines inside strings like:
        "response": "Here are the options:\n\n(A) ..."
    where the \\n are actual 0x0A bytes, not the two-character escape \\n.
    json.loads() rejects these as "Invalid control character".

    This function walks the JSON character-by-character, tracking whether
    we're inside a quoted string. Control chars inside strings are replaced
    with their JSON escape sequences. Control chars outside strings (which
    are just whitespace formatting) are left alone.
    """
    _CTRL_MAP = {
        '\n': '\\n',
        '\r': '\\r',
        '\t': '\\t',
        '\x08': '\\b',   # backspace
        '\x0c': '\\f',   # form feed
    }
    result: list[str] = []
    i = 0
    in_string = False
    n = len(json_str)

    while i < n:
        ch = json_str[i]

        if not in_string:
            if ch == '"':
                in_string = True
            result.append(ch)
            i += 1
        else:
            # Inside a JSON string
            if ch == '\\' and i + 1 < n:
                # Already-escaped sequence — pass through both chars
                result.append(ch)
                result.append(json_str[i + 1])
                i += 2
            elif ch == '"':
                in_string = False
                result.append(ch)
                i += 1
            elif ch in _CTRL_MAP:
                # Literal control char inside a string — escape it
                result.append(_CTRL_MAP[ch])
                i += 1
            else:
                result.append(ch)
                i += 1

    return ''.join(result)


def repair_json_for_latex(json_str: str) -> str:
    """
    Fix LaTeX backslashes inside JSON string values without corrupting
    the surrounding JSON structure.

    Walks the JSON text character-by-character, tracking whether the
    current position is inside a quoted string value.  When inside a
    string, any backslash that does NOT form a valid JSON escape
    sequence is doubled (so the backslash becomes a literal character
    in the parsed output).

    For the ambiguous single-char escapes that are BOTH valid JSON
    escapes AND common LaTeX command prefixes (\\b \\f \\n \\r \\t),
    a heuristic is used: if the character after the escape letter is
    also alphabetic (e.g. \\frac, \\not, \\text, \\rfloor, \\binom),
    it is treated as LaTeX and the backslash is doubled.  Otherwise
    it is kept as a JSON escape (e.g. a real \\n newline).
    """
    result: list[str] = []
    i = 0
    in_string = False
    n = len(json_str)

    while i < n:
        ch = json_str[i]

        if not in_string:
            # Outside a string — just copy and watch for opening quote
            if ch == '"':
                in_string = True
            result.append(ch)
            i += 1
        else:
            # Inside a JSON string value
            if ch == '"':
                # Unescaped quote → end of string
                in_string = False
                result.append(ch)
                i += 1

            elif ch == '\\':
                if i + 1 >= n:
                    # Trailing backslash at end of input → double it
                    result.append('\\\\')
                    i += 1
                    continue

                next_ch = json_str[i + 1]

                if next_ch in ('"', '\\', '/'):
                    # Unambiguously valid JSON escapes — pass through
                    result.append(ch)
                    result.append(next_ch)
                    i += 2

                elif next_ch == 'u':
                    # \uXXXX unicode escape — but only if followed by 4 hex digits.
                    # Otherwise it's LaTeX like \underline → double the backslash.
                    remaining = json_str[i+2:i+6]
                    if len(remaining) == 4 and all(c in '0123456789abcdefABCDEF' for c in remaining):
                        # Valid \uXXXX — pass through
                        result.append(ch)
                        result.append(next_ch)
                        i += 2
                    else:
                        # LaTeX like \underline → double the backslash
                        result.append('\\\\')
                        i += 1

                elif next_ch in ('n', 't', 'r', 'b', 'f'):
                    # Ambiguous: valid JSON escape OR LaTeX command prefix
                    # Heuristic: if followed by another letter → LaTeX
                    #   \frac  → LaTeX (f + r)
                    #   \n     → JSON newline (n + non-alpha or end)
                    #   \not   → LaTeX (n + o)
                    #   \text  → LaTeX (t + e)
                    #   \binom → LaTeX (b + i)
                    #   \rfloor→ LaTeX (r + f)
                    if i + 2 < n and json_str[i + 2].isalpha():
                        # Looks like a LaTeX command → double the backslash
                        result.append('\\\\')
                        i += 1  # advance past backslash only; next_ch stays
                    else:
                        # Standalone escape → keep as JSON escape
                        result.append(ch)
                        result.append(next_ch)
                        i += 2

                else:
                    # Not a valid JSON escape char (\s, \l, \c, \p, \d, …)
                    # Must be LaTeX → double the backslash
                    result.append('\\\\')
                    i += 1  # advance past backslash only
            else:
                result.append(ch)
                i += 1

    return ''.join(result)


def _parse_sim_json_response(raw_response: str, instance_id: str = "", expertise: str = "", goal: str = "", turn_count: int = -1) -> dict:
    """
    Parse structured JSON output from the user simulator.

    The sim returns JSON (possibly inside ```json ... ``` fences) with keys:
        - current_status / current_understanding
        - thought
        - response   ← the actual user message to forward

    Falls back to using the raw text as the response if parsing fails.
    Automatically repairs invalid LaTeX backslash escapes on first failure.
    """
    clean = _strip_think_blocks(raw_response)

    # Try ```json ... ``` fenced block first.
    # Use greedy (.*) to match up to the LAST ```, not the first — the response
    # may contain nested code fences (e.g. Asymptote/Python code) that would
    # prematurely terminate a non-greedy match.
    json_match = re.search(r"```json\s*(.*)\s*```", clean, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try bare JSON object
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            error_msg = "No JSON object found in sim response"
            logger.error(f"[CToMUserSimulator] {error_msg}, using raw text")
            _log_json_parse_failure(instance_id, raw_response, error_msg, expertise, goal, turn_count)
            return {"response": clean.strip(), "_parse_failed": True}

    # Attempt 1: parse as-is
    # Attempt 2: escape literal control characters (newlines etc.) inside strings
    # Attempt 3: escape control chars + repair LaTeX backslash escapes
    last_error = None
    ctrl_escaped = _escape_control_chars_in_json_strings(json_str)
    candidates = [
        json_str,                                  # raw
        ctrl_escaped,                              # control chars escaped
        repair_json_for_latex(ctrl_escaped),       # control chars + LaTeX repair
    ]
    for attempt, candidate in enumerate(candidates):
        try:
            parsed = json.loads(candidate)
            response = parsed.get("response", "")
            # A NESTED object ({"response": {...}}) puts a dict in a slot every
            # caller is typed to receive as str, and it propagates rather than
            # failing here: extract_sim_response() calls .strip() on it and kills
            # the rollout and with it the whole job (1002602, step 99), while the
            # ordinary turn path silently tests `termination_signal in response`
            # against its KEYS. Treat any non-str payload as the empty-response
            # case handled just below.
            bad_type = None if isinstance(response, str) else type(response).__name__
            if bad_type is not None:
                response = ""
            if not response:
                # Model genuinely produced empty response — don't substitute raw text.
                # Return empty string so the agent loop's empty-content guard can
                # handle it properly (terminate loop or retry).
                error_msg = (
                    f"Parsed JSON 'response' field is {bad_type}, not str"
                    if bad_type is not None
                    else "Parsed JSON has empty 'response' field"
                )
                logger.error(f"[CToMUserSimulator] {error_msg}")
                _log_json_parse_failure(instance_id, raw_response, error_msg, expertise, goal, turn_count)
                return {
                    "current_status": parsed.get("current_status", parsed.get("current_understanding", "")),
                    "thought": parsed.get("thought", ""),
                    "response": "",
                    "_parse_failed": False,  # JSON parsed fine, model just gave empty response
                    "_empty_response": True,
                }
            if attempt > 0:
                repair_names = ["raw", "control_char_escape", "control_char+latex_repair"]
                logger.info(
                    f"[CToMUserSimulator] JSON parsed on attempt {attempt+1} "
                    f"({repair_names[attempt]}) (instance={instance_id})"
                )
            return {
                "current_status": parsed.get("current_status", parsed.get("current_understanding", "")),
                "thought": parsed.get("thought", ""),
                "response": response,
                "_parse_failed": False,
            }
        except json.JSONDecodeError as e:
            last_error = e
            continue

    # Both attempts failed
    error_msg = f"JSONDecodeError (after repair attempt): {last_error}"
    logger.warning(f"[CToMUserSimulator] JSON decode failed: {last_error}, using raw text")
    _log_json_parse_failure(instance_id, raw_response, error_msg, expertise, goal, turn_count)
    return {"response": clean.strip(), "_parse_failed": True}


class CToMUserSimulator(BaseInteraction):
    """
    Personalized user simulator for multi-turn dialogue training.

    Supports three connection modes:
    - Together API (default): set TOGETHER_API_KEY env var
    - Single local vLLM server: set api_base to one URL (e.g. "http://host:8100/v1")
    - Multiple local vLLM servers (no proxy): set api_base to a comma-separated
      list of URLs (e.g. "http://host:8101/v1,http://host:8102/v1,...").
      Each worker instance is deterministically assigned to one backend via a
      global atomic counter, giving even round-robin distribution.

    Prompt templates are loaded from external files specified by `prompt_dir`:
        {prompt_dir}/sim_solving.txt
    """

    def __init__(self, config: dict):
        super().__init__(config)
        _config = copy.deepcopy(config)
        enable_log = _config.pop("enable_log", False)
        log_path = _config.pop("conversation_log_path", "") or _CONVERSATION_LOG
        self._conversation_logger = (
            _ConversationLogger(log_path) if enable_log and log_path else None
        )

        self.name = _config.pop("name")
        self.termination_signal = _config.pop("termination_signal", TERMINATION_SIGNAL)
        self.understood_token = _config.pop("understood_token", UNDERSTOOD_TOKEN)
        # CToM: let the strict answer checker end a solving episode instead of the
        # simulator's [[TERMINATE CHAT]] token. Off by default so every prior run
        # reproduces; the expert-only v5 launcher turns it on. When on, the
        # simulator is neither asked to emit the token nor checked for it.
        self.answer_gated_termination = bool(
            _config.pop("answer_gated_termination", False)
        )
        self.num_retries = _config.pop("num_retries", 5)

        # ── Prompt template directory ──
        prompt_dir = _config.pop("prompt_dir", None)
        if prompt_dir:
            self.prompt_template_solving = _load_prompt_template(
                os.path.join(prompt_dir, "sim_solving.txt")
            )
            # Optional novice-specific templates for a weaker novice simulator that
            # needs stronger formatting guidance. Fall back to the shared (expert)
            # templates when the novice variant file is absent.
            self.prompt_template_solving_novice = (
                _load_prompt_template_optional(os.path.join(prompt_dir, "sim_solving_novice.txt"))
                or self.prompt_template_solving
            )
            self.prompt_template_writing = _load_prompt_template_optional(
                os.path.join(prompt_dir, "sim_writing.txt")
            )
            self.prompt_template_bigcodebench = _load_prompt_template_optional(
                os.path.join(prompt_dir, "sim_code.txt")
            )
            logger.info(f"CToMUserSimulator loaded prompt templates from {prompt_dir}")
            # sft_v6 (BDR): the [INTERNAL] controller marker must never reach a
            # simulator prompt, or the simulator could learn to echo it and the
            # sim-visible filter would stop being the only thing that hides
            # internal turns. Cheap to assert once; catastrophic to discover late.
            for _name, _tmpl in (
                ("sim_solving", self.prompt_template_solving),
                ("sim_solving_novice", self.prompt_template_solving_novice),
                ("sim_writing", self.prompt_template_writing),
                ("sim_code", self.prompt_template_bigcodebench),
            ):
                if _tmpl and "[INTERNAL]" in _tmpl:
                    raise ValueError(f"{_name}.txt contains the BDR [INTERNAL] marker")
        else:
            raise ValueError(
                "prompt_dir is required in CToMUserSimulator config. "
                "Set it to the directory containing sim_solving.txt"
            )

        # ── Backend configuration ──
        api_base_raw = _config.pop("api_base", None)
        self.api_key = _config.pop("api_key", None) or os.environ.get("TOGETHER_API_KEY")

        if not api_base_raw and not self.api_key:
            raise ValueError(
                "Either api_base (for local server) or TOGETHER_API_KEY (for Together API) must be set."
            )

        # ── Multi-backend support ──
        # Accept comma-separated URLs: "http://h:8101/v1,http://h:8102/v1,..."
        # Each CToMUserSimulator instance gets deterministically assigned to one
        # backend via a global atomic counter for even distribution.
        if api_base_raw and "," in api_base_raw:
            self._all_backends = [u.strip() for u in api_base_raw.split(",") if u.strip()]
            self._backend_idx = _next_backend_index(len(self._all_backends))
            self.api_base = self._all_backends[self._backend_idx]
            logger.info(
                f"CToMUserSimulator worker assigned to backend {self._backend_idx}/{len(self._all_backends)}: "
                f"{self.api_base}"
            )
            print(
                f"[CToMUserSimulator] Worker assigned to backend {self._backend_idx}/{len(self._all_backends)}: "
                f"{self.api_base}"
            )
        else:
            self._all_backends = [api_base_raw] if api_base_raw else []
            self._backend_idx = 0
            self.api_base = api_base_raw
            if self.api_base:
                logger.info(f"CToMUserSimulator using single local server: {self.api_base}")
            else:
                logger.info("CToMUserSimulator using Together API")

        # Per-expertise model configuration
        self.user_model_novice = _config.pop("user_model_novice", NOVICE_MODEL)
        self.user_model_expert = _config.pop("user_model_expert", EXPERT_MODEL)
        self.user_model_default = _config.pop(
            "user_model_default", self.user_model_expert
        )
        self.user_model_writing = _config.pop(
            "user_model_writing", self.user_model_default
        )
        self.user_model_bigcodebench = _config.pop(
            "user_model_bigcodebench", self.user_model_default
        )
        _config.pop("user_model", None)

        # Context window truncation budget (0 = disabled)
        self.max_context_tokens = _config.pop("max_context_tokens", 0)

        # Replay q0 verbatim: deliver the problem statement itself as the first
        # user turn instead of asking the persona simulator to phrase it. Run
        # 816911 showed the simulators withhold the problem from their first
        # message (>=90% of the problem's numbers present in q0: expert 36%,
        # novice 13%), so the assistant answered blind and task reward was
        # unattributable. Personas still drive every later turn. Set
        # replay_q0: false in the interaction config for the old behavior.
        # (Must be popped: leftover keys become chat.completions.create kwargs.)
        self.replay_q0 = bool(_config.pop("replay_q0", True))

        # Remaining config items → chat.completions.create kwargs
        self.completion_kwargs = _config

        self._instance_dict = {}
        self._async_client = None

    def _get_async_client(self):
        """
        Lazily create the async client with explicit connection pool limits
        to prevent file descriptor / connection exhaustion under high concurrency.
        """
        if self._async_client is None:
            if self.api_base:
                import httpx
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(
                    base_url=self.api_base,
                    api_key=self.api_key or "not-needed",
                    http_client=httpx.AsyncClient(
                        limits=httpx.Limits(
                            max_connections=100,
                            max_keepalive_connections=20,
                            keepalive_expiry=30,
                        ),
                        timeout=httpx.Timeout(timeout=120.0),
                    ),
                )
            else:
                from together import AsyncTogether
                self._async_client = AsyncTogether(api_key=self.api_key)
        return self._async_client

    def _get_user_model(self, expertise: str) -> str:
        if expertise == "expert":
            return self.user_model_expert
        else:
            return self.user_model_novice

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        expertise = kwargs.get("expertise", "novice")
        goal = kwargs.get("goal", "solving")
        task_type = kwargs.get("task_type", "math")
        if task_type not in SPEC_TASK_TYPES:
            assert expertise in ("expert", "novice"), f"Invalid expertise: {expertise}"
            assert goal == "solving", f"Unsupported goal: {goal}"

        interaction_kwargs = dict(kwargs)
        interaction_kwargs["ground_truth"] = ground_truth

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
            "user_token_count": 0,
            "turn_count": 0,
            "connection_failed": False,
            # Interactions run concurrently in one worker. Keep every sample's
            # hidden specification isolated instead of storing kwargs globally.
            "interaction_kwargs": interaction_kwargs,
        }
        assert "single_turn_prompt" in kwargs, "single_turn_prompt is required"
        return instance_id

    @rollout_trace_op
    def _solving_ending_rule(self) -> str:
        """The prompt's rule about how a solving conversation ends.

        Gate OFF: the historical instruction — the simulator emits the stop token
        when it judges the problem solved. Byte-identical to every prior run.

        Gate ON: the checker owns the ending, so the simulator is told not to end
        anything. The one case it must handle is the reason the gate can miss a
        solved problem: the answer is right but never wrapped in \\boxed{}, which
        the strict checker cannot see. There the expert should ask for the boxed
        form rather than move on.
        """
        if not self.answer_gated_termination:
            return (
                f'- If the AI has fully solved the problem correctly, set response to '
                f'"{self.termination_signal}".'
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

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], counterfactual: bool = False, **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Generate the simulated user's next turn.

        counterfactual=True asks "what WOULD this user say to this assistant
        message?" without advancing the dialogue: no turn_count increment, no
        stored response, no connection_failed flag. sft_v5 uses it to label the
        prediction phase with the user's reaction to the DRAFT — a turn that
        never happened, since the user only ever saw the revision. Every mutation
        of instance state below is therefore guarded on this flag; the prompt
        construction, retry and parse logic are shared verbatim so the
        counterfactual label is drawn from exactly the same distribution as a
        real turn.
        """
        assert messages[-1]["role"] in ["system", "assistant"], (
            "Last message input to the user model must be from system or assistant role"
        )

        instance = self._instance_dict.get(instance_id)
        if instance is None:
            raise KeyError(f"Unknown interaction instance: {instance_id}")
        interaction_kwargs = instance["interaction_kwargs"]
        expertise = interaction_kwargs.get("expertise", "novice")
        goal = interaction_kwargs.get("goal", "solving")
        task_type = interaction_kwargs.get("task_type", "math")
        problem = interaction_kwargs["single_turn_prompt"]
        ground_truth = interaction_kwargs.get("ground_truth", "")
        persona_desc = (
            PERSONA_MATRIX[(expertise, goal)] if task_type not in SPEC_TASK_TYPES else ""
        )

        # Spec tasks supply q0 directly. The simulator only enters after
        # the agent has seen the public problem and asks for hidden details.
        # With replay_q0 (default) math works the same way: the first user turn
        # IS the problem statement, so the assistant never answers blind.
        if (task_type in SPEC_TASK_TYPES or self.replay_q0) and messages[-1]["role"] == "system":
            if not counterfactual:
                instance["turn_count"] += 1
                instance["response"] = problem
            return False, problem, 0.0, {
                "understood": False,
                "parse_failed": False,
                "connection_failed": False,
                "replayed_q0": True,
            }

        # ── Answer-checker termination gate (solving only) ────────────────────
        # Historically the episode ended when the SIMULATOR chose to emit
        # [[TERMINATE CHAT]], which made a model-generated control token the
        # arbiter of episode length. It is unreliable in both directions: the
        # expert falsely terminates on wrong answers, and it frequently fails to
        # terminate after a correct one, so dialogues ran on past the answer.
        # With this gate the strict checker decides: a correct boxed answer ends
        # the episode immediately, and the simulator is never asked.
        # Counterfactual calls are exempt — they must not end anything, they only
        # label the prediction slot.
        if (
            self.answer_gated_termination
            and not counterfactual
            and goal == "solving"
            and task_type not in SPEC_TASK_TYPES
            and ground_truth
            and messages
            and messages[-1].get("role") == "assistant"
        ):
            if _answer_is_correct(messages[-1].get("content", ""), ground_truth):
                if instance_id in self._instance_dict:
                    self._instance_dict[instance_id]["answer_gate_fired"] = True
                logger.debug(
                    "[CToMUserSimulator] answer gate: correct boxed answer, ending episode"
                )
                # Empty response -> the agent loop appends no user turn and stops.
                return True, "", 0.0, {
                    "understood": False,
                    "parse_failed": False,
                    "connection_failed": False,
                    "answer_gate": True,
                }

        if task_type == "writing":
            if not self.prompt_template_writing:
                raise FileNotFoundError(
                    "CollabLLM writing interaction requires prompts/sim_writing.txt"
                )
            user_model = self.user_model_writing
        elif task_type == "bigcodebench":
            if not self.prompt_template_bigcodebench:
                raise FileNotFoundError(
                    "CollabLLM BigCodeBench interaction requires prompts/sim_code.txt"
                )
            user_model = self.user_model_bigcodebench
        else:
            user_model = self._get_user_model(expertise)
        client = self._get_async_client()

        # Truncate oldest messages if context exceeds budget
        truncated = truncate_messages(messages, max_tokens=self.max_context_tokens)
        chat_history = self._parse_messages(truncated, strip_sys_prompt=True)

        # Track turn count
        turn_count = self._instance_dict.get(instance_id, {}).get("turn_count", 0)
        if instance_id in self._instance_dict and not counterfactual:
            self._instance_dict[instance_id]["turn_count"] = turn_count + 1

        # Build prompt from external template. Spec-task users hold hidden
        # information; math users retain the persona-based prompts.
        is_novice = expertise == "novice"
        if task_type == "writing":
            prompt = self.prompt_template_writing.format(
                problem=problem,
                hidden_information=interaction_kwargs.get("hidden_information", ""),
                target_words=interaction_kwargs.get("target_words", "300"),
                chat_history=chat_history,
                termination_signal=self.termination_signal,
            )
        elif task_type == "bigcodebench":
            # The BigCodeBench human holds the full task specification, INCLUDING
            # the required signature line, so the entry-point name is fixed. The
            # agent is told it up front in the system prompt and only the
            # BEHAVIOUR is withheld here.
            prompt = self.prompt_template_bigcodebench.format(
                problem=problem,
                hidden_information=interaction_kwargs.get("hidden_information", ""),
                chat_history=chat_history,
                termination_signal=self.termination_signal,
            )
        elif goal == "solving":
            tmpl = self.prompt_template_solving_novice if is_novice else self.prompt_template_solving
            prompt = tmpl.format(
                persona_desc=persona_desc,
                problem=problem,
                ground_truth=ground_truth,
                chat_history=chat_history,
                termination_signal=self.termination_signal,
                ending_rule=self._solving_ending_rule(),
            )
        else:
            tmpl = self.prompt_template_solving_novice if is_novice else self.prompt_template_solving
            prompt = tmpl.format(
                persona_desc=persona_desc,
                problem=problem,
                ground_truth=ground_truth,
                chat_history=chat_history,
                understood_token=self.understood_token,
                termination_signal=self.termination_signal,
            )

        max_parse_retries = 2  # extra retries specifically for JSON parse failures
        raw_response = ""
        connection_failed = False
        parsed = None
        t0 = time.time()

        for _parse_attempt in range(1 + max_parse_retries):
            raw_response = ""
            connection_failed = False

            for i in range(self.num_retries):
                try:
                    completion = await client.chat.completions.create(
                        model=user_model,
                        messages=[{"role": "user", "content": prompt}],
                        **self.completion_kwargs,
                    )
                    raw_response = completion.choices[0].message.content
                    connection_failed = False
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    print(f"\n\nFind error when generating simulated user response: {error_str}\n\n")
                    if "rate" in error_str or "429" in error_str:
                        backoff = max(2**i, 60)
                        print(
                            f"[CToMUserSimulator] Rate limited (attempt {i+1}/{self.num_retries}): {e}. "
                            f"Retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                    elif "connection" in error_str or "connect" in error_str or "timeout" in error_str:
                        backoff = min(2 ** (i + 1), 30) + random.uniform(0, 2)
                        connection_failed = True
                        print(
                            f"[CToMUserSimulator] Connection error (attempt {i+1}/{self.num_retries}) "
                            f"to {self.api_base}: {type(e).__name__}: {e}. "
                            f"Retrying in {backoff:.1f}s..."
                        )
                        await asyncio.sleep(backoff)
                    else:
                        backoff = min(2 ** i, 15)
                        connection_failed = True
                        print(
                            f"[CToMUserSimulator] Unexpected error (attempt {i+1}/{self.num_retries}) "
                            f"to {self.api_base}: {type(e).__name__}: {e}. "
                            f"Retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                    continue

            # If connection failed completely, don't attempt parse retry
            if connection_failed and not raw_response:
                break

            # Parse structured JSON output — extract the "response" field
            parsed = _parse_sim_json_response(
                raw_response,
                instance_id=instance_id,
                expertise=expertise,
                goal=goal,
                turn_count=turn_count,
            )

            if not parsed.get("_parse_failed"):
                break  # Parsed successfully

            # Parse failed — regenerate if we have retries left
            if _parse_attempt < max_parse_retries:
                print(
                    f"[CToMUserSimulator] JSON parse failed (attempt {_parse_attempt + 1}/{1 + max_parse_retries}), "
                    f"regenerating sim response for instance {instance_id}..."
                )
            else:
                logger.error(
                    f"[CToMUserSimulator] JSON parse failed after {1 + max_parse_retries} attempts "
                    f"for instance {instance_id}. Using raw text fallback."
                )

        gen_time = time.time() - t0

        # Log conversation (replaces proxy logging when proxy is skipped)
        if self._conversation_logger and raw_response:
            self._conversation_logger.log({
                "timestamp": time.time(),
                "gen_time_s": round(gen_time, 3),
                "backend_idx": self._backend_idx,
                "model": user_model,
                "num_turns": len(messages),
                "messages": messages,
                "response": raw_response,
                "temperature": self.completion_kwargs.get("temperature"),
                "max_tokens": self.completion_kwargs.get("max_tokens"),
            })

        if connection_failed and not raw_response:
            logger.error(
                f"[CToMUserSimulator] All {self.num_retries} retries exhausted "
                f"(backend={self.api_base}). "
                f"Signaling termination for instance {instance_id}."
            )
            if instance_id in self._instance_dict and not counterfactual:
                self._instance_dict[instance_id]["connection_failed"] = True
            return True, "", 0.0, {"connection_failed": True, "understood": False}

        response = parsed["response"]

        if parsed.get("_parse_failed"):
            logger.error(
                f"[CToMUserSimulator] JSON parse failed for instance {instance_id}. "
                f"Raw response (first 200 chars): {raw_response[:200]}"
            )

        if not counterfactual:
            instance["response"] = response
            instance["connection_failed"] = False

        # Determine termination
        should_terminate = False
        if task_type in SPEC_TASK_TYPES:
            should_terminate = self.termination_signal in response
        elif goal == "solving":
            # With the answer gate on, the checker above is the ONLY way a solving
            # episode ends. Reaching here means the answer was not correct, so the
            # simulator's token is deliberately ignored — it is exactly the
            # false-termination case the gate exists to remove.
            should_terminate = (
                False
                if self.answer_gated_termination
                else self.termination_signal in response
            )

        reward = 0.0
        extra_info = {
            "understood": self.understood_token in response,
            "sim_thought": parsed.get("thought", ""),
            "sim_status": parsed.get("current_status", ""),
            "parse_failed": parsed.get("_parse_failed", False),
            "connection_failed": False,
        }

        logger.debug(f"[CToMUserSimulator] User: {response}")
        return should_terminate, response, reward, extra_info

    def is_connection_failed(self, instance_id: str) -> bool:
        """Check if the last generate_response call for this instance failed due to connection."""
        return self._instance_dict.get(instance_id, {}).get("connection_failed", False)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    def _parse_messages(self, messages, strip_sys_prompt=True):
        if messages is None:
            return ""
        if strip_sys_prompt:
            messages = [msg for msg in messages if msg["role"] != "system"]
        messages = [remove_think_block(msg) for msg in messages]
        chat = "\n".join(f"**{m['role'].capitalize()}**: {m['content']}" for m in messages)
        return chat
