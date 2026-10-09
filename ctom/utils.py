import logging
import os
import re

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Task families where the human holds a hidden specification rather than being
# drawn from the persona matrix: collaborative writing ("writing") and
# CollabLLM MediumDocEdit writing ("writing") and CollabLLM BigCodeBench
# ("bigcodebench"). Everything that distinguishes them from the math personas is
# shared -- q0 is replayed verbatim, there is no persona and no answer gate, and
# the episode ends when the agent submits behind the answer marker -- so the
# pipeline branches on membership in this tuple, not on the individual name.
# Defined here because both the interaction and the agent loop must agree on it.
#
# function, BigCodeBench by running a shipped unittest suite. They share the
# submission marker and nothing else in the scoring path.
SPEC_TASK_TYPES = ("writing", "bigcodebench")

# ──────────────────────────────────────────────────────────────────────
# Message parsing utilities
# ──────────────────────────────────────────────────────────────────────

# Legacy tag format, predating the JSON contract. Kept only so old transcripts
# stay parseable; nothing emits it any more.
LEGACY_PREDICTION_START_TAG = "<user_prediction>"
LEGACY_PREDICTION_END_TAG = "</user_prediction>"

# Canonical controller-owned delimiters (sft_v3 and, since 2026-07-20, the
# assistant's own output format — see build_math.TAG_SYSTEM_PROMPT).
# Defined here rather than in prediction_loss.py because utils is the base module
# both the agent loop and the prediction-loss path import from; prediction_loss
# re-exports these names for its existing importers.
ANSWER_START_TAG = "[ANSWER_START]"
ANSWER_END_TAG = "[ANSWER_END]"
PREDICTION_START_TAG = "[PREDICTION_START]"
PREDICTION_END_TAG = "[PREDICTION_END]"

# sft_v4 (predict-then-revise) adds a third controller-owned phase. The turn is
# draft -> prediction -> revision, and only the REVISION is shown to the user, so
# every downstream consumer that reads ["answer"] must see the revision. The
# draft is returned under its own key for analysis and is never sent to the sim.
REVISION_START_TAG = "[REVISION_START]"
REVISION_END_TAG = "[REVISION_END]"

_ALL_PHASE_TAGS = (
    ANSWER_START_TAG,
    ANSWER_END_TAG,
    PREDICTION_START_TAG,
    PREDICTION_END_TAG,
    REVISION_START_TAG,
    REVISION_END_TAG,
)


def _get_role(msg) -> str:
    return msg.role if hasattr(msg, "role") else msg.get("role", "")


def _get_content(msg) -> str:
    return msg.content if hasattr(msg, "content") else msg.get("content", "")


def parse_messages(messages, strip_sys_prompt=True):
    """Format a message list into a readable string."""
    if messages is None:
        return ""
    if strip_sys_prompt:
        messages = strip_system_prompt(messages)
    chat = "\n".join(
        f"**{_get_role(m).capitalize()}**: {_get_content(m)}" for m in messages
    )
    return chat


def strip_system_prompt(messages):
    """Remove system messages from a message list."""
    return [msg for msg in messages if _get_role(msg) != "system"]


# ──────────────────────────────────────────────────────────────────────
# Think-block handling
# ──────────────────────────────────────────────────────────────────────

def remove_think_block(msg: dict) -> dict:
    """Remove <think>.*?</think> from message content."""
    if "content" in msg and isinstance(msg["content"], str):
        msg = dict(msg)  # shallow copy
        msg["content"] = re.sub(r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL).strip()
    return msg


def strip_think_blocks_str(text: str) -> str:
    """Remove <think>...</think> blocks from a raw string."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def is_valid_messages(msg: dict) -> bool:
    """
    Validate a message dict. Checks:
    1. <think> is paired with </think>
    2. Content inside and outside <think> is non-empty
    3. At most one <think> block (no nesting)
    4. Not empty after removing trailing <|im_end|>
    """
    content = msg.get("content")
    if not isinstance(content, str):
        return True

    if not content.strip():
        return False

    num_think_open = content.count("<think>")
    num_think_close = content.count("</think>")

    if num_think_open != num_think_close:
        return False
    if num_think_open > 1:
        return False

    if num_think_open == 0:
        visible_content = content
    else:
        # /no_think mode legitimately emits an EMPTY <think></think> block, so do
        # not reject an empty think body. Strip the think block and validate only
        # the visible content (the answer/JSON after </think>), which the final
        # check below requires to be non-empty.
        visible_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    visible_content = visible_content.strip()
    if visible_content.endswith("<|im_end|>"):
        visible_content = visible_content[: -len("<|im_end|>")]

    if not visible_content.strip():
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# JSON extraction (lenient parser for LLM outputs)
# ──────────────────────────────────────────────────────────────────────

def extract_json(s):
    """
    Extract and parse a JSON object from a potentially messy LLM response.
    Handles single quotes, triple quotes, and lenient value parsing.
    """

    def convert_value(value):
        true_values = {"true": True, "false": False, "null": None}
        value_lower = value.lower()
        if value_lower in true_values:
            return true_values[value_lower]
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    def parse_number(s, pos):
        start = pos
        while pos < len(s) and s[pos] in "-+0123456789.eE":
            pos += 1
        num_str = s[start:pos]
        try:
            if "." in num_str or "e" in num_str.lower():
                return float(num_str), pos
            else:
                return int(num_str), pos
        except ValueError:
            logger.error(f"Invalid number at position {start}: {num_str}")
            raise

    def skip_whitespace(s, pos):
        while pos < len(s) and s[pos] in " \t\n\r":
            pos += 1
        return pos

    def parse_string(s, pos):
        quote_char = s[pos]
        assert quote_char in ('"', "'")
        pos += 1
        result = ""
        while pos < len(s):
            c = s[pos]
            if c == "\\":
                pos += 1
                if pos >= len(s):
                    raise ValueError("Invalid escape sequence")
                c = s[pos]
                escape_sequences = {
                    "n": "\n", "t": "\t", "r": "\r",
                    "\\": "\\", quote_char: quote_char,
                }
                result += escape_sequences.get(c, c)
            elif c == quote_char:
                pos += 1
                return convert_value(result), pos
            else:
                result += c
            pos += 1
        raise ValueError("Unterminated string")

    def parse_key(s, pos):
        pos = skip_whitespace(s, pos)
        if s[pos] in ('"', "'"):
            key, pos = parse_string(s, pos)
            return key, pos
        else:
            raise ValueError(f"Expected string for key at position {pos}")

    def parse_object(s, pos):
        obj = {}
        assert s[pos] == "{"
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "}":
            pos = skip_whitespace(s, pos)
            key, pos = parse_key(s, pos)
            pos = skip_whitespace(s, pos)
            if pos >= len(s) or s[pos] != ":":
                raise ValueError(f'Expected ":" at position {pos}')
            pos += 1
            pos = skip_whitespace(s, pos)
            value, pos = parse_value(s, pos)
            obj[key] = value
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "}":
                break
            elif pos < len(s) and s[pos] != "}":
                raise ValueError(f'Expected "," or "}}" at position {pos}')
        if pos >= len(s) or s[pos] != "}":
            raise ValueError(f'Expected "}}" at position {pos}')
        pos += 1
        return obj, pos

    def parse_array(s, pos):
        lst = []
        assert s[pos] == "["
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "]":
            value, pos = parse_value(s, pos)
            lst.append(value)
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "]":
                break
            elif pos < len(s) and s[pos] != "]":
                raise ValueError(f'Expected "," or "]" at position {pos}')
        if pos >= len(s) or s[pos] != "]":
            raise ValueError(f'Expected "]" at position {pos}')
        pos += 1
        return lst, pos

    def parse_triple_quoted_string(s, pos):
        if s[pos: pos + 3] == "'''":
            quote_str = "'''"
        elif s[pos: pos + 3] == '"""':
            quote_str = '"""'
        else:
            raise ValueError(f"Expected triple quotes at position {pos}")
        pos += 3
        result = ""
        while pos < len(s):
            if s[pos: pos + 3] == quote_str:
                pos += 3
                return convert_value(result), pos
            else:
                result += s[pos]
                pos += 1
        raise ValueError("Unterminated triple-quoted string")

    def parse_value(s, pos):
        pos = skip_whitespace(s, pos)
        if pos >= len(s):
            raise ValueError("Unexpected end of input")
        if s[pos] == "{":
            return parse_object(s, pos)
        elif s[pos] == "[":
            return parse_array(s, pos)
        elif s[pos: pos + 3] in ("'''", '"""'):
            return parse_triple_quoted_string(s, pos)
        elif s[pos] in ('"', "'"):
            return parse_string(s, pos)
        elif s[pos: pos + 4].lower() == "true":
            return True, pos + 4
        elif s[pos: pos + 5].lower() == "false":
            return False, pos + 5
        elif s[pos: pos + 4].lower() == "null":
            return None, pos + 4
        elif s[pos] in "-+0123456789.":
            return parse_number(s, pos)
        else:
            raise ValueError(f"Unexpected character at position {pos}: {s[pos]}")

    json_start = s.index("{")
    json_end = s.rfind("}")
    s = s[json_start: json_end + 1]
    s = s.strip()
    result, pos = parse_value(s, 0)
    pos = skip_whitespace(s, pos)
    if pos != len(s):
        raise ValueError(f"Unexpected content at position {pos}")
    return result


# ──────────────────────────────────────────────────────────────────────
# Context window truncation
# ──────────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: word count * 1.3 (covers sub-word tokenization)."""
    return int(len(text.split()) * 1.3)


def truncate_messages(
    messages: list,
    max_tokens: int,
    keep_last: int = 2,
) -> list:
    """Truncate a conversation by dropping the earliest non-system messages.

    Keeps:
      - All system messages (always at the front)
      - At least the last ``keep_last`` non-system messages (so the model
        always sees the most recent exchange)

    Drops the oldest non-system messages first until the total estimated
    token count is within ``max_tokens``.

    Args:
        messages: Conversation message list (dicts or objects with role/content).
        max_tokens: Maximum total token budget. 0 or negative disables truncation.
        keep_last: Minimum number of recent non-system messages to always keep.

    Returns:
        A (possibly shorter) list of messages.  Never mutates the input.
    """
    if max_tokens <= 0:
        return messages

    system_msgs = [m for m in messages if _get_role(m) == "system"]
    non_system_msgs = [m for m in messages if _get_role(m) != "system"]

    def _total_tokens(msgs):
        return sum(_estimate_tokens(_get_content(m)) for m in msgs)

    # Fast path: already fits
    if _total_tokens(messages) <= max_tokens:
        return messages

    # Protect the last `keep_last` non-system messages from truncation
    protected = non_system_msgs[-keep_last:] if keep_last > 0 else []
    droppable = non_system_msgs[:-keep_last] if keep_last > 0 else list(non_system_msgs)

    # Drop from the front (oldest) until we fit
    while droppable and _total_tokens(system_msgs + droppable + protected) > max_tokens:
        dropped = droppable.pop(0)
        logger.debug(
            f"Truncated {_get_role(dropped)} message "
            f"({_estimate_tokens(_get_content(dropped))} est. tokens)"
        )

    result = system_msgs + droppable + protected

    if len(result) < len(messages):
        n_dropped = len(messages) - len(result)
        logger.info(
            f"Context truncation: dropped {n_dropped} earliest messages "
            f"({len(messages)} -> {len(result)} messages, "
            f"~{_total_tokens(result)} est. tokens)"
        )

    return result


# ──────────────────────────────────────────────────────────────────────
# CToM-specific utilities
# ──────────────────────────────────────────────────────────────────────

def extract_prediction(content: str) -> tuple[str, str]:
    """Extract prediction from assistant content (JSON format)."""
    parsed = parse_assistant_response(content)
    return parsed["answer"], parsed["prediction"]


def has_prediction_block(content: str) -> bool:
    """Check if content contains a valid JSON response with prediction."""
    parsed = parse_assistant_response(content)
    return bool(parsed["prediction"])


def format_persona_context(expertise: str, goal: str) -> str:
    """Format persona information for display/logging."""
    return f"[Persona: expertise={expertise}, goal={goal}]"


def count_user_tokens(messages: list, approximate: bool = True) -> int:
    """Count total tokens generated by the user across all turns."""
    total = 0
    for msg in messages:
        if _get_role(msg) == "user":
            total += len(_get_content(msg).split())
    return total


def count_turns(messages: list) -> tuple[int, int]:
    """
    Count user and assistant turns.
    Returns (user_turns, assistant_turns).
    """
    user_turns = 0
    assistant_turns = 0
    for msg in messages:
        role = _get_role(msg)
        if role == "user":
            user_turns += 1
        elif role == "assistant":
            assistant_turns += 1
    return user_turns, assistant_turns


def is_valid_response(content: str) -> bool:
    msg = {"content": content}
    if not is_valid_messages(msg):
        return False
    if not has_prediction_block(content):
        logger.warning("Assistant response missing 'prediction' field in JSON")
    return True


def extract_sim_response(text: str) -> str:
    """Extract the 'response' field from user simulator JSON output.

    The simulator returns structured JSON like:
        ```json
        {"current_status": "...", "thought": "...", "response": "..."}
        ```
    This extracts just the 'response' field.  Returns text as-is if
    parsing fails (graceful degradation).
    """
    import json as _json
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        parsed = _json.loads(clean)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed["response"]
    except (ValueError, TypeError):
        pass
    # Try finding a JSON object in the text
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            parsed = _json.loads(match.group())
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed["response"]
        except (ValueError, TypeError):
            pass
    return text


import json

def _repair_json_escapes(s: str) -> str:
    """Double-up backslashes that aren't valid JSON escape sequences."""
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    return re.sub(
        r'\\(?!["\\/bfnrtu])',
        r'\\\\',
        s,
    )


def _escape_control_chars_in_json_strings(json_str: str) -> str:
    """Escape literal control characters (newline, tab, ...) that appear INSIDE
    JSON string values. LLMs routinely emit raw 0x0A newlines inside a multi-line
    ``answer``, which json.loads rejects as "Invalid control character". Walk
    char-by-char tracking string state; escape control chars inside strings,
    leave formatting whitespace outside strings alone.

    (Mirror of the user-simulator parser's escaper in user_simulator.py.)"""
    _CTRL_MAP = {'\n': '\\n', '\r': '\\r', '\t': '\\t', '\x08': '\\b', '\x0c': '\\f'}
    result: list[str] = []
    i, n, in_string = 0, len(json_str), False
    while i < n:
        ch = json_str[i]
        if not in_string:
            if ch == '"':
                in_string = True
            result.append(ch)
            i += 1
        else:
            if ch == '\\' and i + 1 < n:      # already-escaped sequence — keep both
                result.append(ch); result.append(json_str[i + 1]); i += 2
            elif ch == '"':
                in_string = False; result.append(ch); i += 1
            elif ch in _CTRL_MAP:             # literal control char inside string — escape
                result.append(_CTRL_MAP[ch]); i += 1
            else:
                result.append(ch); i += 1
    return ''.join(result)


def _truncation_closure_candidates(s: str) -> list[str]:
    """Repair candidates for a generation that was length-clipped mid-JSON:
    terminate the open string and append the missing closing braces, so only the
    partial answer/prediction value is lost rather than the whole parse. A second
    candidate trims back to the last complete string VALUE first, for clips that
    land inside a key or between members. Returns [] when the text is already
    balanced (not truncated)."""
    in_string = False
    depth = 0
    expecting_value = False  # saw ':' after the current member's key
    last_value_end = -1      # index of the closing quote of the last complete string value
    depth_at_value = 0
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
                if expecting_value:
                    last_value_end = i
                    depth_at_value = depth
                    expecting_value = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
                expecting_value = False
            elif ch == '}':
                depth -= 1
                expecting_value = False
            elif ch == ':':
                expecting_value = True
            elif ch == ',':
                expecting_value = False
        i += 1
    if not in_string and depth <= 0:
        return []
    candidates = []
    head = s[:-1] if i > n else s  # i overshot n: drop the dangling escape backslash
    if in_string:
        head += '"'
    candidates.append(head + '}' * max(depth, 0))
    if last_value_end >= 0 and depth_at_value > 0 and last_value_end + 1 < n:
        candidates.append(s[: last_value_end + 1] + '}' * depth_at_value)
    return candidates


def _json_repair_candidates(s: str) -> list[str]:
    """Progressively-repaired variants to try with json.loads: raw, then literal
    control chars escaped, then also invalid backslashes doubled, and finally —
    only reached when the intact variants all fail — truncation closures for
    length-clipped generations. Order matters: escape control chars first so
    genuine ``\\n`` escapes survive."""
    ctrl = _escape_control_chars_in_json_strings(s)
    out = [s, ctrl, _repair_json_escapes(ctrl)]
    for closed in _truncation_closure_candidates(ctrl):
        out.append(closed)
        out.append(_repair_json_escapes(closed))
    return out


def _parse_tag_format(text: str) -> dict | None:
    """Parse the [ANSWER_START]/[PREDICTION_START] delimiter format.

    Returns None when the text carries no answer delimiter at all, so the caller
    can fall through to the legacy JSON parser (old checkpoints still emit JSON).

    Deliberately lenient about the CLOSING delimiters: a length-clipped
    generation often has [ANSWER_START] and its body but no [ANSWER_END], and
    dropping that answer would be worse than keeping a truncated one. A missing
    closer therefore reads to the start of the next delimiter, or to end-of-text.
    """
    if (
        ANSWER_START_TAG not in text
        and PREDICTION_START_TAG not in text
        and REVISION_START_TAG not in text
    ):
        return None

    def _section(start_tag: str, end_tag: str) -> str:
        start = text.find(start_tag)
        if start == -1:
            return ""
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            # No closer: stop at whichever other delimiter comes next.
            candidates = [
                pos
                for pos in (text.find(tag, start) for tag in _ALL_PHASE_TAGS)
                if pos != -1
            ]
            end = min(candidates) if candidates else len(text)
        return text[start:end].strip()

    answer = _section(ANSWER_START_TAG, ANSWER_END_TAG)
    prediction = _section(PREDICTION_START_TAG, PREDICTION_END_TAG)
    revision = _section(REVISION_START_TAG, REVISION_END_TAG)

    # An [ANSWER_START] that opened but produced nothing is not a usable parse;
    # let the JSON path have a turn at it rather than returning a blank answer.
    if not answer and not prediction and not revision:
        return None
    if REVISION_START_TAG in text:
        # sft_v4: the revision is the visible answer. Callers reading ["answer"]
        # (reward scoring, the sim transcript, eval) must never see the draft.
        # A clipped/empty revision falls back to the draft rather than losing the
        # turn entirely; phase_validity is what penalizes that case in training.
        return {
            "answer": revision or answer,
            "prediction": prediction,
            "draft": answer,
        }
    return {"answer": answer, "prediction": prediction}


def parse_assistant_response(response_text: str) -> dict:
    text = strip_think_blocks_str(response_text.strip())

    # Tag format first (2026-07-20 onward); JSON below stays for the older
    # checkpoints trained against build_math.SYSTEM_PROMPT.
    tagged = _parse_tag_format(text)
    if tagged is not None:
        return tagged

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try the whole text, then a JSON object extracted from it, each with
    # progressive repairs. Control-char escaping handles the common case where
    # the model emits raw newlines inside a multi-line "answer" (would otherwise
    # break json.loads and silently drop the prediction → R_pred broken).
    match = re.search(r"\{.*\}", text, re.DOTALL)
    sources = [text] + ([match.group()] if match else [])
    for src in sources:
        for candidate in _json_repair_candidates(src):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                # Coerce to strings: `.get(k, "")` still returns None when the key
                # is present with a JSON null (e.g. {"answer": null}), which would
                # propagate as content=None and crash Message(**msg) construction
                # (content union has no None member). Empty string instead routes
                # to the existing empty-generation handling.
                ans = parsed.get("answer")
                pred = parsed.get("prediction")
                return {
                    "answer": str(ans) if ans is not None else "",
                    "prediction": str(pred) if pred is not None else "",
                }

    return {"answer": response_text.strip(), "prediction": ""}