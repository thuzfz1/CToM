# Submission markers and final-answer extraction.
#
# All three domains end an episode the same way: the assistant prefixes its
# finished work with ANSWER_MARKER, and everything after the marker is the
# submission. Writing takes the text, coding takes the code.
#
# `strip_code_fences` and `_extract_code_from_text` also recover a submission
# from an unmarked turn, because the policy does not always emit the marker.

import re

def strip_code_fences(code: str) -> str:
    """Pull the body out of a ```python ...``` / ``` ...``` fenced block, if present.

    Fence-agnostic and
    safe on unfenced input (returns it unchanged).
    """
    if "```python" in code:
        return code.split("```python", 1)[1].split("```", 1)[0]
    if "```" in code:
        parts = code.split("```")
        if len(parts) >= 3:
            return parts[1]
    return code


# Final-answer extraction ─────────────────────────────────────────────────────

# The backend agent signals completion with this marker, followed by raw code.
ANSWER_MARKER = "I WANT TO ANSWER:"
# The policy writes the marker in whatever case it likes — measured in a smoke run: 18.5% of assistant turns declared a submission but only 1.7% did
# so in this exact case. Matching case-sensitively meant `answer_is_final` almost
# never fired, so episodes ran on past the submission and the code that WAS
# written scored as no-submit. Prose instructions cannot enforce casing; the
# reader has to be tolerant.
_ANSWER_MARKER_RE = re.compile(re.escape(ANSWER_MARKER), re.IGNORECASE)


def find_answer_marker(text: str):
    """Return the match for the submission marker in any case, or None."""
    return _ANSWER_MARKER_RE.search(text or "")

# A line that starts a Python code construct (used to salvage unfenced code).
_CODE_START_RE = re.compile(r"^\s*(?:def |class |import |from |@)")


def _extract_code_from_text(text: str) -> str:
    """Best-effort pull of a runnable Python snippet out of ONE assistant message.

    Handles, in order: text after the answer marker, a fenced ```python block,
    or the tail of the message starting at the first def/class/import line.
    Returns "" if the message carries no code. Trailing prose after unfenced code
    is left in — check_correctness simply scores it 0 if it fails to exec, same
    as before, so this can only help, never hurt.
    """
    if not text:
        return ""
    _m = find_answer_marker(text)
    if _m:
        text = text[_m.end():]
    # A ```python / ``` fenced block wins when present.
    fenced = strip_code_fences(text)
    if fenced is not text:
        fenced = fenced.strip()
        if "def " in fenced or "class " in fenced:
            return fenced
    # Unfenced: grab from the first code-looking line to the end of the message.
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if _CODE_START_RE.match(ln):
            return "\n".join(lines[i:]).strip()
    return ""


def extract_final_code(messages: list) -> str:
    """Extract the agent's final submitted code from a conversation.

    Preference order:
      1. the last assistant turn that contains ANSWER_MARKER (explicit submission);
      2. the last assistant turn that CONTAINS runnable code even WITHOUT the
         marker — so an agent that wrote a complete function but forgot to say
         "I WANT TO ANSWER:" is still scored on that code (fair comparison,
         matches how a human collaborator would read the answer off the chat);
      3. otherwise the last non-empty assistant turn verbatim.
    The returned string may still be fenced; check_correctness strips fences.
    """
    def _role(m):
        return m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")

    def _content(m):
        return m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")

    assistant_contents = [_content(m) for m in messages if _role(m) == "assistant"]
    # 1. Explicit marker submission.
    for content in reversed(assistant_contents):
        m = find_answer_marker(content) if content else None
        if m:
            code = _extract_code_from_text(content)
            return code or content[m.end():].strip()
    # 2. Code produced without the marker (the fairness fix).
    for content in reversed(assistant_contents):
        code = _extract_code_from_text(content)
        if code and "def " in code:
            return code
    # 3. Fallback: last non-empty assistant turn.
    for content in reversed(assistant_contents):
        if content and content.strip():
            return content.strip()
    return ""


def answer_is_final(content: str) -> bool:
    """True if an assistant answer declares the final submission (backend task).

    Case-insensitive on purpose — see the note on ANSWER_MARKER.
    """
    return bool(content) and find_answer_marker(content) is not None


def forced_submit_marker(draft: str) -> str:
    """Marker text to force at the START of a REVISION whose DRAFT submitted.

    sft_v4/sft_v5 decode draft -> prediction -> revision, and only the revision
    is visible and marker-checked. Measured on writing eval 945368/945369 at
    step 208, the model wrote "I WANT TO ANSWER:" in the draft and dropped it
    from the revision in 57% (aux0.1) and 15% (aux0.0) of conversations; in
    67/67 inspected cases the revision WAS the document with the marker stripped
    — not a change of mind. The document still scored (extract_final_document
    falls back on length) but the episode never terminated, so it kept re-asking
    until it hit the turn cap.

    This is a DECODE-TIME decision: it is made before the revision exists, so
    the marker can be emitted as controller tokens and the model writes the
    document after it. Prepending to the text afterwards instead would leave the
    token ids without the marker, so the policy would never learn to emit it and
    the trained tokens would not match the visible message.

    Returns "" when the draft did not submit, so this can only ever propagate a
    decision the model itself made. Callers gate on the task family
    (SPEC_TASK_TYPES); math has no submission marker and must never reach here.
    Pair with `compose_revision` to join the result to the sampled text.
    """
    return ANSWER_MARKER + "\n" if answer_is_final(draft) else ""


def compose_revision(forced_marker: str, revision: str) -> str:
    """Join a forced marker to the sampled revision without double-stamping.

    The forced marker is already in the model's context when the revision is
    sampled, so the continuation normally does not repeat it — but if it does,
    one marker is what the reader and `answer_is_final` need, not two.
    """
    if not forced_marker or not revision:
        return revision
    return revision if answer_is_final(revision) else forced_marker + revision
