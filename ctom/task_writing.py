# CollabLLM MediumDocEdit-Chat scoring utilities.
#
# The writing task reward: it turns a finished
# collaborative-writing dialogue into a scalar R_task in [0, 1] by comparing the
# document the agent submitted against the target article the human simulator was
# holding.
#
# Metric choice. CollabLLM scores this dataset with
# The reported metric is ROUGE-L F1 (LCS over words): it both ranks
# correctly and keeps a usable spread, so it is the default reward.
#
# Floor note: rouge_l does not start at 0 -- two unrelated articles share stop
# words and score ~0.11. Treat 0.11, not 0.0, as "no content transferred". GRPO
# normalizes within the group so the offset is harmless there; under GAE the
# critic absorbs it.
#
# Select with CTOM_WRITING_METRIC=rouge_l, or per-dataset
# via interaction_kwargs["writing_metric"].
#
# No third-party dependency on purpose (the training venv has no nltk /
# rouge_score / sacrebleu), and everything here is pure-python and deterministic,
# so the reward worker cannot stall or drift.

import math
import os
import re
from collections import Counter
from typing import Iterable, Sequence

# Same submission marker as the backend task, so `answer_is_final` and the
# agent-loop termination path are shared verbatim between the two domains.
from ctom.submission import ANSWER_MARKER, answer_is_final, find_answer_marker  # noqa: F401

DEFAULT_METRIC = os.getenv("CTOM_WRITING_METRIC", "rouge_l")

# An assistant turn shorter than this (in words) is treated as conversation, not
# as a submitted document, when no explicit marker is present. Target articles
# run 138-442 words; a genuine draft is well clear of this floor.
MIN_DOCUMENT_WORDS = int(os.getenv("CTOM_WRITING_MIN_DOC_WORDS", "60"))

_WORD_RE = re.compile(r"\w+", re.UNICODE)
# A whole message wrapped in one fenced block: strip the fence, keep the prose.
_FENCED_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())




def _ngram_counts(seq: Sequence, n: int) -> Counter:
    if len(seq) < n:
        return Counter()
    return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))








def _lcs_length(a: Sequence, b: Sequence) -> int:
    """Length of the longest common subsequence, O(len(a) * len(b)) time and
    O(min) memory. Documents are a few hundred words, so this is cheap."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 over word tokens — the default R_task for the writing domain."""
    hyp, ref = _words(hypothesis), _words(reference)
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(hyp), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


_METRICS = {"rouge_l": rouge_l}


def document_similarity(hypothesis: str, reference: str, metric: str = "") -> float:
    """Score a submitted document against the target article, clipped to [0, 1]."""
    name = (metric or DEFAULT_METRIC or "rouge_l").strip()
    if name not in _METRICS:
        raise ValueError(
            f"Unknown writing metric {name!r}; expected one of {sorted(_METRICS)}"
        )
    if not hypothesis or not hypothesis.strip() or not reference or not reference.strip():
        return 0.0
    return max(0.0, min(1.0, _METRICS[name](hypothesis, reference)))


def strip_document_fences(text: str) -> str:
    """Unwrap a document the model wrapped in a single ``` block."""
    m = _FENCED_RE.match(text or "")
    return m.group(1) if m else (text or "")


def _role(m) -> str:
    return m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")


def _content(m) -> str:
    return m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")


def extract_final_document(messages: Iterable) -> str:
    """Pull the agent's submitted document out of a conversation.

    Preference order mirrors extract_final_code:
      1. the last assistant turn carrying ANSWER_MARKER -> the text after it;
      2. otherwise the latest assistant turn long enough to be a document, so an
         agent that wrote the article but forgot the marker is still scored on
         what it wrote rather than on a clarifying question;
      3. otherwise the last non-empty assistant turn.
    """
    assistant_contents = [_content(m) for m in messages if _role(m) == "assistant"]

    for content in reversed(assistant_contents):
        m = find_answer_marker(content) if content else None
        if m:
            submitted = strip_document_fences(content[m.end():].strip()).strip()
            if submitted:
                return submitted
            # Marker with nothing after it -> fall through to the length rule.
            break

    for content in reversed(assistant_contents):
        body = strip_document_fences(content or "").strip()
        if len(_words(body)) >= MIN_DOCUMENT_WORDS:
            return body

    for content in reversed(assistant_contents):
        if content and content.strip():
            return strip_document_fences(content).strip()
    return ""
