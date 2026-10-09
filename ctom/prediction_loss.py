# Copyright 2025 The CToM Authors
# Auxiliary "predict-the-next-user-turn" SFT objective.
#
# Background
# ----------
# In `sft` prediction mode the assistant's next-turn prediction is no longer a
# reward term.  Instead, for every assistant turn a_t that is followed by a real
# user turn q_{t+1}, we build a supervised sequence in which the assistant's
# prediction slot is *teacher-forced* toward q_{t+1}, and a cross-entropy loss is
# applied ONLY over the q_{t+1} tokens.  The sampled prediction e_t the model
# produced during rollout is discarded from this sequence (it only survives in
# the PPO rollout, where its tokens are masked out of the PPO response).
#
# One sequence is produced per conversation (several masked spans, one per
# assistant turn), so the SFT batch is row-aligned 1:1 with the PPO batch and can
# ride along through dispatch / mini-batching and land in the same optimizer step.
#
# Token-span localization uses the fast tokenizer's offset mapping, so this is
# robust to the exact chat-template formatting (no hard-coded role markers).

import json
import logging
import os

import torch

from ctom.utils import (
    ANSWER_END_TAG,
    ANSWER_START_TAG,
    PREDICTION_END_TAG,
    PREDICTION_START_TAG,
    REVISION_END_TAG,
    REVISION_START_TAG,
    _get_content,
    _get_role,
    extract_sim_response,
)

# The user simulator's "you solved it, stop" token. It is a control signal, not a
# user utterance: on expert rows ~30% of prediction targets are exactly this string
# (novice rows: 0%), so on expert-only data a third of the supervision budget would
# teach a fixed 5-token constant. Those turns also score F1 ~0.04 at eval, i.e. the
# slot buys nothing. Measured on runs 893455 / 891422.
TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"


def is_terminate_only_target(target: str) -> bool:
    """True when a supervision target carries no content beyond the stop signal.

    Only strips the signal itself — a turn like "Great, thanks! [[TERMINATE CHAT]]"
    still has a real user utterance in it and is kept, with the signal removed by
    `strip_termination_signal` so the model is never taught to emit the token.
    """
    if not target:
        return True
    return not target.replace(TERMINATION_SIGNAL, "").strip()


def strip_termination_signal(target: str) -> str:
    """Remove the control token, keeping whatever the user actually said."""
    if not target:
        return target
    return target.replace(TERMINATION_SIGNAL, "").strip()

__all__ = [
    "ANSWER_START_TAG",
    "ANSWER_END_TAG",
    "PREDICTION_START_TAG",
    "PREDICTION_END_TAG",
    "REVISION_START_TAG",
    "REVISION_END_TAG",
]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Marker for the prediction key as emitted by json.dumps with default
# separators: `{"answer": "...", "prediction": "..."}`.
_PRED_KEY_MARKER = '"prediction":'

# sft_v3 controller-owned delimiters are fixed tokens, never sampled.
# Now defined in ctom/utils.py (single source of truth) and re-exported above,
# since the assistant emits the same delimiters as of 2026-07-20.


def _find_prediction_value_charspan(text):
    """Locate the character span of the prediction VALUE in a raw assistant
    output string (the model's own `{"answer": ..., "prediction": "..."}`).

    Tolerant of single/double quotes and arbitrary whitespace around the key and
    colon. Returns (start, end) char indices of the value (excluding the wrapping
    quotes), or None if no well-formed prediction value is found. Uses the LAST
    `prediction` key occurrence (the real field, in case the answer mentions it).
    """
    key = text.rfind("prediction")
    if key == -1:
        return None
    # Walk to the colon after the key.
    i = key + len("prediction")
    n = len(text)
    # skip closing quote of the key and whitespace up to ':'
    while i < n and text[i] in "\"' \t\r\n":
        i += 1
    if i >= n or text[i] != ":":
        return None
    i += 1
    while i < n and text[i] in " \t\r\n":
        i += 1
    if i >= n or text[i] not in "\"'":
        return None
    quote = text[i]
    val_start = i + 1
    j = val_start
    while j < n:
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == quote:
            return val_start, j
        j += 1
    # Unterminated: the generation was length-clipped inside the prediction
    # value (a well-formed block never ends here). Treat the remainder as the
    # partial value so the clipped prediction is still masked out of PPO and
    # still marks the sft_v2 context boundary.
    if val_start < n:
        return val_start, n
    return None


def locate_prediction_value_token_span(token_ids, tokenizer):
    """Find the [start, end) token-index span of the prediction value inside a
    block of just-generated assistant tokens, so it can be masked out of the PPO
    response. Returns None (fail-open: leave tokens in PPO) if the span can't be
    reliably localized.
    """
    token_ids = list(token_ids)
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    # Re-encode must round-trip to the exact ids, else offsets don't align.
    if enc["input_ids"] != token_ids:
        return None
    span = _find_prediction_value_charspan(text)
    if span is None:
        return None
    val_start, val_end = span
    idxs = [
        k for k, (a, b) in enumerate(enc["offset_mapping"])
        if b > a and val_start <= a < val_end
    ]
    if not idxs:
        return None
    return idxs[0], idxs[-1] + 1


def _iter_assistant_targets(messages):
    """Yield (assistant_index, next_user_content) for every assistant turn that
    is immediately followed (before any other assistant turn) by a non-empty
    user turn. That user turn is the teacher-forcing target q_{t+1}.
    """
    n = len(messages)
    for i in range(n):
        if _get_role(messages[i]) != "assistant":
            continue
        nxt = None
        for j in range(i + 1, n):
            role = _get_role(messages[j])
            if role == "user":
                content = _get_content(messages[j])
                if content and content.strip():
                    nxt = content
                break
            if role == "assistant":
                break  # no intervening user turn -> no target for turn i
        if nxt is not None:
            yield i, nxt


def _render_assistant_turn(answer, prediction, fmt):
    """Render one assistant turn in the requested output format.

    MUST match what the policy is prompted to emit. If the RL parquet carries
    TAG_SYSTEM_PROMPT but this renders JSON, the auxiliary loss teacher-forces
    the model toward a format the rollout prompt does not ask for — the aux loss
    then actively fights the format cold-start, and json.dumps re-escapes the
    LaTeX in the answer, reintroducing the exact pathology the tag format exists
    to remove.
    """
    if fmt == "tag":
        return (
            f"{ANSWER_START_TAG}\n{answer}\n{ANSWER_END_TAG}\n"
            f"{PREDICTION_START_TAG}\n{prediction}\n{PREDICTION_END_TAG}"
        )
    return json.dumps({"answer": answer, "prediction": prediction}, ensure_ascii=False)


def _locate_prediction_value(text, target, cursor, fmt):
    """Character span of the prediction VALUE for the next turn at/after cursor.

    Returns (val_start, val_end, next_cursor) or None if it cannot be localized.

    Both branches SCAN FORWARD over every delimiter occurrence at/after cursor
    and accept the first whose value matches `target`, rather than trusting the
    first hit. This is required because the tag/JSON *system prompt* embeds the
    format spec verbatim — e.g. TAG_SYSTEM_PROMPT contains a literal
    "[PREDICTION_START]\n<your prediction of the user's next message>\n..." block,
    and a JSON system prompt can contain a literal '"prediction":' — so the first
    delimiter in the templated text belongs to the instructions, not an assistant
    turn. Matching on that occurrence made the builder return None for EVERY
    conversation and silently skip the aux SFT loss (observed on run 847342).

    The tag branch matches the raw (unescaped) target; the JSON branch matches
    the JSON-escaped inner text, json.dumps(target)[1:-1].
    """
    if fmt == "tag":
        search = cursor
        while True:
            midx = text.find(PREDICTION_START_TAG, search)
            if midx == -1:
                return None
            i = midx + len(PREDICTION_START_TAG)
            while i < len(text) and text[i] in " \t\r\n":
                i += 1
            if text.startswith(target, i):
                return i, i + len(target), i + len(target)
            # System-prompt spec, or an assistant that echoed the delimiter:
            # keep scanning for the real prediction slot.
            search = midx + len(PREDICTION_START_TAG)

    # `target` is rendered into `text` as a JSON-escaped string value. The
    # escaped inner text equals json.dumps(target)[1:-1].
    inner = json.dumps(target, ensure_ascii=False)[1:-1]
    search = cursor
    while True:
        midx = text.find(_PRED_KEY_MARKER, search)
        if midx == -1:
            return None
        qopen = text.find('"', midx + len(_PRED_KEY_MARKER))
        if qopen == -1:
            return None
        val_start = qopen + 1
        val_end = val_start + len(inner)
        if text[val_start:val_end] == inner:
            # past the closing quote, for the next turn's search
            return val_start, val_end, val_end + 1
        # A '"prediction":' that isn't this turn's value (e.g. in the system
        # prompt's format spec): keep scanning.
        search = midx + len(_PRED_KEY_MARKER)


def build_prediction_sft_sequence(messages, tokenizer, max_len=None, fmt="json"):
    """Build one teacher-forced next-user-turn SFT sequence for a conversation.

    fmt: "json" for the legacy {"answer","prediction"} contract, "tag" for the
    [ANSWER_START]/[PREDICTION_START] delimiters. This MUST match the system
    prompt baked into the RL parquet.

    Args:
        messages: list of {"role", "content"} (or objects with .role/.content).
            Assistant contents are expected to be the cleaned *answer* strings
            (predictions already stripped by the agent loop); user contents are
            the cleaned simulator responses.
        tokenizer: a HF *fast* tokenizer (needs return_offsets_mapping).
        max_len: optional cap on the sequence length (truncates from the right).
            Spans that fall entirely beyond the cap contribute no loss.

    Returns:
        dict with int lists {"input_ids", "position_ids", "loss_mask"} where
        loss_mask is 1 exactly on the q_{t+1} tokens spliced into each prediction
        slot, or None if the conversation has no usable (a_t, q_{t+1}) pair.
    """
    msgs = [{"role": _get_role(m), "content": _get_content(m)} for m in messages]

    targets = dict(_iter_assistant_targets(msgs))
    if not targets:
        return None

    # Rebuild assistant messages with prediction=<real next turn>, in `fmt`.
    # Turns without a target get an empty prediction (kept for context fidelity,
    # contributes no loss).
    rebuilt = []
    pred_values = []  # the spliced q strings, in sequence order (loss spans)
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            target = targets.get(i, "")
            content = _render_assistant_turn(m["content"], target, fmt)
            rebuilt.append({"role": "assistant", "content": content})
            if i in targets:
                pred_values.append(target)
        else:
            rebuilt.append(m)

    if not pred_values:
        return None

    text = tokenizer.apply_chat_template(
        rebuilt, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    loss_mask = [0] * len(input_ids)

    # Locate each prediction value's character span, scanning left-to-right.
    cursor = 0
    for target in pred_values:
        located = _locate_prediction_value(text, target, cursor, fmt)
        if located is None:
            logger.warning(
                "SFT builder: could not localize a prediction value (fmt=%s); "
                "skipping a span.", fmt
            )
            continue
        val_start, val_end, cursor = located

        for k, (a, b) in enumerate(offsets):
            if b <= a:  # special / empty-offset token
                continue
            # Include any token whose first character is inside the value span.
            # Using the token *start* (not full containment) keeps the final
            # value char when it merges with the closing quote into one token
            # (e.g. `?"`); the trailing `"}` tokens start at/after val_end and
            # stay masked out, as intended.
            if val_start <= a < val_end:
                loss_mask[k] = 1

    if max_len is not None and len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        loss_mask = loss_mask[:max_len]

    if sum(loss_mask) == 0:
        # Everything got truncated away or localization failed entirely.
        return None

    position_ids = list(range(len(input_ids)))
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
    }


def build_prediction_sft_v2_rows(
    prompt_ids, response_ids, pred_value_spans, messages, tokenizer, max_len=None
):
    """sft_v2: per-turn ON-POLICY-context SFT rows.

    build_prediction_sft_sequence (v1) builds ONE teacher-forced sequence per
    conversation, rewriting EVERY prediction slot to the real next user turn —
    so later turns condition on teacher-forced history. This variant instead
    builds ONE row per scored assistant turn t:

        context = the ACTUAL rollout tokens (prompt + response) up to the start
                  of turn t's prediction-value span — token-identical to what
                  the policy generated, including its own sampled predictions
                  e_0 .. e_{t-1} in earlier turns' slots
        target  = the real next user turn q_{t+1}, JSON-escaped and tokenized,
                  spliced in place of the sampled e_t; loss ONLY on its tokens

    The sequence ends right after the target: with a causal model nothing after
    the loss span can affect it, so no closing JSON syntax is appended.

    Args:
        prompt_ids: valid (unpadded) prompt token ids for this rollout.
        response_ids: valid (unpadded) response token ids — assistant AND user
            tokens interleaved, exactly as rolled out.
        pred_value_spans: per generated assistant turn, [start, end) of the
            prediction VALUE tokens in response space, recorded by the agent
            loop in lockstep with the predictions list; [-1, -1] marks a turn
            whose span could not be localized (skipped here).
        messages: the cleaned conversation; user turn i+1 is turn i's target.
        tokenizer: HF tokenizer.
        max_len: optional cap per row. Left-truncates (drops the OLDEST
            context) so the target span always survives; rows whose target
            alone would fill the cap are dropped.

    Returns:
        list of {"input_ids", "position_ids", "loss_mask"} dicts, one per
        usable turn (possibly empty).
    """
    user_contents = [_get_content(m) for m in messages if _get_role(m) == "user"]
    # spans[i] / predictions[i] belong to assistant turn i, which predicts the
    # user turn that follows it — user_contents[i + 1].
    targets = [extract_sim_response(c) for c in user_contents[1:]]

    prompt_ids = list(prompt_ids)
    response_ids = list(response_ids)
    n_resp = len(response_ids)

    rows = []
    for i, span in enumerate(pred_value_spans or []):
        if i >= len(targets):
            break
        try:
            s, e = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        if s < 0 or e <= s or s > n_resp:
            continue  # localization failed for this turn, or span truncated away
        target = targets[i]
        if not target or not target.strip():
            continue
        # The value is rendered inside a JSON string in the rollout, so
        # teacher-force its JSON-escaped form (matches v1's splicing).
        inner = json.dumps(target, ensure_ascii=False)[1:-1]
        tgt_ids = tokenizer(inner, add_special_tokens=False)["input_ids"]
        if not tgt_ids:
            continue

        input_ids = prompt_ids + response_ids[:s] + list(tgt_ids)
        loss_mask = [0] * (len(prompt_ids) + s) + [1] * len(tgt_ids)
        if max_len is not None and len(input_ids) > max_len:
            if len(tgt_ids) >= max_len:
                # No room for any conditioning context — unusable row.
                logger.warning(
                    "SFT v2 builder: target longer than max_len; skipping a turn."
                )
                continue
            input_ids = input_ids[-max_len:]
            loss_mask = loss_mask[-max_len:]

        rows.append({
            "input_ids": input_ids,
            "position_ids": list(range(len(input_ids))),
            "loss_mask": loss_mask,
        })
    return rows

def build_prediction_sft_v3_rows(
    prediction_context_ids,
    messages,
    tokenizer,
    max_len=None,
    targets_override=None,
    skip_terminate=True,
):
    """Build per-turn sft_v3 rows from controller-owned token snapshots.

    prediction_context_ids[t] is the complete token sequence immediately after
    the fixed PREDICTION_START prefix for assistant turn t. Storing this sequence
    during generation avoids offsets drifting when veRL re-renders chat history.
    The real next user turn and explicit end boundary are teacher-forced targets.

    targets_override (sft_v5) replaces the real next user turns with per-turn
    counterfactual reactions to that turn's DRAFT. The context is unchanged and
    already correct for this: the snapshot ends right after PREDICTION_START,
    with the draft in scope and the revision not yet written, so pairing it with
    the reaction-to-the-draft makes conditioning and target agree. Pairing it
    with the real next user turn — a reaction to the *revision* — is what made
    sft_v4's signal invalid. Entries may be None (a failed counterfactual call);
    those turns are skipped, costing one supervision slot and nothing else.
    """
    if targets_override is not None:
        targets = list(targets_override)
    else:
        user_contents = [_get_content(m) for m in messages if _get_role(m) == "user"]
        targets = [extract_sim_response(c) for c in user_contents[1:]]

    end_ids = tokenizer("\n" + PREDICTION_END_TAG, add_special_tokens=False)["input_ids"]
    if not end_ids:
        raise ValueError("sft_v3 prediction end delimiter tokenized to an empty sequence")

    rows = []
    for turn_index, raw_context in enumerate(prediction_context_ids or []):
        if turn_index >= len(targets):
            break
        try:
            context_ids = [int(token_id) for token_id in raw_context]
        except (TypeError, ValueError):
            continue
        if not context_ids:
            continue

        target = targets[turn_index]
        if not target or not target.strip():
            continue
        if skip_terminate:
            # A bare stop signal is a control token, not something to anticipate.
            if is_terminate_only_target(target):
                continue
            # Mixed turn ("Thanks! [[TERMINATE CHAT]]"): keep the utterance, drop
            # the token, so the model never learns to emit it.
            target = strip_termination_signal(target)
            if not target:
                continue
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        supervised_ids = list(target_ids) + list(end_ids)
        if not target_ids:
            continue

        input_ids = context_ids + supervised_ids
        loss_mask = [0] * len(context_ids) + [1] * len(supervised_ids)
        target_start = len(context_ids)

        if max_len is not None and len(input_ids) > max_len:
            if len(supervised_ids) >= max_len:
                logger.warning(
                    "SFT v3 builder: target plus stop boundary fills max_len; skipping a turn."
                )
                continue
            dropped = len(input_ids) - max_len
            input_ids = input_ids[dropped:]
            loss_mask = loss_mask[dropped:]
            target_start -= dropped

        rows.append(
            {
                "input_ids": input_ids,
                "position_ids": list(range(len(input_ids))),
                "loss_mask": loss_mask,
                "turn_index": turn_index,
                "target_start": target_start,
                "target_length": len(supervised_ids),
            }
        )
    return rows


def prediction_log_probs_to_reward(log_probs, loss_mask, transform="neg_nll", max_nll=20.0):
    """Reduce detached target-token log-probabilities to one turn reward."""
    log_probs = torch.as_tensor(log_probs, dtype=torch.float32)
    loss_mask = torch.as_tensor(loss_mask, dtype=torch.bool)
    if log_probs.shape != loss_mask.shape:
        raise ValueError(
            f"log_probs and loss_mask must have the same shape, got "
            f"{tuple(log_probs.shape)} and {tuple(loss_mask.shape)}"
        )
    if not bool(loss_mask.any()):
        raise ValueError("prediction reward requires at least one supervised token")
    nll = -log_probs[loss_mask].mean()
    if max_nll is not None:
        nll = nll.clamp(max=float(max_nll))
    if transform == "neg_nll":
        return float((-nll).item())
    if transform == "prob":
        return float(torch.exp(-nll).item())
    raise ValueError(f"unknown prediction reward transform: {transform!r}")
