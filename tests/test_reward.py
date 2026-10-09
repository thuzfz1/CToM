#!/usr/bin/env python3
# Offline smoke test for turn-wise reward (no GPU / SLURM / Together API needed).
#
# Validates the highest-risk part of the turn-wise reward feature:
#   - per-turn prediction (F1) and per-turn effort decomposition
#   - dense rm_scores placement on each turn's last token
#   - task credited to the terminal turn only
#   - SFT-mode renormalization
#   - reward_extra_info schema is identical across turn / conversation / empty
#     paths (required so verl's _postprocess can concat the batch)
#
# Run:
#   source .venv/bin/activate
#   PYTHONPATH="$PWD:$PWD/verl" python tests/test_reward.py
import asyncio

import pytest
import torch

from ctom.reward import (
    PredictionMode,
    _compute_f1_prediction_reward_per_pair,
    compute_effort_reward_per_turn,
    compute_task_reward,
    CToMRewardManager,
)

# A synthetic 2-assistant-turn conversation: q0,a0,q1,a1,q2.
MSGS = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "q0 what is 2+2"},
    {"role": "assistant", "content": "the answer is 4"},
    {"role": "user", "content": "q1 ok why is that"},
    {"role": "assistant", "content": "because addition \\boxed{4}"},
    {"role": "user", "content": "q2 great thanks [[TERMINATE CHAT]]"},
]
PREDS = ["ok why is that", "great thanks"]   # e0 predicts q1, e1 predicts q2
TURN_END = [10, 25]                           # last token of a0, a1 (response space)
RESPONSE_LEN = 40
GT = "\\boxed{4}"


def _mgr(prediction_mode=PredictionMode.F1):
    m = CToMRewardManager.__new__(CToMRewardManager)
    m.reward_weights = {"task": 0.5, "prediction": 0.3, "effort": 0.2}
    m.max_turns = 4
    m.max_user_tokens = 256
    m.alpha = 0.5
    m.max_assistant_tokens = 0
    m.beta = 0.0
    m.count_revision_tokens = False  # unsent sft_v7 drafts unbilled; the 2026-09-13
                                     # default. tests/test_reward.py
                                     # covers the charged path.
    m.effort_token_units = "words"   # historical unit; tests/test_reward.py covers "tokens"
    m.tokenizer = None               # unused under "words"; _effort_length falls back
    m.prediction_mode = prediction_mode
    m.llm_kwargs = {}
    m.eval_logger = None
    m.task_reward_mode = "math"
    m.length_target_words = 3000
    m.invalid_phase_penalty = -1.0
    m.reward_placement = "turn"
    m.effort_mode = "bonus"   # historical default; see tests/test_reward.py
    return m


def test_per_turn_components():
    f1 = _compute_f1_prediction_reward_per_pair(MSGS, PREDS)
    eff = compute_effort_reward_per_turn(MSGS, 4, 256, 0.5)
    task = compute_task_reward(MSGS, GT, "solving")
    assert len(f1) == 2 and len(eff) == 2, (f1, eff)
    assert task == 1.0, task
    # effort decreases with turn index (later turns worth less)
    assert eff[0] > eff[1], eff
    print(f"[ok] per-turn: f1={[round(x,3) for x in f1]} effort={[round(x,3) for x in eff]} task={task}")


def test_dense_placement():
    dense = CToMRewardManager._dense_from_turns([0.3, 0.8], [10, 25], 40)
    nz = (dense != 0).nonzero().flatten().tolist()
    assert nz == [10, 25] and abs(dense[10] - 0.3) < 1e-6 and abs(dense[25] - 0.8) < 1e-6
    # out-of-range index (truncated away) is dropped, not crashed
    d2 = CToMRewardManager._dense_from_turns([0.5, 0.9], [5, 100], 40)
    assert (d2 != 0).nonzero().flatten().tolist() == [5]
    # scalar placement = last valid token
    ds = CToMRewardManager._dense_from_scalar(0.7, valid_len=12, total_len=40)
    assert (ds != 0).nonzero().flatten().tolist() == [11]
    print("[ok] dense placement (turn-end tokens, oob-drop, scalar-last-token)")


def test_turnwise_assembly():
    m = _mgr(PredictionMode.F1)
    out = asyncio.run(m._run_single_turnwise(
        message_lst=MSGS, predictions_lst=PREDS, ground_truth=GT, goal="solving",
        data_source="math", interaction_kwargs={}, tool_extra_fields={},
        turn_end_indices=TURN_END, response_len=RESPONSE_LEN,
    ))
    dense = out["token_level_scores"]
    f1 = _compute_f1_prediction_reward_per_pair(MSGS, PREDS)
    eff = compute_effort_reward_per_turn(MSGS, 4, 256, 0.5)
    w = m.reward_weights
    r0 = w["prediction"] * f1[0] + w["effort"] * eff[0]
    r1 = w["prediction"] * f1[1] + w["effort"] * eff[1] + w["task"] * 1.0  # task terminal
    assert abs(dense[10] - r0) < 1e-5, (dense[10].item(), r0)
    assert abs(dense[25] - r1) < 1e-5, (dense[25].item(), r1)
    assert abs(out["reward_score"].item() - (r0 + r1)) < 1e-5
    info = out["reward_extra_info"]
    assert info["reward/num_scored_turns"] == 2.0
    assert abs(info["reward/turn_reward_mean"] - (r0 + r1) / 2) < 1e-5
    print(f"[ok] turn-wise assembly: r0={r0:.4f} r1={r1:.4f} total={r0+r1:.4f}")


def test_sft_renormalization():
    m = _mgr(PredictionMode.SFT)
    out = asyncio.run(m._run_single_turnwise(
        message_lst=MSGS, predictions_lst=PREDS, ground_truth=GT, goal="solving",
        data_source="math", interaction_kwargs={}, tool_extra_fields={},
        turn_end_indices=TURN_END, response_len=RESPONSE_LEN,
    ))
    dense = out["token_level_scores"]
    eff = compute_effort_reward_per_turn(MSGS, 4, 256, 0.5)
    wn = m.reward_weights["task"] + m.reward_weights["effort"]
    r0 = (m.reward_weights["effort"] * eff[0]) / wn                       # no pred in sft
    r1 = (m.reward_weights["effort"] * eff[1] + m.reward_weights["task"] * 1.0) / wn
    assert out["reward_extra_info"]["reward/prediction"] == 0.0
    assert abs(dense[10] - r0) < 1e-5 and abs(dense[25] - r1) < 1e-5
    print(f"[ok] sft renormalization: r0={r0:.4f} r1={r1:.4f}")


def test_sft_v3_invalid_phase_gets_only_penalty():
    m = _mgr(PredictionMode.SFT_V3)
    phase_validity = [
        {"answer_valid": True, "prediction_valid": True},
        {"answer_valid": True, "prediction_valid": False},
    ]
    out = asyncio.run(m._run_single_turnwise(
        message_lst=MSGS, predictions_lst=PREDS, ground_truth=GT, goal="solving",
        data_source="math", interaction_kwargs={},
        tool_extra_fields={"phase_validity": phase_validity},
        turn_end_indices=TURN_END, response_len=RESPONSE_LEN,
    ))
    dense = out["token_level_scores"]
    effort = compute_effort_reward_per_turn(MSGS, 4, 256, 0.5)
    weight_norm = m.reward_weights["task"] + m.reward_weights["effort"]
    assert dense[10] == pytest.approx(
        m.reward_weights["effort"] * effort[0] / weight_norm
    )
    assert dense[25] == pytest.approx(m.invalid_phase_penalty)



def test_schema_uniform():
    # Turn-wise and conversation reward_extra_info must share the same keys so
    # verl's _postprocess can build a uniform batch.
    #
    # Compared against REWARD_EXTRA_INFO_SCHEMA, not a literal set: this test
    # used to hard-code the nine keys it knew about, which is why it stayed green
    # while a composition branch quietly added a key to ONE path
    # and killed runs 948583/948961 with KeyError. Asserting against the single
    # source means a new key must be declared for every path or nothing passes.
    from ctom.reward import REWARD_EXTRA_INFO_SCHEMA

    expected = set(REWARD_EXTRA_INFO_SCHEMA)
    m = _mgr(PredictionMode.F1)
    out = asyncio.run(m._run_single_turnwise(
        message_lst=MSGS, predictions_lst=PREDS, ground_truth=GT, goal="solving",
        data_source="math", interaction_kwargs={}, tool_extra_fields={},
        turn_end_indices=TURN_END, response_len=RESPONSE_LEN,
    ))
    assert set(out["reward_extra_info"]) == expected, set(out["reward_extra_info"]) ^ expected
    print(f"[ok] reward_extra_info schema uniform ({len(expected)} keys)")


if __name__ == "__main__":
    test_per_turn_components()
    test_dense_placement()
    test_turnwise_assembly()
    test_sft_renormalization()
    test_schema_uniform()
    print("\nALL TURN-WISE REWARD SMOKE CHECKS PASSED")
