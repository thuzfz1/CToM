"""Regression checks for the September 5 v7 eval reward recipe."""
import pytest

from evaluation import eval_math


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def test_token_effort_prices_shown_answer_and_excludes_q0():
    messages = [
        {"role": "user", "content": "a long fixed question"},
        {"role": "assistant", "content": "abcd"},
        {"role": "user", "content": "xy"},
    ]
    scores = eval_math.compute_effort_reward_per_turn(
        messages, 4, 10, 0.5, 10, 0.4,
        tokenizer=CharacterTokenizer(), token_units="tokens",
    )
    assert scores == pytest.approx([0.5 * 0.75 + 0.1 * 0.8 + 0.4 * 0.6])


@pytest.mark.parametrize("valid,task,effort,expected", [
    (True, 0.8, 0.2, 0.74),
    (False, 0.8, 0.2, -0.2),
    (True, 1.0, 0.0, 1.0),
    (False, 1.0, 0.0, -0.2),
])
def test_v7_cost_and_terminal_validity(monkeypatch, valid, task, effort, expected):
    monkeypatch.setattr(eval_math, "compute_task_reward", lambda *a: 1.0)
    monkeypatch.setattr(eval_math, "compute_f1_prediction_reward", lambda *a: 0.0)
    monkeypatch.setattr(eval_math, "compute_effort_reward_per_turn", lambda *a, **kw: [0.7])
    result = eval_math._make_result(
        dict(goal="solving", ground_truth="4", index=0, single_turn_prompt="q", expertise="expert"),
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "4"}], [],
        phase_validity=[dict(think_valid=True, decision_valid=True, draft_valid=valid, prediction_valid=True)],
        sft_v3_mode=True, effort_mode="cost", reward_w_task=task,
        reward_w_effort=effort, invalid_phase_penalty=-0.2,
    )
    assert result["r_training_aligned"] == pytest.approx(expected)
    assert result["r_effort_sum"] == pytest.approx(-0.3 * effort if valid else 0.0)
