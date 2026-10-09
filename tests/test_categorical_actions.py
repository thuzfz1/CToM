"""Behavior, probability, gradient and reduction contracts for categorical_v1."""
import asyncio
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from evaluation.eval_math import AsyncAssistantClient, run_v7_round
from ctom import controller
from tests.test_agent_loop import CLOSE, GEN, HEADER, FakeInteraction, _agent_data, _loop
from verl.utils import tensordict_utils as tu
from verl.workers.utils.categorical_actions import (
    action_batch_from_records, categorical_outputs, episode_action_mean, shifted_action_ids,
)
from verl.workers.utils.losses import ppo_loss, sft_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
from verl.workers.config.actor import ActorConfig


class EvalAssistant(AsyncAssistantClient):
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.v7_action_protocol = "categorical_v1"
        self.v7_think_max_tokens = 64
        self.v7_ask_max_tokens = 64
        self.v7_max_inner_revisions = 3
        self.pred_answer_max_tokens = 200
        self.prediction_max_tokens = 64
        self.trajectory_response_length = 6000

    def _encode_text(self, text):
        return list(map(ord, text))

    def v7_slot_ids(self):
        return list(HEADER), list(CLOSE), list(GEN)

    async def _generate_completion_phase(self, prompt, max_tokens, stop_text, allowed_token_ids=None):
        self.calls.append((max_tokens, allowed_token_ids))
        text = self.script.pop(0)
        return text, self._encode_text(text)


@pytest.mark.parametrize("script,last", [
    (["plan", "B", "Which audience?"], ""),
    (["plan", "A", "draft", "Please fix this?", "revision", "ok"], ""),
    (["plan", "C"], "prior draft"),
    (["plan", "B", "", "fallback", "ok"], ""),
])
def test_train_eval_protocol_parity(script, last):
    loop = _loop(script)
    loop.config.actor_rollout_ref.actor.v7_action_protocol = "categorical_v1"
    data = _agent_data()
    data.v7_last_shown = last
    assistant = EvalAssistant(script)
    result = asyncio.run(run_v7_round(assistant, list(data.prompt_ids), len(data.response_mask),
                                     task_type="math", last_shown=last))
    asyncio.run(loop._run_v7_round(data, {"temperature": 1, "top_p": .8, "top_k": 20}))
    assert result["prompt"] == data.prompt_ids
    assert result["response_tokens_used"] == len(data.response_mask)
    record = data.v7_rounds[0]
    for key in ("decision", "sampled_decision", "action_offset", "action_token_ids", "visible",
                "predictions", "drafts", "generated_tokens", "visible_tokens", "invalid_count"):
        assert record[key] == result["record"][key], key
    params = loop.server_manager.calls[1]["sampling_params"]
    assert params["allowed_token_ids"] == ([65, 66, 67] if last else [65, 66])
    assert params["max_tokens"] == 1 and params["top_p"] == 1 and params["top_k"] == -1
    assert params["repetition_penalty"] == 1
    if record["decision"] == controller.ASK:
        assert record["visible"] == "Which audience?"
        assert data.prediction_context_ids == data.v7_slots == record["predictions"] == []
        assert "prediction_valid" not in data.phase_validity[0]
        data.interaction = FakeInteraction([], [])
        asyncio.run(loop._collect_v7_counterfactual_targets(data))
        assert not data.interaction.cf_seen
    else:
        assert all(slot["kind"] == "draft" for slot in data.v7_slots)
        assert len(data.prediction_context_ids) == len(record["drafts"])


def test_illegal_first_commit_fails_and_prompt_is_idempotent():
    with pytest.raises(ValueError):
        controller.parse_action_token([67], [65, 66, 67], False)
    with pytest.raises(ValueError):
        controller.action_token_ids(lambda text: [1, 2])
    original = [{"role": "system", "content": "Original task instructions"}]
    result = controller.action_prompt_messages(original, "categorical_v1")
    assert original[0]["content"] == "Original task instructions"
    assert controller.action_prompt_messages(result, "categorical_v1") == result
    assert "No anticipated-user slot follows ASK" in result[0]["content"]


def test_categorical_probabilities_gradients_and_language_parity():
    # Huge ordinary-vocabulary logit must not affect a masked decision.
    logits = torch.tensor([[.2, 2., -1., 100.], [.1, .2, .3, .4], [3., -2., 1., 50.]], requires_grad=True)
    labels = torch.tensor([1, 3, 2])
    ids = torch.tensor([[0, 1, -1], [-1, -1, -1], [0, 1, 2]])
    full_lp = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    lp, out = categorical_outputs(logits, labels, ids, full_lp)
    expected0 = logits[0, :2].log_softmax(-1)
    expected2 = logits[2, :3].log_softmax(-1)
    torch.testing.assert_close(lp, torch.stack((expected0[1], full_lp[1], expected2[2])))
    torch.testing.assert_close((lp - lp.detach()).exp(), torch.ones(3))
    entropy = out["action_entropy"].sum()
    grad = torch.autograd.grad(-entropy, logits)[0]
    assert not grad[1].any() and not grad[:, 3].any() and grad[0, 2] == 0
    assert grad[0, 1] > 0 and grad[0, 0] < 0  # GD decreases dominant action logit.
    assert torch.isfinite(grad).all()
    assert out["action_p_commit"][0] == 0
    torch.testing.assert_close(out["action_p_ask"][0], expected0[1].exp())


def _padded_batch():
    return TensorDict({
        "prompts": torch.tensor([[0, 5, 6], [4, 5, 6]]),
        "responses": torch.tensor([[1, 7, 0, 0], [7, 2, 7, 1]]),
        "input_ids": torch.tensor([[0, 5, 6, 1, 7, 0, 0], [4, 5, 6, 7, 2, 7, 1]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]]),
        "response_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
        "position_ids": torch.tensor([[0, 0, 1, 2, 3, 0, 0], [0, 1, 2, 3, 4, 5, 6]]),
    }, batch_size=[2])


def test_response_offsets_survive_padding_and_causal_shift():
    batch = _padded_batch()
    record = lambda off, ids: {"action_protocol": "categorical_v1", "action_offset": off, "action_token_ids": ids}
    records = [[record(0, [0, 1, -1])], [record(1, [0, 1, 2]), record(3, [0, 1, 2])]]
    batch["action_token_ids"] = action_batch_from_records(records, batch["responses"], batch["response_mask"])
    converted = left_right_2_no_padding(batch)
    shifted = converted["action_token_ids_shifted"]
    assert shifted[0].tolist() == [[-1]*3, [0, 1, -1], [-1]*3, [-1]*3]
    assert shifted[1][2].tolist() == [-1]*3
    assert shifted[1][3].tolist() == [0, 1, 2]
    assert shifted[1][5].tolist() == [0, 1, 2]
    assert shifted[1][-1].tolist() == [-1]*3  # No cross-episode wrap.
    entropy = torch.nested.as_nested_tensor([torch.tensor([0., .6, 0., 0.]),
                                             torch.tensor([0., 0., 0., .8, 0., .9, 0.])], layout=torch.jagged)
    torch.testing.assert_close(no_padding_2_padding(entropy, converted), torch.tensor([[.6, 0., 0., 0.], [0., .8, 0., .9]]))
    # Exercise the actual worker minibatch selector with the new 3D nested data.
    reordered = tu.index_select_tensor_dict(converted, [1, 0])
    assert reordered["action_token_ids_shifted"][0].tolist() == shifted[1].tolist()
    assert reordered["action_token_ids"][0].tolist() == converted["action_token_ids"][1].tolist()


def test_episode_entropy_is_independent_of_length_partition_and_dp():
    values = torch.tensor([[.4, .0, 0, 0], [.2, .8, 0, 0], [0., 0, 0, 0]])
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
    whole = episode_action_mean(values, mask, global_batch_size=3, dp_size=1)
    assert whole.item() == pytest.approx(.3)
    micro = sum(episode_action_mean(v[None], m[None], global_batch_size=3, dp_size=1)
                for v, m in zip(values, mask))
    torch.testing.assert_close(whole, micro)
    rank1 = episode_action_mean(values[:1], mask[:1], global_batch_size=3, dp_size=2)
    rank2 = episode_action_mean(values[1:], mask[1:], global_batch_size=3, dp_size=2)
    torch.testing.assert_close(whole, (rank1 + rank2) / 2)


def test_draft_target_mean_in_worker_and_empty_loss_is_graph_connected():
    from verl.workers.engine_workers import TrainingWorker

    worker = object.__new__(TrainingWorker)
    worker.model_config = {"use_remove_padding": True}
    worker.engine_config = SimpleNamespace(use_dynamic_bsz=True, max_token_len_per_gpu=100,
                                           micro_batch_size_per_gpu=1, use_fused_kernels=False)
    data = TensorDict({
        "sft_input_ids": torch.nested.as_nested_tensor([torch.arange(7), torch.tensor([0])], layout=torch.jagged),
        "sft_position_ids": torch.nested.as_nested_tensor([torch.tensor([0, 1, 2, 0, 1, 2, 3]), torch.tensor([0])], layout=torch.jagged),
        "sft_loss_mask": torch.nested.as_nested_tensor([torch.tensor([0, 0, 1, 0, 0, 1, 1]), torch.tensor([0])], layout=torch.jagged),
        "sft_seq_lens": torch.nested.as_nested_tensor([torch.tensor([3, 4]), torch.tensor([1])], layout=torch.jagged),
    }, batch_size=[2])
    tu.assign_non_tensor(data, sft_target_mean=True)
    aux = worker._build_sft_td(data)
    assert aux["loss_mask"].values().tolist() == [0, 0, 1, 0, 0, .5, .5, 0]
    tu.assign_non_tensor(aux, batch_num_tokens=2, dp_size=1)
    lp_values = torch.tensor([0., -2., 0., 0., -4., -8., 0., 0.], requires_grad=True)
    lp = torch.nested.nested_tensor_from_jagged(lp_values, aux["input_ids"].offsets())
    loss, _ = sft_loss(None, {"log_probs": lp}, aux)
    assert loss.item() == pytest.approx(4.)  # mean(2, mean(4,8)), not (2+4+8)/3.
    loss.backward()
    assert lp_values.grad.tolist() == [0., -.5, 0., 0., -.25, -.25, 0., 0.]
    empty = TensorDict({key: torch.nested.as_nested_tensor([aux[key].unbind()[-1]], layout=torch.jagged)
                        for key in ("input_ids", "position_ids", "loss_mask")}, batch_size=[1])
    tu.assign_non_tensor(empty, batch_num_tokens=0, dp_size=1)
    empty_values = torch.tensor([-.2], requires_grad=True)
    empty_lp = torch.nested.nested_tensor_from_jagged(empty_values, empty["input_ids"].offsets())
    zero, _ = sft_loss(None, {"log_probs": empty_lp}, empty)
    zero.backward()
    assert zero == 0 and empty_values.grad.item() == 0


def test_ppo_decision_entropy_is_separate_and_lambda_zero_keeps_same_policy_loss():
    batch = _padded_batch()
    ids = torch.full((2, 4, 3), -1)
    ids[0, 0] = torch.tensor([0, 1, -1])
    ids[1, 1] = torch.tensor([0, 1, 2])
    batch["action_token_ids"] = ids
    batch["old_log_probs"] = torch.zeros(2, 4)
    batch["advantages"] = torch.zeros(2, 4)
    batch = left_right_2_no_padding(batch)
    tu.assign_non_tensor(batch, batch_num_tokens=6, global_batch_size=2, dp_size=1)
    values = torch.tensor([0., .4, 0., 0., 0., 0., 0., .8, 0., 0., 0.], requires_grad=True)
    ent = torch.nested.nested_tensor_from_jagged(values, batch["input_ids"].offsets())
    lp = torch.nested.nested_tensor_from_jagged(torch.zeros_like(values, requires_grad=True), batch["input_ids"].offsets())
    out = {"log_probs": lp, **{key: ent for key in ("action_entropy", "action_p_ask", "action_p_draft", "action_p_commit")}}
    def cfg(coef):
        return ActorConfig(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True,
                           v7_action_protocol="categorical_v1", decision_entropy_coef=coef)
    zero, _ = ppo_loss(cfg(0), out, batch)
    loss, metrics = ppo_loss(cfg(.1), out, batch)
    assert zero.item() == 0
    assert loss.item() == pytest.approx(-.06)
    loss.backward()
    assert values.grad[1].item() == pytest.approx(-.05)
    assert values.grad[7].item() == pytest.approx(-.05)
    assert (values.grad != 0).sum() == 2
    assert "actor/pre_draft_action_p_ask_sum_per_episode" in metrics


def test_fsdp_forward_outputs_use_categorical_for_old_current_and_reference(monkeypatch):
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    # The installed FlashAttention CE kernel is GPU-only. Keep the actual
    # engine adapter under test, replacing only its ordinary vocabulary CE.
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits",
                        lambda logits, labels, **kw: logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1))

    data = _padded_batch()
    ids = torch.full((2, 4, 3), -1)
    ids[0, 0] = torch.tensor([0, 1, -1])
    ids[1, 1] = torch.tensor([0, 1, 2])
    data["action_token_ids"] = ids
    data = left_right_2_no_padding(data)
    tu.assign_non_tensor(data, temperature=.7, use_remove_padding=True, use_fused_kernels=False,
                         calculate_entropy=False)
    engine = SimpleNamespace(use_ulysses_sp=False)
    model_inputs, args = FSDPEngineWithLMHead.prepare_model_inputs(engine, data)
    assert args["action_token_ids_shifted"].shape == (11, 3)
    raw_logits = torch.randn(1, 11, 10, requires_grad=True)
    outputs = []
    for _ in ("current", "old", "reference"):
        outputs.append(FSDPEngineWithLMHead.prepare_model_outputs(
            engine, SimpleNamespace(logits=raw_logits.clone()), args, data, None))
    logits = raw_logits[0] / .7
    expected = logits[1, :2].log_softmax(-1)[1]
    for out in outputs:
        torch.testing.assert_close(out["log_probs"].values()[1], expected)
        torch.testing.assert_close(out["action_p_ask"].values()[1], expected.exp())
        assert out["action_p_commit"].values()[1] == 0
    entropy = outputs[0]["action_entropy"].values().sum()
    entropy.backward()
    assert (raw_logits.grad.abs().sum(-1) > 0).sum() == 2
    assert not raw_logits.grad[..., 3:].any()


def test_ask_then_draft_collects_only_one_correctly_aligned_target():
    loop = _loop(["plan", "B", "Which audience?", "plan", "A", "draft", "Please revise?"] , max_inner=0)
    loop.config.actor_rollout_ref.actor.v7_action_protocol = "categorical_v1"
    interaction = FakeInteraction([], ["Make this shorter."])
    data = _agent_data(interaction=interaction)
    asyncio.run(loop._run_v7_round(data, {}))
    data.messages.append({"role": "user", "content": "Experts."})
    asyncio.run(loop._run_v7_round(data, {}))
    asyncio.run(loop._collect_v7_counterfactual_targets(data))
    assert len(data.prediction_context_ids) == len(data.v7_slots) == len(data.counterfactual_targets) == 1
    assert data.v7_slots[0]["slot"] == 0 and data.v7_slots[0]["kind"] == "draft"
    assert data.counterfactual_targets[0] == "Make this shorter."
    assert len(interaction.cf_seen) == 1
    assert interaction.cf_seen[0][-1]["content"] == "draft"
    assert any(m["content"] == "Which audience?" for m in interaction.cf_seen[0])


def test_empty_action_rows_have_finite_zero_entropy_and_no_new_gradient():
    logits = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([1, 2, 3, 4])
    full = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    lp, out = categorical_outputs(logits, labels, torch.full((4, 3), -1), full)
    torch.testing.assert_close(lp, full)
    entropy = out["action_entropy"].sum()
    assert entropy == 0 and torch.isfinite(entropy)
    entropy.backward()
    assert not logits.grad.any()


def test_tokenizer_and_installed_vllm_accept_the_constrained_alphabet():
    from tests.test_eval_parity import _tokenizer
    from vllm import SamplingParams
    tokenizer = _tokenizer()
    alphabet = controller.action_token_ids(lambda text: tokenizer.encode(text, add_special_tokens=False))
    assert [tokenizer.decode([i]) for i in alphabet] == list(controller.ACTION_LABELS)
    params = SamplingParams(max_tokens=1, **controller.categorical_sampling_params({"temperature": 1.}, alphabet[:2]))
    assert params.allowed_token_ids == alphabet[:2] and params.max_tokens == 1


def test_config_rejects_entropy_without_categorical_protocol():
    with pytest.raises(ValueError, match="requires v7_action_protocol"):
        ActorConfig(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True, decision_entropy_coef=.001)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ActorConfig(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True, decision_entropy_coef=float("nan"))
