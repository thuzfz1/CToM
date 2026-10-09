"""Unit tests for the sft_v7 agent loop (the paper's method section).

What breaks silently and only shows up as a wasted run:
1. An anticipated (policy-written) user turn or an internal draft leaking to
   the simulator.
2. The verdict not gating the inner loop (every draft shown), or the loop not
   stopping at the cap.
3. Prediction slots / contexts / targets drifting out of lockstep.
4. A terminate-only counterfactual target being dropped instead of substituted.
5. The round leaving the token stream mid-user-turn (a dangling generation
   prompt before the real reply).

Run:  RAYON_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_agent_loop.py -q
"""
import asyncio
import os
import re
from types import SimpleNamespace

from omegaconf import OmegaConf

from ctom import controller
from ctom.agent_loop import AgentState, CToMAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentData


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(t) for t in token_ids if t != 0)


class FakeServer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("FakeServer ran out of scripted outputs")
        text = self.outputs.pop(0)
        ids = [ord(c) for c in text]
        return SimpleNamespace(token_ids=ids, log_probs=[-0.1] * len(ids), num_preempted=0, extra_fields={})


class FakeInteraction:
    def __init__(self, replies, cf_replies=()):
        self.replies = list(replies)
        self.cf_replies = list(cf_replies)
        self.seen = []
        self.cf_seen = []

    async def generate_response(self, request_id, messages, counterfactual=False, **kwargs):
        if counterfactual:
            self.cf_seen.append([dict(m) for m in messages])
            return False, self.cf_replies.pop(0), 0.0, {}
        self.seen.append([dict(m) for m in messages])
        return False, self.replies.pop(0), 0.0, {}


HEADER = [ord(c) for c in "<u>[ANT]\n"]
CLOSE = [ord(c) for c in "</u>"]
GEN = [ord(c) for c in "<a>"]


def _loop(outputs, *, max_inner=3, response_length=6000, draft_cap=200, max_user_turns=3):
    loop = object.__new__(CToMAgentLoop)
    loop.config = OmegaConf.create(
        {
            "reward": {"reward_kwargs": {"prediction_mode": "sft_v7"}},
            "actor_rollout_ref": {
                "actor": {
                    "bdr_draft_max_tokens": draft_cap,
                    "pred_prediction_max_tokens": 64,
                    "v7_think_max_tokens": 64,
                    "v7_ask_max_tokens": 64,
                    "v7_max_inner_revisions": max_inner,
                    "v7_verdict_mode": "parse",
                },
                "rollout": {"max_model_len": 0},
            },
        }
    )
    loop.tokenizer = CharTokenizer()
    loop.server_manager = FakeServer(outputs)
    loop.response_length = response_length
    loop.max_assistant_turns = 0
    loop.max_user_turns = max_user_turns

    async def _slot_ids():
        return list(HEADER), list(CLOSE), list(GEN)

    loop._v7_user_slot_ids = _slot_ids

    async def _tmpl(messages, remove_system_prompt=False, **kw):
        assert remove_system_prompt
        return [ord(c) for c in f"<u>{messages[-1]['content']}</u><a>"]

    loop.apply_chat_template = _tmpl
    return loop


def _agent_data(task_type="math", interaction=None):
    data = AgentData(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q0"}],
        image_data=None, video_data=None, metrics={}, request_id="r",
        tools_kwargs={}, interaction=interaction, interaction_kwargs={"task_type": task_type},
    )
    data.prompt_ids = [1, 2, 3]
    data.response_mask = [0, 0]
    data.response_logprobs = [0.0, 0.0]
    data.user_turns = 1
    data.turn_end_indices = []
    data.phase_validity = []
    data.prediction_context_ids = []
    data.pred_value_spans = []
    data.counterfactual_targets = []
    data.draft_texts = []
    data.v7_rounds = []
    data.v7_slots = []
    data.v7_hidden = set()
    data.v7_visible = {}
    data.v7_last_shown = ""
    data.turn_scores = []
    return data


def _text(loop, data):
    return loop.tokenizer.decode(data.prompt_ids)


def test_generated_tokens_include_hidden_drafts_but_not_controller_tokens():
    script = ["plan", "DRAFT", "discarded draft", "Please recheck?", "shown draft", "thanks"]
    loop = _loop(script)
    data = _agent_data()
    asyncio.run(loop._run_v7_round(data, {}))
    assert data.v7_rounds[-1]["generated_tokens"] == sum(map(len, script))
    assert data.v7_rounds[-1]["visible"] == "shown draft"
    assert len(data.response_mask) > data.v7_rounds[-1]["generated_tokens"]


# ── 1. draft -> unsatisfied -> new assistant turn -> satisfied -> shown ──────
def test_inner_revision_gated_by_parsed_verdict():
    loop = _loop(["plan", "DRAFT", "draft A", "Hmm, step 2 looks wrong?", "draft B", "Thanks, that makes sense."])
    data = _agent_data()
    state = asyncio.run(loop._run_v7_round(data, {"temperature": 1.0}))
    assert state == AgentState.INTERACTING
    rec = data.v7_rounds[0]
    assert rec["decision"] == controller.DRAFT
    assert rec["verdicts"] == [controller.UNSATISFIED, controller.SATISFIED]
    assert rec["inner_iters"] == 1 and rec["visible"] == "draft B"
    assert data.metrics["v7_inner_iters"] == 1 and data.metrics["v7_draft_changed"] == 1
    # lockstep: two prediction slots, two contexts, two spans
    assert len(data.prediction_context_ids) == 2 == len(data.pred_value_spans) == len(data.v7_slots)
    # simulator view: only the shown draft, no anticipated turn, no internal draft
    view = loop._v7_sim_view(data)
    assert [m["content"] for m in view] == ["sys", "q0", "draft B"]
    assert all(controller.ANTICIPATED_MARKER not in m["content"] for m in view)
    # token stream: draft A -> anticipated (closed) -> NEW assistant turn -> draft B -> anticipated closed, no dangling gen prompt
    t = _text(loop, data)
    assert t.count("<u>[ANT]\n") == 2 and t.count("</u><a>[DRAFT_START]") == 1
    assert t.endswith("[PREDICTION_END]</u>"), t[-40:]
    assert len(data.turn_end_indices) == 1 and len(data.phase_validity) == 1
    assert data.phase_validity[0]["invalid_count"] == 0
    # the anticipated tokens carry policy gradient, the header/close do not
    slot_start, slot_end = data.pred_value_spans[1]
    assert all(data.response_mask[k] == 1 for k in range(slot_start, slot_end))


def test_inner_cap_shows_the_last_draft():
    loop = _loop(["plan", "DRAFT", "d0", "I'm still not sure.", "d1", "Can you double check?"], max_inner=1)
    data = _agent_data()
    asyncio.run(loop._run_v7_round(data, {}))
    rec = data.v7_rounds[0]
    assert rec["verdicts"] == [controller.UNSATISFIED, controller.UNSATISFIED]
    assert rec["visible"] == "d1" and data.metrics["v7_inner_cap_hit"] == 1
    assert [m["content"] for m in loop._v7_sim_view(data)] == ["sys", "q0", "d1"]


# ── 2. ASK: question + the prediction as the assumption ──────────────────────
def test_ask_shows_prediction_as_assumption():
    loop = _loop(["plan", "ASK", "Is n even?", "Yes, n is even."])
    data = _agent_data()
    state = asyncio.run(loop._run_v7_round(data, {}))
    assert state == AgentState.INTERACTING
    view = loop._v7_sim_view(data)
    assert view[-1]["role"] == "assistant"
    assert view[-1]["content"].startswith("Is n even?")
    assert 'I expect the answer is: "Yes, n is even."' in view[-1]["content"]
    assert data.v7_slots[0]["kind"] == "ask" and data.v7_slots[0]["text"] == "Is n even?"
    assert data.v7_slots[0]["history"][-1]["content"] == "q0"
    assert data.metrics["v7_decision_ask"] == 1


# ── 3. COMMIT ────────────────────────────────────────────────────────────────
def test_commit_without_shown_draft_is_coerced_to_draft():
    loop = _loop(["plan", "COMMIT", "draft X", "Looks good, thanks."])
    data = _agent_data()
    asyncio.run(loop._run_v7_round(data, {}))
    assert data.v7_rounds[0]["decision"] == controller.DRAFT
    assert data.v7_rounds[0]["coerce_reason"] == "commit_without_draft"
    assert data.metrics["v7_commit_without_draft"] == 1


def test_commit_after_a_shown_draft_ends_with_that_draft():
    loop = _loop(["plan", "COMMIT"])
    data = _agent_data()
    data.v7_last_shown = "the answer is \\boxed{4}"
    state = asyncio.run(loop._run_v7_round(data, {}))
    assert state == AgentState.TERMINATED
    assert data.v7_rounds[0]["visible"] == "the answer is \\boxed{4}"
    assert loop._v7_sim_view(data)[-1]["content"] == "the answer is \\boxed{4}"
    assert not loop.server_manager.outputs  # nothing else generated


def test_writing_commit_prepends_the_submit_marker():
    loop = _loop(["plan", "COMMIT"])
    data = _agent_data(task_type="writing")
    data.v7_last_shown = "The document."
    asyncio.run(loop._run_v7_round(data, {}))
    assert loop._v7_sim_view(data)[-1]["content"].startswith("I WANT TO ANSWER:")


# ── 4. invalid decision, empty ask ──────────────────────────────────────────
def test_unparseable_decision_is_invalid_and_drafts():
    loop = _loop(["plan", "maybe", "d", "ok that makes sense"])
    data = _agent_data()
    asyncio.run(loop._run_v7_round(data, {}))
    assert data.v7_rounds[0]["decision"] == controller.DRAFT
    assert data.metrics["v7_decision_invalid"] == 1
    assert data.phase_validity[0]["decision_valid"] is False
    assert data.phase_validity[0]["invalid_count"] == 1


def test_empty_ask_falls_back_to_a_draft():
    loop = _loop(["plan", "ASK", "", "d", "great, thanks"])
    data = _agent_data()
    asyncio.run(loop._run_v7_round(data, {}))
    assert data.v7_rounds[0]["decision"] == controller.DRAFT and data.metrics["v7_empty_ask"] == 1
    assert data.v7_rounds[0]["visible"] == "d"


# ── 5. simulator never sees hidden turns; real reply appended to both views ──
def test_interacting_state_passes_the_sim_view_and_appends_the_real_reply():
    sim = FakeInteraction(replies=["Are you sure about step 2?"])
    loop = _loop(["plan", "DRAFT", "d0", "Hmm, not sure?", "d1", "makes sense"])
    data = _agent_data(interaction=sim)
    asyncio.run(loop._run_v7_round(data, {}))
    state = asyncio.run(loop._handle_interacting_state(data))
    assert state == AgentState.GENERATING
    assert [m["content"] for m in sim.seen[0]] == ["sys", "q0", "d1"]
    assert data.messages[-1] == {"role": "user", "content": "Are you sure about step 2?"}
    assert loop._v7_sim_view(data)[-1]["content"] == "Are you sure about step 2?"
    assert data.user_turns == 2
    assert _text(loop, data).endswith("<u>Are you sure about step 2?</u><a>")


# ── 6. counterfactual labels: per slot, bare question for ASK, terminate substituted ──
def test_counterfactual_targets_per_slot_with_substitution_and_gap_cells():
    sim = FakeInteraction(replies=[], cf_replies=["Can you recheck the sum?", "", "ok"])
    loop = _loop(["plan", "DRAFT", "d0", "I'm still not sure?", "d1", "great", "plan2", "ASK", "Is n even?", "yes"])
    data = _agent_data(interaction=sim)
    asyncio.run(loop._run_v7_round(data, {}))
    data.messages.append({"role": "user", "content": "real reply"})
    data.user_turns += 1
    asyncio.run(loop._run_v7_round(data, {}))
    asyncio.run(loop._collect_v7_counterfactual_targets(data))
    # The substituted slot is computed for telemetry but carries NO aux target:
    # that constant was the string the verdict parser reads as SATISFIED, so
    # supervising on it taught the policy its own exit password (2026-09-06).
    assert data.counterfactual_targets == ["Can you recheck the sum?", None, "ok"]
    assert data.metrics["v7_cf_dropped_satisfied"] == 1
    # histories: draft slots replay the sim view + the draft; the ask slot the bare question
    assert sim.cf_seen[0][-1]["content"] == "d0" and sim.cf_seen[1][-1]["content"] == "d1"
    assert sim.cf_seen[2][-1]["content"] == "Is n even?"
    assert all(controller.ANTICIPATED_MARKER not in m["content"] for h in sim.cf_seen for m in h)
    assert data.metrics["v7_cf_ok"] == 3 and data.metrics.get("v7_cf_missing", 0) == 0
    assert data.metrics["v7_cf_terminate_substituted"] == 1   # the "" reply on a correct draft
    # slot 0: model UNSAT, cf pushback -> tp; slot 1: model SAT, cf accept -> tn
    assert data.metrics["v7_gap_tp"] == 1 and data.metrics["v7_gap_tn"] == 1
    assert data.metrics["v7_ask_confirmed"] == 1


# ── 7. the model closing the user turn itself ────────────────────────────────
def test_generated_end_of_turn_token_is_popped_once():
    loop = _loop(["plan", "DRAFT", "d", "makes sense\x00"])
    data = _agent_data()
    asyncio.run(loop._run_v7_round(data, {}))
    assert data.metrics["v7_prediction_eos_popped"] == 1
    t = _text(loop, data)
    assert t.endswith("makes sense\n[PREDICTION_END]</u>")
    assert 0 not in data.prompt_ids[3:]


# ── 8. budget guard ──────────────────────────────────────────────────────────
def test_no_room_terminates_without_emitting():
    loop = _loop(["plan"], response_length=30)
    data = _agent_data()
    data.response_mask = [0] * 25
    data.response_logprobs = [0.0] * 25
    state = asyncio.run(loop._run_v7_round(data, {}))
    assert state == AgentState.TERMINATED and data.metrics["v7_turn_skipped_no_budget"] == 1
    assert loop.server_manager.outputs == ["plan"]


def test_inner_revision_without_room_shows_the_draft_and_is_not_a_truncation():
    """No room for another draft ends the loop exactly as the cap does.

    The round still completes and the draft is shown, so this must NOT be
    counted as v7_turn_skipped_no_budget: that counter means an episode was
    truncated at a round boundary, and the smoke gate treats it as fatal.
    On the 6-round writing shape the benign case is routine.
    """
    loop = _loop(["plan", "DRAFT", "d0", "Can you add the date?"], response_length=190)
    data = _agent_data()
    state = asyncio.run(loop._run_v7_round(data, {}))
    rec = data.v7_rounds[0]
    assert rec["verdicts"] == [controller.UNSATISFIED] and rec["inner_iters"] == 0
    assert rec["visible"] == "d0" and [m["content"] for m in loop._v7_sim_view(data)][-1] == "d0"
    assert data.metrics["v7_inner_stopped_no_budget"] == 1
    assert data.metrics.get("v7_turn_skipped_no_budget", 0) == 0
    assert state in (AgentState.INTERACTING, AgentState.TERMINATED)


# ── 9. every emitted v7_* counter is declared and aggregated ─────────────────
def _srcs():
    # Both sources are read from the imported modules, so neither side depends
    # on a layout: AgentLoopMetrics lives in the patched veRL, wherever that is
    # installed, and the rollout loop in this package.
    import verl.experimental.agent_loop.agent_loop as verl_agent_loop

    from ctom import agent_loop as ctom_agent_loop

    return (open(ctom_agent_loop.__file__).read(),
            open(verl_agent_loop.__file__).read())


def test_agent_loop_metrics_declares_and_aggregates_every_v7_key():
    loop_src, metrics_src = _srcs()
    # keys the loop bumps: literal `_bump(agent_data, "v7_x")` plus the
    # f-string families expanded by hand
    emitted = set(re.findall(r'_bump\(agent_data, "(v7_\w+)"', loop_src))
    emitted |= {"v7_verdict_satisfied", "v7_verdict_unsatisfied", "v7_decision_draft",
                "v7_decision_ask", "v7_decision_commit", "v7_gap_tp", "v7_gap_fp", "v7_gap_fn", "v7_gap_tn"}
    assert len(emitted) >= 20, sorted(emitted)
    model = metrics_src.split("class AgentLoopMetrics", 1)[1].split("class ", 1)[0]
    declared = set(re.findall(r"^\s{4}(v7_\w+):", model, re.M))
    agg = set(re.findall(r'"(v7_\w+)"', metrics_src.split("def _performance_metrics", 1)[1]))
    assert emitted <= declared, f"dropped by AgentLoopMetrics: {sorted(emitted - declared)}"
    assert declared == agg, f"declared-but-unlogged: {sorted(declared - agg)}"


# ── satisfied slots carry no aux target (pred_sft_skip_terminate) ────────────
# The canonical substitute was ~20% of the auxiliary supervision and is also the
# string the verdict parser reads as SATISFIED, so the aux loss was teaching the
# policy the password to its own exit gate. Those slots are now unsupervised.
def _cf_loop(outputs, cf_replies, *, skip_terminate=True):
    loop = _loop(outputs)
    loop.config.actor_rollout_ref.actor.pred_sft_skip_terminate = skip_terminate
    return loop, FakeInteraction([], cf_replies)


def test_terminate_only_slots_are_dropped_from_the_aux_target():
    loop, inter = _cf_loop(
        ["plan", "DRAFT", "draft A", "Hmm, step 2 looks wrong?", "draft B", "That makes sense, thanks."],
        ["I don't buy step 2.", "[[TERMINATE CHAT]]"],
    )
    data = _agent_data(interaction=inter)
    asyncio.run(loop._run_v7_round(data, {}))
    asyncio.run(loop._collect_v7_counterfactual_targets(data))
    # slot 0 got real words; slot 1's reply was terminate-only -> no target
    assert data.counterfactual_targets[0] == "I don't buy step 2."
    assert data.counterfactual_targets[1] is None
    # the label is still computed: the call succeeded and fills the 2x2
    assert data.metrics["v7_cf_ok"] == 2 and data.metrics.get("v7_cf_missing", 0) == 0
    assert data.metrics["v7_cf_dropped_satisfied"] == 1
    assert data.metrics["v7_cf_terminate_substituted"] == 1
    assert sum(data.metrics.get("v7_gap_" + c, 0) for c in ("tp", "fp", "fn", "tn")) == 2


def test_an_empty_reply_to_a_draft_is_also_dropped():
    loop, inter = _cf_loop(
        ["plan", "DRAFT", "draft A", "That makes sense, thanks."], [""],
    )
    data = _agent_data(interaction=inter)
    asyncio.run(loop._run_v7_round(data, {}))
    asyncio.run(loop._collect_v7_counterfactual_targets(data))
    assert data.counterfactual_targets[0] is None
    assert data.metrics["v7_cf_dropped_satisfied"] == 1
    assert data.metrics["v7_cf_ok"] == 1, "an empty reply is the satisfied case, not a failure"


def test_the_old_behaviour_is_one_flag_away():
    loop, inter = _cf_loop(
        ["plan", "DRAFT", "draft A", "That makes sense, thanks."], [""],
        skip_terminate=False,
    )
    data = _agent_data(interaction=inter)
    asyncio.run(loop._run_v7_round(data, {}))
    asyncio.run(loop._collect_v7_counterfactual_targets(data))
    assert data.counterfactual_targets[0] == controller.TERMINATE_SUBSTITUTE["math"]
    assert data.metrics.get("v7_cf_dropped_satisfied", 0) == 0
