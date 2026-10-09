#!/usr/bin/env bash
# CToM (code name: sft_v7) RL training — single entry point for all three domains.
#
# This is a portable replacement for the authors' cluster launcher: it contains
# only the hyperparameters that define the experiments in the paper, with no
# scheduler, no multi-node bootstrap and no site-specific paths.
#
#   bash scripts/train_ctom.sh --domain math    --arm ctom
#   bash scripts/train_ctom.sh --domain math    --arm task-only
#   bash scripts/train_ctom.sh --domain math    --arm no-effort
#   bash scripts/train_ctom.sh --domain math    --arm no-prediction
#   bash scripts/train_ctom.sh --domain writing --arm ctom
#   bash scripts/train_ctom.sh --domain coding  --arm ctom
#
# Prerequisites (see README.md):
#   1. veRL installed from commit 044bbba2 with patches/verl-ctom.patch applied.
#   2. A user-simulator server reachable at --sim-url
#      (scripts/serve_user_simulator.sh).
#   3. Prepared parquet data under --data-root (data_prep/build_*.py).
#   4. A running Ray cluster, or none at all (veRL starts a local one).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
DOMAIN=""
ARM="ctom"
POLICY="${POLICY:-Qwen/Qwen3-8B}"
SIM_URL="${SIM_URL:-http://127.0.0.1:8100}"
SIM_MODEL="${SIM_MODEL:-Qwen/Qwen3-32B-FP8}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-}"
NNODES="${NNODES:-1}"
NGPUS="${NGPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
# Main paper setting: the single-token categorical controller head, decision
# entropy 0. `legacy` is the free-text controller; entropy 1e-3 is the ablation.
ACTION_PROTOCOL="${ACTION_PROTOCOL:-categorical_v1}"
DECISION_ENTROPY="${DECISION_ENTROPY:-0.0}"
INNER_REVISIONS="${INNER_REVISIONS:-3}"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)            DOMAIN="$2";            shift 2 ;;
        --arm)               ARM="$2";               shift 2 ;;
        --policy)            POLICY="$2";            shift 2 ;;
        --sim-url)           SIM_URL="$2";           shift 2 ;;
        --sim-model)         SIM_MODEL="$2";         shift 2 ;;
        --data-root)         DATA_ROOT="$2";         shift 2 ;;
        --save-dir)          SAVE_DIR="$2";          shift 2 ;;
        --nnodes)            NNODES="$2";            shift 2 ;;
        --ngpus-per-node)    NGPUS="$2";             shift 2 ;;
        --tensor-parallel)   TP_SIZE="$2";           shift 2 ;;
        --action-protocol)   ACTION_PROTOCOL="$2";   shift 2 ;;
        --decision-entropy)  DECISION_ENTROPY="$2";  shift 2 ;;
        --inner-revisions)   INNER_REVISIONS="$2";   shift 2 ;;
        --)                  shift; EXTRA+=("$@");   break   ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$DOMAIN" ]] || { echo "ERROR: --domain math|writing|coding is required" >&2; exit 1; }

# ── Arm: CToM and its ablations (paper Table 1) ────────────────────────────
# ctom           = prediction NLL loss on  + interaction cost on.
# no-effort      = prediction NLL loss on  + interaction cost off (w_effort = 0).
# no-prediction  = prediction NLL loss off (lambda_pred = 0) + interaction cost on.
# task-only      = both off, i.e. GRPO on the task reward alone, same controller.
case "$ARM" in
    ctom)          PRED_SFT_COEF=0.1; REWARD_W_EFFORT=0.01 ;;
    no-effort)     PRED_SFT_COEF=0.1; REWARD_W_EFFORT=0.0  ;;
    no-prediction) PRED_SFT_COEF=0.0; REWARD_W_EFFORT=0.01 ;;
    task-only)     PRED_SFT_COEF=0.0; REWARD_W_EFFORT=0.0  ;;
    *) echo "ERROR: --arm must be ctom, no-effort, no-prediction or task-only" >&2; exit 1 ;;
esac

# ── Per-domain constants (Table: shared training hyperparameters) ───────────
case "$DOMAIN" in
math)
    DATASET=math-l5-expert
    TRAIN_FILE="$DATA_ROOT/math-l5-expert/rl_train.v7_prompt.parquet"
    VAL_FILE="$DATA_ROOT/math-l5-expert/rl_validation.v7_prompt.val200.parquet"
    MAX_USER_TURNS=3;  MAX_ASSISTANT_TURNS=4;  EFFORT_MAX_TURNS=4
    MAX_RESPONSE_LENGTH=30720;  ROLLOUT_MAX_MODEL_LEN=32768;  PPO_MAX_TOKEN_LEN=36000
    TOTAL_EPOCHS=2
    DRAFT_MAX_TOKENS=0            # 0 => the draft cap is PRED_ANSWER_MAX_TOKENS
    PRED_ANSWER_MAX_TOKENS=2048;  PREDICTION_MAX_TOKENS=384
    EFFORT_TOKEN_UNITS=tokens
    EFFORT_MAX_USER_TOKENS=332;   EFFORT_MAX_ASSISTANT_TOKENS=1024
    # The strict answer checker, not the simulator, ends a math episode.
    ANSWER_GATED_TERMINATION=true
    ;;
writing)
    DATASET=collabllm-medium
    TRAIN_FILE="$DATA_ROOT/collabllm-medium/rl_train.v7_prompt.parquet"
    VAL_FILE="$DATA_ROOT/collabllm-medium/rl_validation.v7_prompt.parquet"
    MAX_USER_TURNS=6;  MAX_ASSISTANT_TURNS=6;  EFFORT_MAX_TURNS=6
    MAX_RESPONSE_LENGTH=30720;  ROLLOUT_MAX_MODEL_LEN=32768;  PPO_MAX_TOKEN_LEN=36000
    TOTAL_EPOCHS=16
    DRAFT_MAX_TOKENS=1536
    PRED_ANSWER_MAX_TOKENS=1664;  PREDICTION_MAX_TOKENS=256
    EFFORT_TOKEN_UNITS=words
    EFFORT_MAX_USER_TOKENS=2000;  EFFORT_MAX_ASSISTANT_TOKENS=900
    ANSWER_GATED_TERMINATION=false
    ;;
coding)
    DATASET=collabllm-bigcodebench
    TRAIN_FILE="$DATA_ROOT/collabllm-bigcodebench/rl_train.v7b_prompt.parquet"
    VAL_FILE="$DATA_ROOT/collabllm-bigcodebench/rl_validation.v7b_prompt.parquet"
    MAX_USER_TURNS=6;  MAX_ASSISTANT_TURNS=6;  EFFORT_MAX_TURNS=6
    MAX_RESPONSE_LENGTH=24576;  ROLLOUT_MAX_MODEL_LEN=32768;  PPO_MAX_TOKEN_LEN=30000
    TOTAL_EPOCHS=16
    DRAFT_MAX_TOKENS=1024
    PRED_ANSWER_MAX_TOKENS=1664;  PREDICTION_MAX_TOKENS=256
    EFFORT_TOKEN_UNITS=words
    EFFORT_MAX_USER_TOKENS=2000;  EFFORT_MAX_ASSISTANT_TOKENS=600
    ANSWER_GATED_TERMINATION=false
    : "${CTOM_BCB_PYTHON:?set CTOM_BCB_PYTHON to the BigCodeBench execution interpreter}"
    export CTOM_BCB_PYTHON
    export CTOM_BCB_TIMEOUT="${CTOM_BCB_TIMEOUT:-30}"
    ;;
*) echo "ERROR: --domain must be math, writing or coding" >&2; exit 1 ;;
esac

for f in "$TRAIN_FILE" "$VAL_FILE"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f — build it with data_prep/build_*.py (see README)." >&2; exit 1; }
done

RUN_DIR="${RUN_DIR:-$PROJECT_DIR/runs/ctom-$DOMAIN-$ARM-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
SAVE_DIR="${SAVE_DIR:-$RUN_DIR/checkpoints}"
# One trace file per rollout worker holding every plan, decision, internal draft,
# prediction and verdict. Set CTOM_V7_TRACE=off to disable.
export CTOM_V7_TRACE="${CTOM_V7_TRACE:-$RUN_DIR/v7_trace.jsonl}"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ── User-simulator interaction config ───────────────────────────────────────
# api_base also accepts a comma-separated list of replica URLs.
INTERACTION_CONFIG="$RUN_DIR/interaction.yaml"
cat > "$INTERACTION_CONFIG" <<EOF
interaction:
  - name: ctom
    class_name: ctom.user_simulator.CToMUserSimulator
    config: {
      "api_base": "${SIM_URL}/v1",
      "api_key": "not-needed",
      "user_model_novice": "${SIM_MODEL}",
      "user_model_expert": "${SIM_MODEL}",
      "user_model_default": "${SIM_MODEL}",
      "user_model_bigcodebench": "${SIM_MODEL}",
      "prompt_dir": "${PROJECT_DIR}/prompts",
      "temperature": 0.7,
      "max_tokens": 1024,
      "termination_signal": "[[TERMINATE CHAT]]",
      "understood_token": "<understood>",
      "answer_gated_termination": ${ANSWER_GATED_TERMINATION},
      "num_retries": 3,
      "enable_log": true,
    }
EOF

# The categorical controller scores its decision token from the rollout logprobs.
CATEGORICAL_OVERRIDES=()
if [[ "$ACTION_PROTOCOL" == "categorical_v1" ]]; then
    CATEGORICAL_OVERRIDES+=(
        actor_rollout_ref.rollout.calculate_log_probs=True
        actor_rollout_ref.rollout.logprobs_mode=processed_logprobs
    )
fi

cat <<BANNER
============================================================
  CToM (sft_v7) training
    domain            $DOMAIN ($DATASET)
    arm               $ARM  (pred_sft_coef=$PRED_SFT_COEF, w_effort=$REWARD_W_EFFORT)
    controller        $ACTION_PROTOCOL, decision entropy $DECISION_ENTROPY
    inner revisions   <= $INNER_REVISIONS
    policy init       $POLICY
    user simulator    $SIM_MODEL @ $SIM_URL
    rounds            user<=$MAX_USER_TURNS assistant<=$MAX_ASSISTANT_TURNS
    checkpoints       $SAVE_DIR
    phase trace       $CTOM_V7_TRACE
============================================================
BANNER

python3 -m verl.trainer.main_ppo \
    "${CATEGORICAL_OVERRIDES[@]}" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=36 \
    data.max_prompt_length=4096 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.seed=42 \
    reward_model.reward_manager=ctom \
    custom_reward_function.path="$PROJECT_DIR/ctom/reward.py" \
    custom_reward_function.name=ctom_reward_func \
    +reward.reward_model.reward_kwargs.prediction_mode=sft_v7 \
    +reward.reward_model.reward_kwargs.pred_sft_format=v7 \
    +reward.reward_model.reward_kwargs.reward_composition=ctom \
    +reward.reward_model.reward_kwargs.reward_granularity=turn \
    +reward.reward_model.reward_kwargs.reward_placement=turn \
    +reward.reward_model.reward_kwargs.reward_weights.task=1.0 \
    +reward.reward_model.reward_kwargs.reward_weights.prediction=0.0 \
    +reward.reward_model.reward_kwargs.reward_weights.effort=$REWARD_W_EFFORT \
    +reward.reward_model.reward_kwargs.effort_mode=cost \
    +reward.reward_model.reward_kwargs.effort_kwargs.max_turns=$EFFORT_MAX_TURNS \
    +reward.reward_model.reward_kwargs.effort_kwargs.max_user_tokens=$EFFORT_MAX_USER_TOKENS \
    +reward.reward_model.reward_kwargs.effort_kwargs.max_assistant_tokens=$EFFORT_MAX_ASSISTANT_TOKENS \
    +reward.reward_model.reward_kwargs.effort_kwargs.alpha=0.5 \
    +reward.reward_model.reward_kwargs.effort_kwargs.beta=0.4 \
    +reward.reward_model.reward_kwargs.effort_kwargs.token_units=$EFFORT_TOKEN_UNITS \
    +reward.reward_model.reward_kwargs.effort_kwargs.count_revision_tokens=false \
    +reward.reward_model.reward_kwargs.invalid_phase_penalty=-0.2 \
    actor_rollout_ref.model.path="$POLICY" \
    critic.model.path="$POLICY" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.pred_sft_coef=$PRED_SFT_COEF \
    actor_rollout_ref.actor.pred_sft_coef_final=-1 \
    actor_rollout_ref.actor.pred_sft_grad_ratio_cap=0.5 \
    actor_rollout_ref.actor.pred_sft_skip_terminate=true \
    actor_rollout_ref.actor.pred_nll_reward_coef=0.0 \
    actor_rollout_ref.actor.pred_answer_max_tokens=$PRED_ANSWER_MAX_TOKENS \
    actor_rollout_ref.actor.pred_prediction_max_tokens=$PREDICTION_MAX_TOKENS \
    actor_rollout_ref.actor.bdr_draft_max_tokens=$DRAFT_MAX_TOKENS \
    actor_rollout_ref.actor.v7_think_max_tokens=256 \
    actor_rollout_ref.actor.v7_ask_max_tokens=512 \
    actor_rollout_ref.actor.v7_max_inner_revisions=$INNER_REVISIONS \
    actor_rollout_ref.actor.v7_verdict_mode=parse \
    actor_rollout_ref.actor.v7_action_protocol=$ACTION_PROTOCOL \
    actor_rollout_ref.actor.decision_entropy_coef=$DECISION_ENTROPY \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts=1 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_DIR/configs/agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.num_workers=9 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=ctom \
    trainer.experiment_name="ctom-$DOMAIN-$ARM" \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.save_freq=50 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$SAVE_DIR" \
    "${EXTRA[@]}" 2>&1 | tee -a "$RUN_DIR/training.log"
