#!/usr/bin/env bash
# CToM (sft_v7) evaluation — held-out protocol for all three domains.
#
# Three stages, all driven from here:
#   1. turn the veRL FSDP LoRA checkpoint into a merged HF model;
#   2. serve it with vLLM and run the domain's rollout driver against the
#      frozen user simulator;
#   3. score. Math and coding score inline; writing writes rollouts.jsonl and
#      All three score inline; writing's ROUGE-L is computed from the rollout.
#
#   bash scripts/eval_ctom.sh --domain math    --checkpoint runs/.../global_step_180
#   bash scripts/eval_ctom.sh --domain writing --checkpoint ... --base-model Qwen/Qwen3-8B
#   bash scripts/eval_ctom.sh --domain coding  --checkpoint ...
#
# Pass --checkpoint base to evaluate the untrained policy (the base anchor in
# the paper's tables): no merge, the base model is served directly.
#
# The protocol constants below are the ones the reported numbers were produced
# with. Assistant decoding is avg@8 at temperature 1.0 on math and coding, and
# a single near-greedy rollout on writing, matching the CollabLLM protocol.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOMAIN=""
CKPT=""
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SIM_URL="${SIM_URL:-http://127.0.0.1:8100}"
SIM_MODEL="${SIM_MODEL:-Qwen/Qwen3-32B-FP8}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
OUT_DIR="${OUT_DIR:-}"
ASSISTANT_PORT="${ASSISTANT_PORT:-8201}"
ACTION_PROTOCOL="${ACTION_PROTOCOL:-categorical_v1}"
INNER_REVISIONS="${INNER_REVISIONS:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"        # 0 = the whole evaluation split
GPU_UTIL="${GPU_UTIL:-0.85}"
TP_SIZE="${TP_SIZE:-1}"
KEEP_SERVER=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)           DOMAIN="$2";           shift 2 ;;
        --checkpoint)       CKPT="$2";             shift 2 ;;
        --base-model)       BASE_MODEL="$2";       shift 2 ;;
        --sim-url)          SIM_URL="$2";          shift 2 ;;
        --sim-model)        SIM_MODEL="$2";        shift 2 ;;
        --data-root)        DATA_ROOT="$2";        shift 2 ;;
        --out-dir)          OUT_DIR="$2";          shift 2 ;;
        --action-protocol)  ACTION_PROTOCOL="$2";  shift 2 ;;
        --inner-revisions)  INNER_REVISIONS="$2";  shift 2 ;;
        --max-samples)      MAX_SAMPLES="$2";      shift 2 ;;
        --tensor-parallel)  TP_SIZE="$2";          shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$DOMAIN" ]] || { echo "ERROR: --domain math|writing|coding is required" >&2; exit 1; }
[[ -n "$CKPT"   ]] || { echo "ERROR: --checkpoint <dir>|base is required" >&2; exit 1; }

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ── 1. Merge the LoRA adapter into the base model ───────────────────────────
# veRL writes lora_alpha=0 into the extracted adapter; training used 32, and the
# adapter is inert at 0, so it has to be restored before the merge.
if [[ "$CKPT" == "base" ]]; then
    ASSISTANT_MODEL="$BASE_MODEL"
    LABEL="base"
else
    ACTOR_DIR="$CKPT/actor"
    [[ -f "$ACTOR_DIR/fsdp_config.json" ]] || { echo "ERROR: $ACTOR_DIR/fsdp_config.json not found" >&2; exit 1; }
    EXTRACT_DIR="$CKPT/extracted_adapter"
    MERGED_DIR="$CKPT/merged_for_eval"
    LABEL="$(basename "$CKPT")"
    if [[ ! -f "$MERGED_DIR/config.json" ]]; then
        if [[ ! -d "$EXTRACT_DIR/lora_adapter" ]]; then
            python3 -m verl.model_merger merge --backend fsdp \
                --local_dir "$ACTOR_DIR" --target_dir "$EXTRACT_DIR"
        fi
        python3 - "$EXTRACT_DIR/lora_adapter/adapter_config.json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as fh:
    cfg = json.load(fh)
cfg["lora_alpha"] = 32
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=4)
PY
        python3 "$PROJECT_DIR/evaluation/merge_lora.py" \
            --base_model "$BASE_MODEL" \
            --lora_adapter "$EXTRACT_DIR/lora_adapter" \
            --output_dir "$MERGED_DIR"
    fi
    ASSISTANT_MODEL="$MERGED_DIR"
fi

OUT_DIR="${OUT_DIR:-$PROJECT_DIR/eval_outputs/ctom-$DOMAIN-$LABEL}"
mkdir -p "$OUT_DIR"

# ── 2. Serve the assistant ──────────────────────────────────────────────────
# --served-model-name is the BASE id so the drivers' tokenizer lookup and the
# served name agree whether the model is merged or raw.
python -m vllm.entrypoints.openai.api_server \
    --model "$ASSISTANT_MODEL" \
    --served-model-name "$BASE_MODEL" \
    --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --tensor-parallel-size "$TP_SIZE" \
    --host 0.0.0.0 --port "$ASSISTANT_PORT" \
    > "$OUT_DIR/vllm_assistant.log" 2>&1 &
ASST_PID=$!
cleanup() { kill "$ASST_PID" 2>/dev/null || true; wait "$ASST_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for the assistant server on port $ASSISTANT_PORT ..."
for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$ASSISTANT_PORT/health" > /dev/null && break
    kill -0 "$ASST_PID" 2>/dev/null || { echo "ERROR: vLLM died, see $OUT_DIR/vllm_assistant.log" >&2; exit 1; }
    sleep 5
done
ASSISTANT_URL="http://127.0.0.1:$ASSISTANT_PORT/v1"

COMMON=(
    --prompt_dir "$PROJECT_DIR/prompts"
    --output_dir "$OUT_DIR"
    --max_samples "$MAX_SAMPLES"
    --seed 42
    --assistant_api_base "$ASSISTANT_URL"
    --assistant_model_name "$BASE_MODEL"
    --assistant_output_mode sft_v7
    --v7_action_protocol "$ACTION_PROTOCOL"
    --v7_think_max_tokens 256
    --v7_ask_max_tokens 512
    --v7_max_inner_revisions "$INNER_REVISIONS"
    --top_p 1.0
    --max_context_tokens 0
    --user_sim_api_base "$SIM_URL/v1"
)

# ── 3. Run the domain driver ────────────────────────────────────────────────
case "$DOMAIN" in
math)
    # all 365 held-out Level-5 problems, avg@8 at T=1.0, expert-only simulator.
    python3 "$PROJECT_DIR/evaluation/eval_math.py" "${COMMON[@]}" \
        --test_data "$DATA_ROOT/math-l5-expert/rl_validation.v7_prompt.parquet" \
        --base_model_path "$BASE_MODEL" \
        --max_turns 4 \
        --temperature 1.0 \
        --n_samples 8 \
        --num_workers 16 \
        --max_model_len 32768 \
        --max_response_length 30720 \
        --trajectory_response_length 30720 \
        --pred_answer_max_tokens 2048 \
        --prediction_max_tokens 384 \
        --user_sim_novice_model "$SIM_MODEL" \
        --user_sim_expert_model "$SIM_MODEL" \
        --user_sim_temperature 0.7 \
        --user_sim_max_tokens 1024 \
        --user_sim_max_model_len 32768 \
        --assistant_system_prompt_from_data \
        --replay_q0 \
        --answer_gated_termination \
        --effort_mode cost \
        --effort_token_units tokens \
        --effort_max_turns 4 \
        --effort_max_user_tokens 332 \
        --effort_max_assistant_tokens 1024 \
        --effort_alpha 0.5 \
        --effort_beta 0.4 \
        --invalid_phase_penalty -0.2 \
        --reward_w_task 1.0 --reward_w_pred 0.0 --reward_w_effort 0.01
    ;;
writing)
    # CollabLLM MediumDocEdit.
    python3 "$PROJECT_DIR/evaluation/eval_writing.py" "${COMMON[@]}" \
        --test_data "$DATA_ROOT/collabllm-medium/rl_validation.v7_prompt.parquet" \
        --tokenizer_path "$BASE_MODEL" \
        --max_turns 6 \
        --temperature 1.0 \
        --num_workers 16 \
        --max_response_length 30720 \
        --trajectory_response_length 30720 \
        --pred_answer_max_tokens 1536 \
        --prediction_max_tokens 256 \
        --user_sim_model "$SIM_MODEL" \
        --user_sim_temperature 0.7 \
        --user_sim_max_tokens 384
    cleanup; trap - EXIT
    echo ""
    echo "Per-task rollouts and ROUGE-L scores: $OUT_DIR/rollouts.jsonl"
    ;;
coding)
    # CollabLLM BigCodeBench: the driver runs the hidden unit tests inline, so
    # there is no second scoring stage and no judge.
    : "${CTOM_BCB_PYTHON:?set CTOM_BCB_PYTHON to the BigCodeBench execution interpreter}"
    export CTOM_BCB_PYTHON CTOM_BCB_TIMEOUT="${CTOM_BCB_TIMEOUT:-30}"
    python3 "$PROJECT_DIR/evaluation/eval_coding.py" "${COMMON[@]}" \
        --test_data "$DATA_ROOT/collabllm-bigcodebench/rl_validation.v7b_prompt.parquet" \
        --tokenizer_path "$BASE_MODEL" \
        --max_turns 6 \
        --temperature 1.0 \
        --n_samples 8 \
        --num_workers 24 \
        --max_response_length 24576 \
        --trajectory_response_length 24576 \
        --pred_answer_max_tokens 1024 \
        --prediction_max_tokens 256 \
        --pred_revision_max_tokens 1664 \
        --user_sim_model "$SIM_MODEL" \
        --user_sim_temperature 0.7 \
        --user_sim_max_tokens 384
    ;;
*) echo "ERROR: --domain must be math, writing or coding" >&2; exit 1 ;;
esac

echo ""
echo "Results: $OUT_DIR"
