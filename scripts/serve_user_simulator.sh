#!/usr/bin/env bash
# Serve the frozen user simulator behind an OpenAI-compatible endpoint.
#
# The simulator is a frozen Qwen3-32B-FP8 in every experiment reported in the
# paper; personas are carried by the data, not by the model. Both the training
# rollout loop and the evaluation drivers talk to this one endpoint.
#
#   bash scripts/serve_user_simulator.sh                 # port 8100
#   PORT=8200 MAX_MODEL_LEN=32768 bash scripts/serve_user_simulator.sh
#
# Context length: 16384 during training (what the runs used), 32768 during
# math/writing evaluation because those protocols keep longer visible histories.
# For more throughput, start several replicas on different ports and pass the
# comma-separated list as --sim-url / --user_sim_api_base.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B-FP8}"
PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TP_SIZE="${TP_SIZE:-1}"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --tensor-parallel-size "$TP_SIZE" \
    --host 0.0.0.0 \
    --port "$PORT"
