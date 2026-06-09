#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"
MODEL_ID="${LLAMA_GUARD_3_MODEL:-meta-llama/Llama-Guard-3-8B}"
BATCH_SIZE="${LLAMA_GUARD_BATCH_SIZE:-2}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

run_score() {
  local responses="$1"

  if [[ ! -f "$responses" ]]; then
    echo "Skipping missing responses file: $responses"
    return 0
  fi

  echo "Scoring $responses with $MODEL_ID"
  "$PYTHON_BIN" scripts/score_llamaguard.py \
    --responses "$responses" \
    --model "$MODEL_ID" \
    --batch-size "$BATCH_SIZE"
}

run_score "outputs/vicuna_vllm_jbb_subset/responses.jsonl"
run_score "outputs/gpt4o_mini_jbb_subset/responses.jsonl"
run_score "outputs/vicuna_benign_vllm_jbb_subset/responses.jsonl"
run_score "outputs/vicuna_dictionary_filter_vllm_jbb_subset/responses.jsonl"
