#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -x "/content/.venv311/bin/python" ]]; then
    PYTHON_BIN="/content/.venv311/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

JUDGE_MODEL="${OPENAI_JUDGE_MODEL:-gpt-4o-mini}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OPENAI_JUDGE_SENSITIVITY_OUTPUT_ROOT:-outputs/api_judge_sensitivity}"
LOG_DIR="${OPENAI_JUDGE_SENSITIVITY_LOG_DIR:-logs/api_judge_sensitivity}"
LIMIT="${OPENAI_JUDGE_LIMIT:-}"

mkdir -p "$OUTPUT_ROOT/$RUN_TIMESTAMP" "$LOG_DIR"
LOG_FILE="$LOG_DIR/$RUN_TIMESTAMP.log"

score_responses() {
  local label="$1"
  local responses="$2"
  local output_dir="$OUTPUT_ROOT/$RUN_TIMESTAMP/$label"

  if [[ ! -f "$responses" ]]; then
    echo "Skipping missing responses file: $responses"
    return 0
  fi

  echo "Scoring $responses with $JUDGE_MODEL -> $output_dir"
  local args=(
    "scripts/score_openai_judge.py"
    "--responses" "$responses"
    "--output-dir" "$output_dir"
    "--judge-model" "$JUDGE_MODEL"
  )
  if [[ -n "$LIMIT" ]]; then
    args+=("--limit" "$LIMIT")
  fi

  "$PYTHON_BIN" "${args[@]}"
}

{
  echo "Started OpenAI judge sensitivity run at $(date -Is)"
  echo "Repo: $REPO_ROOT"
  echo "Python: $PYTHON_BIN"
  echo "Judge model: $JUDGE_MODEL"
  echo "Run timestamp: $RUN_TIMESTAMP"

  status=0
  score_responses "vicuna_harmful" "outputs/vicuna_vllm_jbb_subset/responses.jsonl" || status=1
  score_responses "gpt4o_mini_harmful" "outputs/gpt4o_mini_jbb_subset/responses.jsonl" || status=1

  echo "Finished OpenAI judge sensitivity run at $(date -Is)"
  exit "$status"
} 2>&1 | tee "$LOG_FILE"
