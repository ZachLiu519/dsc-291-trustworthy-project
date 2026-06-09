#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -x "/content/.venv311/bin/python" ]]; then
    PYTHON_BIN="/content/.venv311/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

RUN_SCRIPT="$REPO_ROOT/scripts/run_openai_judge_sensitivity_once.sh"
CRON_LOG="${OPENAI_JUDGE_SENSITIVITY_CRON_LOG:-$REPO_ROOT/logs/api_judge_sensitivity_cron.log}"
mkdir -p "$(dirname "$CRON_LOG")"

if ! command -v crontab >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq cron
  else
    echo "crontab is not installed, and apt-get is unavailable." >&2
    exit 1
  fi
fi

if command -v service >/dev/null 2>&1; then
  service cron start >/dev/null 2>&1 || true
fi

if (($# > 0)); then
  SCHEDULES=("$@")
else
  SCHEDULES=(
    "30 7 * * *"
    "30 15 * * *"
    "30 20 * * *"
    "30 2 * * *"
  )
fi

existing_crontab="$(mktemp)"
new_crontab="$(mktemp)"
trap 'rm -f "$existing_crontab" "$new_crontab"' EXIT

crontab -l >"$existing_crontab" 2>/dev/null || true

awk '
  $0 == "# BEGIN dsc291-openai-judge-sensitivity" { skip = 1; next }
  $0 == "# END dsc291-openai-judge-sensitivity" { skip = 0; next }
  skip != 1 { print }
' "$existing_crontab" >"$new_crontab"

{
  echo "# BEGIN dsc291-openai-judge-sensitivity"
  for schedule in "${SCHEDULES[@]}"; do
    echo "$schedule REPO_ROOT=\"$REPO_ROOT\" PYTHON_BIN=\"$PYTHON_BIN\" bash \"$RUN_SCRIPT\" >> \"$CRON_LOG\" 2>&1"
  done
  echo "# END dsc291-openai-judge-sensitivity"
} >>"$new_crontab"

crontab "$new_crontab"

echo "Installed GPT-4o-mini judge sensitivity cron entries:"
crontab -l
echo "Cron log: $CRON_LOG"
