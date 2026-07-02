#!/usr/bin/env bash
set -euo pipefail

feature="${CODEX_VALIDATION_FEATURE:-general}"

if [[ "${1:-}" == "--feature" ]]; then
  if [[ $# -lt 3 ]]; then
    echo "使い方: $0 [--feature FEATURE] <コマンド> [引数...]" >&2
    exit 2
  fi
  feature="$2"
  shift 2
fi

if (($# == 0)); then
  echo "使い方: $0 [--feature FEATURE] <コマンド> [引数...]" >&2
  exit 2
fi

safe_feature="$(printf '%s' "$feature" | tr -c 'A-Za-z0-9._-' '_' | sed -E 's/^[^A-Za-z0-9]+//; s/[^A-Za-z0-9]+$//')"
if [[ -z "$safe_feature" || "$safe_feature" == "." || "$safe_feature" == ".." ]]; then
  safe_feature="general"
fi

mkdir -p ".codex/validation/artifacts/${safe_feature}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file=".codex/validation/artifacts/${safe_feature}/failure-${timestamp}.log"

set +e
"$@" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e

echo "" | tee -a "$log_file"
echo "コマンド: $*" | tee -a "$log_file"
echo "終了ステータス: $status" | tee -a "$log_file"
echo "ログ: $log_file" | tee -a "$log_file"

exit "$status"
