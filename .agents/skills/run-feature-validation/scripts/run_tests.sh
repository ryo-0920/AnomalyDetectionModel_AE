#!/usr/bin/env bash
set -euo pipefail

if (($# > 0)); then
  "$@"
  exit $?
fi

if [[ -f "Makefile" ]] && grep -Eq '^[[:space:]]*test:' Makefile; then
  exec make test
fi

echo "既定のテスト入口を検出できません。README、AGENTS.md、派生テンプレートで定義されたコマンドを明示してください。例:" >&2
echo "  bash .agents/skills/run-feature-validation/scripts/run_tests.sh <コマンド> [引数...]" >&2
exit 2
