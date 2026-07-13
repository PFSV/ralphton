#!/usr/bin/env bash
# SessionStart: the lab's state, every session. Ledger first, GPU second.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || exit 0

echo "=== LAB STATE ==="

if [ -s reports/claims.yaml ]; then
  echo "-- claims (reports/claims.yaml)"
  grep -E '^\s*-?\s*(id|claim|status):' reports/claims.yaml | head -30
else
  echo "-- claims: reports/claims.yaml is EMPTY. Nothing is currently claimed."
fi

if [ -s runs/registry.jsonl ]; then
  echo "-- last runs"
  tail -3 runs/registry.jsonl
else
  echo "-- runs: registry.jsonl is EMPTY. No registered run exists; no number is admissible yet."
fi

echo "-- GPU (1 is ours, 0 is not)"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

echo "Constitution: CONSTITUTION.md. Permanent questions: QUESTIONS.md."
exit 0
