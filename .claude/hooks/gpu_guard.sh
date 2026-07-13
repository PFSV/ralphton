#!/usr/bin/env bash
# PreToolUse(Bash): GPU 0 belongs to another user. Every CUDA job pins GPU 1 explicitly.
# Blocks (exit 2) any command that looks like it launches compute without an explicit pin.
set -uo pipefail

cmd="$(jq -r '.tool_input.command // ""')"

# Not a python/torch launch -> nothing to guard.
if ! grep -qE '(\.venv/bin/python|python3?[[:space:]]|accelerate|torchrun|deepspeed)' <<<"$cmd"; then
  exit 0
fi
# Read-only inspection and tests are not compute jobs.
if grep -qE '(-m pytest|--help|-c .import|nvidia-smi)' <<<"$cmd"; then
  exit 0
fi

if grep -qE 'CUDA_VISIBLE_DEVICES=0' <<<"$cmd"; then
  echo "BLOCKED: GPU 0 belongs to another user. This lab uses GPU 1 only. Relaunch with CUDA_VISIBLE_DEVICES=1." >&2
  exit 2
fi

if ! grep -qE 'CUDA_VISIBLE_DEVICES=' <<<"$cmd"; then
  echo "BLOCKED: unpinned launch. Every job in this lab pins its device: prefix with CUDA_VISIBLE_DEVICES=1. If this command touches no GPU, pin it anyway (CUDA_VISIBLE_DEVICES='')." >&2
  exit 2
fi

exit 0
