#!/usr/bin/env bash
# PostToolUse(Write|Edit) on reports/ and paper sources: the evidence rule, enforced.
# Any number written into a report must be traceable to a file under runs/.
# Non-blocking: injects a reminder naming the numbers that just appeared.
set -uo pipefail

payload="$(cat)"
path="$(jq -r '.tool_input.file_path // ""' <<<"$payload")"

case "$path" in
  *reports/*|*/paper/*|*.tex|*README.md) ;;
  *) exit 0 ;;
esac

new="$(jq -r '.tool_input.new_string // .tool_input.content // ""' <<<"$payload")"
nums="$(grep -oE '[0-9]+\.[0-9]+|[0-9]+%' <<<"$new" | sort -u | head -12 | tr '\n' ' ')"
[ -z "$nums" ] && exit 0

cat <<EOF
EVIDENCE RULE — numbers just written to ${path}: ${nums}
Each must have been read from a file under runs/ during this session. For each, be able to state:
the value, the file, the run id in runs/registry.jsonl, and the git SHA of that run. Any number you
cannot trace to that triple must be replaced with UNVERIFIED, not with a plausible value.
EOF
exit 0
