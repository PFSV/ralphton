#!/usr/bin/env bash
# Keep the A100 pointed at the laptop's `claude`. The tunnel drops with the network, so
# autossh-style relaunch in a loop. The run now HALTS on a dead LLM rather than silently
# turning into random search, so a drop costs time, not correctness.
while true; do
  SSHPASS="$TABAGENT_SSHPW" sshpass -e ssh -o StrictHostKeyChecking=no \
    -o PreferredAuthentications=password -o PubkeyAuthentication=no \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
    -N -R 8791:127.0.0.1:8791 "$TABAGENT_HOST"
  echo "[tunnel] dropped, reconnecting in 5s" >&2
  sleep 5
done
