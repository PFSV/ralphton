#!/bin/bash
cd ~/tabagent
pkill -9 -f agent_loop 2>/dev/null; sleep 2
rm -f al_agent.log al_random.log agentloop_*.json
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$HOME/.cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TABAGENT_LLM_URL
# proven-fast config: ~60s per round
A="--rounds 2 --k 2 --steps-per-prior 150 --buffer 48 --audit-n 8 --seed 0"
.venv/bin/python -u agent_loop.py --arm agent  $A > al_agent.log  2>&1 &
P1=$!
.venv/bin/python -u agent_loop.py --arm random $A > al_random.log 2>&1 &
P2=$!
wait $P1 $P2
echo "=== BOTH DONE ==="
grep -hE "gap .*DEV|released checkpoint|TEST|wins|agent-|random-" al_agent.log al_random.log | grep -vE "INFO|Warning"
