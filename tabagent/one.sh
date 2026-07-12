#!/bin/bash
cd ~/tabagent
pkill -9 -f agent_loop 2>/dev/null; sleep 3
rm -f one_anchored.log
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$HOME/.cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TABAGENT_LLM_URL
# one run, alone on the box: 1 round, 3 priors (base anchor + 2 agent), small steps
.venv/bin/python -u agent_loop.py --arm anchored --rounds 1 --k 3 \
  --steps-per-prior 150 --buffer 36 --audit-n 6 --seed 0 > one_anchored.log 2>&1
echo "EXIT=$?"
grep -hE "prior [0-9]|gap .*DEV|released checkpoint|TEST|anchored *:|wins" one_anchored.log | grep -vE "INFO|Warning"
