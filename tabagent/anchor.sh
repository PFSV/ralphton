#!/bin/bash
cd ~/tabagent
pkill -9 -f agent_loop 2>/dev/null; sleep 2
rm -f al_anchored.log
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$HOME/.cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TABAGENT_LLM_URL
.venv/bin/python -u agent_loop.py --arm anchored --rounds 2 --k 3 \
  --steps-per-prior 150 --buffer 48 --audit-n 8 --seed 0 > al_anchored.log 2>&1
echo "ANCHORED EXIT=$?"
grep -hE "gap .*DEV|released|TEST|wins|prior [0-9]" al_anchored.log | grep -vE "INFO|Warning"
