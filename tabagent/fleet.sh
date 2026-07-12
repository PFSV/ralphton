#!/bin/bash
cd ~/tabagent
pkill -9 -f agent_loop 2>/dev/null; sleep 3
rm -f fl_*.log
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$HOME/.cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TABAGENT_LLM_URL
A="--rounds 2 --k 3 --steps-per-prior 150 --buffer 48 --audit-n 8"

# fill the card: 6 runs in parallel (~6GB each), and this also buys the seeds we need
# for a paired test instead of a single-seed anecdote.
for s in 0 1; do
  .venv/bin/python -u agent_loop.py --arm anchored --seed $s $A > fl_anchored_$s.log 2>&1 &
  .venv/bin/python -u agent_loop.py --arm agent    --seed $s $A > fl_agent_$s.log    2>&1 &
  .venv/bin/python -u agent_loop.py --arm random   --seed $s $A > fl_random_$s.log   2>&1 &
done
sleep 90
echo "=== gpu now ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 1
wait
echo "=== FLEET DONE ==="
for f in fl_*.log; do
  echo "--- $f"; grep -hE "TEST|released checkpoint|wins" $f | grep -vE "INFO|Warning"
done
