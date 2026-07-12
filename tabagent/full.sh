#!/bin/bash
# FULL SCALE. The headline runs so far were 150 steps x 2 rounds x 1 seed -- small enough
# that a real effect could hide under them. The only setting that ever helped (alpha 128,
# +0.0023) used 1500 steps, so give every arm that budget.
#
#   3 arms x 3 seeds, 3 rounds, 3 priors, 1200 steps/prior  = 3600 steps per round
#   agent    LLM proposes the priors
#   random   same budget, same knob space, no LLM        <- the control that decides it
#   anchored TabICL's own prior kept + agent's added
cd ~/tabagent
pkill -9 -f agent_loop 2>/dev/null; sleep 3
rm -f full_*.log
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$HOME/.cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TABAGENT_LLM_URL
A="--rounds 3 --k 3 --steps-per-prior 1200 --buffer 256 --audit-n 16"

for s in 0 1 2; do
  # three at a time: enough to keep the box busy, few enough that the CPU-side generator
  # (which is the real bottleneck) does not thrash.
  .venv/bin/python -u agent_loop.py --arm agent    --seed $s $A > full_agent_$s.log    2>&1 &
  .venv/bin/python -u agent_loop.py --arm random   --seed $s $A > full_random_$s.log   2>&1 &
  .venv/bin/python -u agent_loop.py --arm anchored --seed $s $A > full_anchored_$s.log 2>&1 &
  wait
  echo "### seed $s done  $(date +%H:%M:%S)"
done
echo "### FULL RUN DONE $(date +%H:%M:%S)"
for f in full_*.log; do
  echo "--- $f"; grep -hE "released checkpoint :|(agent|random|anchored) *: 0\.|wins" $f | grep -vE "INFO|Warning"
done
