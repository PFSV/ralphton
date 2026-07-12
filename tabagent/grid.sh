#!/bin/bash
# FROZEN PROTOCOL: suite=tabarena (DEV 12 / TEST 24), adapter=lora, lr=1e-5, steps=300,
# rounds=5, seeds 0/1/2. Seeds run CONCURRENTLY on one A100 (LoRA is ~10GB each).
cd ~/tabagent
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$HOME/.cache/hf
export TABAGENT_LLM_URL=http://127.0.0.1:8791
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f stage1.jsonl

run_seed () {
  local seed=$1
  for arm in pretrained base random agent; do
    echo "### $arm seed=$seed START $(date +%H:%M:%S)"
    stdbuf -oL .venv/bin/python stage1.py --arm $arm --seed $seed \
      --suite tabarena --adapter lora --lr 1e-5 --steps 300 --rounds 5 --batch-size 4 \
      2>&1 | grep -vE "INFO:|Downloading|not cached|HF_TOKEN|huggingface_hub|^Warning"
  done
  echo "### SEED $seed DONE $(date +%H:%M:%S)"
}

for seed in 0 1 2; do run_seed $seed & done
wait
echo "### GRID DONE $(date +%H:%M:%S)"
