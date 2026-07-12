# Repairing the Prior

An LLM agent revises the synthetic pre-training prior of a tabular foundation model; LoRA
makes each candidate prior cheap to evaluate; held-out real tasks say whether it worked.

## The idea

TabICL/TabPFN are pre-trained on a prior over random structural causal models. That prior is
frozen when the checkpoint ships and was chosen without any downstream task in view. Nobody
re-pre-trains a TFM to fix it — that costs a cluster.

So instead: expose the prior's generating configuration, let an agent edit it from downstream
failure, and LoRA-adapt the *released* checkpoint to each candidate (2.1% of parameters,
~2 min/candidate on one A100). Keep the best on DEV, report it on TEST.

The arm that decides the paper is **random search over the same knobs, same budget**. Without
it, any gain could just be search.

## Layout

| file | what it is |
|---|---|
| `priortrain.py` | prior knob space, `PriorDataset` construction, LoRA injection + training loop |
| `stage1.py` | the search arms (`pretrained` / `base` / `random` / `agent`) and the DEV→TEST protocol |
| `agent.py` | a second, inference-time agent (context/feature/acquire/synthesise under a cost budget) |
| `data.py` | OpenML tasks, the ctx/pool/val/test split, column anonymisation |
| `tfm.py` | frozen TabICL scorer + the credit cost model |
| `llm.py`, `llm_server.py` | LLM backend. Experiments run on the A100; the `claude` CLI only exists on the laptop, so the server calls back through an SSH reverse tunnel. Every response is cached by prompt hash, so the run is resumable and replayable at zero LLM cost. |
| `emit.py` | writes `paper/numbers.tex` + `paper/fig_search.pdf` **from `stage1.jsonl`** |
| `paper/` | ICML 2025 style, 4 pages |

**No number in the paper is typed by hand.** `main.tex` only ever references macros that
`emit.py` generates from the results file.

## Reproducing

```bash
# laptop: serve the agent's brain
python llm_server.py 8765

# A100: run the grid, calling back through the tunnel
ssh -R 8791:127.0.0.1:8765 user@host
export CUDA_VISIBLE_DEVICES=1 TABAGENT_LLM_URL=http://127.0.0.1:8791
for seed in 0 1 2; do
  for arm in pretrained base random agent; do
    python stage1.py --arm $arm --seed $seed --rounds 6 --steps 200
  done
done

# laptop: numbers + figure + pdf
python emit.py && (cd paper && ../tools/tectonic -X compile main.tex --outdir .)
```

## Protocol (frozen before the grid ran)

- Backbone: released `tabicl-classifier-v2`, never re-pre-trained.
- 5 DEV tasks (agent may see summary stats — **never rows, never the task's name**), 6 disjoint
  TEST tasks the loop never touches.
- 6 prior candidates per search arm, 200 LoRA steps each, 3 seeds.
- Significance: paired differences per (task, seed), n=18. Bootstrap 95% CI + Wilcoxon.
- Excluded `adult` on purpose — Bordt et al. (2024) show it is memorised verbatim by
  GPT-class models.

Seeds, tasks, rounds and the selection rule were fixed in code before results existed and
were not revisited afterwards.

## Note on the second agent (`agent.py`)

Built first, kept because it is a clean complementary result: at *inference* time, with the
model frozen and no training at all, an agent buys/derives/synthesises context under a credit
budget (one credit = one real labelled row). On `credit-g` it reached test AUC 0.7789 from a
0.7652 baseline for 33 of 100 credits, spending them on three world-knowledge features
(`monthly_installment_burden`, `repayment_strain`, `liquidity_adjusted_burden`) and 30
uncertainty-sampled rows. Not in the 4-page paper; it is a separate story about inference-time
context, not the prior.
