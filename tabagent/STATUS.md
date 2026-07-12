# STATUS — read this before trusting anything here

Work in progress. This file records what is measured, what is broken, and what is not yet
known. Nothing below is a claim the experiments have earned unless it says so.

## The question

TabICL/TabPFN are pre-trained on a synthetic prior over random SCMs, frozen at publication.
Can an LLM agent *repair* that prior after the fact — revise its generating configuration
from downstream evidence, LoRA-adapt the released checkpoint to the revision, and win on a
real benchmark? The arm that decides it is **random search over the same knobs, same budget**.

## Established (measured, reproducible)

- **The released prior does not look like real tabular data.** Sampling from it and measuring
  against TabArena's 36 TabICL-runnable classification tasks: median 67 features vs 22 real;
  ~0% categorical-like columns vs 39.5% real; minority-class fraction 0.08 vs 0.23.
  (`prior_audit.py`; TabArena metadata is the ground truth, not our own selection.)
- **The agent finds those same axes unprompted.** Shown only anonymised summary statistics of
  DEV tasks — never a row, never a task name — it proposed `max_features 100→50`,
  `cat_prob 0.2→0.35`, `replay_small 0→1`, `max_classes 10→2`. The real answers were 22
  features, 39.5% categorical, small tables, binary.
- **Learning rate was destroying the model, not the prior.** At lr=1e-4 every arm lost ~0.006
  test AUC purely by over-writing pre-trained weights. This invalidated several hours of runs.
  See `sweep_adapt.py`.
- **Full fine-tuning is not the answer.** It moves DEV more than LoRA (+0.0042 vs +0.0002) and
  none of it transfers to TEST (−0.0017). The extra capacity buys DEV overfitting.
  See `sweep_full.py`.

## The open problem (as of this commit)

At lr=1e-5 the adapter is **inert**: `pretrained`, `base` and `random` land on the same test
AUC to four decimals (0.8171 / 0.8171 / 0.8171 at seed 0). So adaptation is currently either
destructive (1e-4) or a no-op (1e-5). Until a setting exists where adapting to a *different*
prior actually moves the model, the central comparison cannot be run — a null here would
measure our optimiser, not the agent.

**This is the next thing to fix.** Not a result.

## Runs that were thrown away, and why

- Grid at lr=1e-4 — the LR was destroying the weights.
- A grid that ran three times concurrently — my `pkill` silently failed, the instances shared
  one results file and OOM'd the GPU.
- An early agent run that was shown dataset *names* (`credit-g`, …). LLMs have memorised those
  tables (Bordt et al. 2024); the agent must see statistics only. Fixed, rerun.

## What is *not* in this repo

`paper/numbers.tex`, `knobs.tex`, `main.pdf` are deliberately absent. They are generated from
`stage1.jsonl` by `emit.py`. An earlier draft built the PDF from placeholder values; those
artifacts are gitignored so a stale number can never be mistaken for a result.

## Layout

| file | what it does |
|---|---|
| `pipeline.py` | the closed loop: audit prior → agent revises → LoRA → TabArena → error analysis → repeat |
| `analysis2.py` | C2ST two-sample test: is the prior's distribution distinguishable from real tables, and *what gives it away* |
| `prior_audit.py` | first-pass audit (medians per axis) |
| `priortrain.py` | the 19 prior knobs, `PriorDataset` construction, LoRA injection, training loop |
| `stage1.py` | the four arms (`pretrained` / `base` / `random` / `agent`) and the DEV→TEST protocol |
| `tabarena.py` | TabArena suite, DEV/TEST split, task cache |
| `sweep_adapt.py`, `sweep_full.py` | LR / adapter sweeps (DEV-selected) |
| `agent.py` | a separate inference-time agent (context/features/data purchase under a credit budget) |
| `llm.py`, `llm_server.py` | LLM backend; experiments run on an A100 and call back to the laptop's `claude` CLI through an SSH reverse tunnel, cached by prompt hash |

## Protocol (frozen before results existed)

TabArena, 36 classification tasks TabICL can run. DEV 12 / TEST 24, interleaved by size.
The agent sees DEV summary statistics only — anonymised, no rows, no names. TEST is scored
once. Significance: paired differences per (task, seed), bootstrap CI + Wilcoxon. `adult` and
other memorised tables are excluded from anything the agent can see.
