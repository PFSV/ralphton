# Repairing the Prior

An LLM agent revises the synthetic pre-training prior of a tabular foundation model; LoRA
makes each candidate prior cheap to evaluate; held-out real tasks say whether it worked.

It does not work — and *why* it does not work is the result. Start with
[`RESULTS.md`](RESULTS.md) for the measured findings, [`NEXT.md`](NEXT.md) for the state of
play and the commands to pick the work back up.

## The idea

TabICL/TabPFN are pre-trained on a prior over random structural causal models. That prior is
frozen when the checkpoint ships and was chosen without any downstream task in view. Nobody
re-pre-trains a TFM to fix it — that costs a cluster.

So instead: expose the prior's generating configuration, let an agent edit it from downstream
failure, and LoRA-adapt the *released* checkpoint to each candidate (2.1% of parameters,
~2 min/candidate on one A100). Keep the best on DEV, report it on TEST.

The arm that decides the paper is **random search over the same knobs, same budget**. Without
it, any gain could just be search.

## What came out

A C2ST separates the prior's tables from real ones perfectly (AUC 1.000). The agent, given only
those statistics, writes a generator that closes 30% of the gap. The model then gets **worse**
— and so does random search, and so does keeping TabICL's own prior as an anchor. Every arm
loses to leaving the checkpoint alone. The unrealism is domain randomisation; narrowing it
destroys the coverage the model was relying on.

That result is at small scale (2 rounds, 150 steps per prior, seed 0). A full-scale run was
reported and then **retracted**: a revoked API key made `llm.ask()` return `""`, the loop read
that as "no proposal" and substituted a random prior while still calling itself `agent`. No arm
has tested LLM-designed priors at full scale. `RESULTS.md` and `NEXT.md` carry the accounting;
the silent fallbacks that caused it are gone (the LLM now raises `LLMDown`, and `preflight()`
proves it is alive before any GPU is spent).

The live hypothesis is the other half: don't repair the prior, **repair the context**. Same
agent, moved to inference time, on a frozen model. It won on `credit-g`, and it is running
across all 24 TEST tasks now — the first time at scale (`context_bench.py`). See `NEXT.md`.

## Layout

| file | what it is |
|---|---|
| `agent_loop.py` | the prior-repair loop: audit → agent proposes knobs + generator code → LoRA → TabArena → error analysis → repeat. Arms: `agent` / `random` / `anchored` |
| `context_bench.py` | **the live experiment**: the same agent at inference time, under a credit budget (1 credit = 1 real labelled row) |
| `agent.py` | that inference-time agent (features / context curation / synthesis / buying real rows) |
| `analysis2.py` | C2ST two-sample test — is the prior distinguishable from real tables, and *what gives it away* |
| `prior_audit.py` | first-pass audit of the released prior (medians per axis) |
| `close_gap.py` | the LLM writes the data generator; graded by per-axis KS distance |
| `realism.py` | compiles and sandboxes the LLM-authored `realize(X, y, rng)` |
| `priortrain.py` | the 19 prior knobs, `PriorDataset` construction, LoRA injection, training loop |
| `stage1.py` | the earlier arm protocol (`pretrained` / `base` / `random` / `agent`), superseded by `agent_loop.py` |
| `tabarena.py` | the benchmark: 36 TabICL-runnable classification tasks, DEV 12 / TEST 24, cached |
| `data.py` | OpenML tasks, the ctx/pool/val/test split, column anonymisation |
| `tfm.py` | frozen TabICL scorer + the credit cost model |
| `sweep_adapt.py`, `sweep_full.py` | LR / adapter / full-fine-tune sweeps, DEV-selected |
| `sig.py` | paired significance, one difference per (task, seed) |
| `llm.py` | LLM backend: falls through openai → tunnel → local `claude` CLI, raises `LLMDown` rather than returning `""`, `preflight()`s before any GPU is spent. Cached by prompt hash, so reruns are free and a crashed run resumes |
| `llm_server.py`, `tunnel.sh` | the reverse tunnel: serve the laptop's `claude` to the A100, and keep the tunnel alive across network drops |
| `emit.py` | writes `paper/numbers.tex` + `paper/fig_search.pdf` **from the results file** |
| `paper/` | ICML 2025 style, 4 pages |

**No number in the paper is typed by hand.** `main.tex` only ever references macros that
`emit.py` generates from the results file, and those generated artifacts (`numbers.tex`,
`knobs.tex`, `main.pdf`) are gitignored — a stale number can never be mistaken for a result.

## Reproducing

```bash
python analysis2.py 0                                # the C2ST diagnosis
python close_gap.py --rounds 6                       # LLM writes the generator, graded by KS
python agent_loop.py --arm agent  --rounds 3 --k 3   # the loop
python agent_loop.py --arm random --rounds 3 --k 3   # the control
python sig.py                                        # paired significance
python emit.py && (cd paper && ../tools/tectonic -X compile main.tex --outdir .)
```

Runs go on the A100 (`CUDA_VISIBLE_DEVICES=1`, GPU 0 belongs to someone else) and must be
detached — `setsid nohup ... < /dev/null &` — because the network drops. The LLM key is read
from `../.env` (gitignored); it is currently **revoked**, so runs go through the reverse tunnel
to the laptop's `claude` (`./tunnel.sh` locally, `TABAGENT_LLM_URL=http://127.0.0.1:8791` on the
server). A fresh key removes that dependency and is the first thing worth fixing. `NEXT.md` has
the traps already stepped in; read it before launching anything long.

## Note on the second agent (`agent.py`)

Built first, kept because it became the live hypothesis. At *inference* time, model frozen, no
training at all: the agent buys/derives/synthesises context under a credit budget. On
`credit-g` it reached test AUC 0.7789 from a 0.7652 baseline for 33 of 100 credits, spending
them on three world-knowledge features (`monthly_installment_burden`, `repayment_strain`,
`liquidity_adjusted_burden`) and 30 uncertainty-sampled real rows. TabICL never sees column
names; that is world knowledge doing work the prior cannot supply.
