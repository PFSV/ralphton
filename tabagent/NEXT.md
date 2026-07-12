# Where this stands, and what to do next

Read `RESULTS.md` first for the measured findings. This file is the handoff: what is running,
what is known, what is still open, and the exact commands to pick it back up.

## RETRACTED: the previous full-scale table was not an agent result

**An earlier version of this file reported a full-scale run of all three arms. Those numbers
were real, but two of the three arms were mislabelled, and every conclusion drawn from them
about *the agent* was wrong. Retracted in full.**

What happened: the OpenAI key in `~/ralphton/.env` was revoked partway through the project.
`llm.ask()` caught the 401, returned `""`, and the caller read an empty string as *"the agent
had no proposal"* and quietly substituted a random prior. Every round of the full-scale run
printed `agent produced nothing; falling back to random for this round` — and then reported
itself as `agent` anyway.

So, precisely:

- `agent` → **was pure random search.** Not an agent result.
- `anchored` → **was base + random priors**, not base + *agent* priors. Still a distinct arm,
  but not what its name claims.
- `random` → genuinely random. The only arm that was what it said.

**No arm tested LLM-designed priors at full scale.** The claim that "the agent followed the
evidence and widened the gap 0.646 → 0.666" was random noise with a story pasted on top.

The small-scale results in `RESULTS.md` §1–§4 are **unaffected** — the LLM cache
(`cache/llm/`) shows live `gpt-4.1` responses through 07:26, and those logs contain real
proposals with real reasons and real generator code. The key died after them. So the project's
core stands; only the full-scale table was fiction.

### The fix, which matters more than the numbers

A dead LLM must **stop** the experiment, not silently redefine it.

- `llm.ask()` now falls through backends (openai → tunnel → local CLI), retries, demotes a
  401'd key, and **raises `LLMDown`** if nothing answers. It can no longer return `""`.
- `agent_loop.py` **aborts** if the agent proposes nothing. It never substitutes random.
- `llm.preflight()` proves the LLM is alive with a PONG **before any GPU is spent**. A run
  that cannot reach its agent does not start.

A second bug of the same family was found in `agent.py`: the safety filter banned the
*substring* `import`, so every feature the LLM proposed was discarded before it ran — the logs
read as "the LLM had no good ideas" when it had never been allowed to have one. (The identical
bug was fixed in `realism.py` weeks earlier and never ported.) With it fixed, `caafe`
immediately added two working features to `diabetes` for +0.0076.

**The lesson to carry:** every silent fallback in this codebase was a lie generator. If a
component can fail, it must fail loudly or not at all.

## What is running right now

`context_bench.py` — the inference-time context agent across all 24 TabArena TEST tasks
(budget 100 credits, seed 0), on GPU 1. This is the first time it has ever run at scale.

```bash
ssh $GPU_HOST 'pgrep -af context_bench; cd ~/tabagent && grep -v 401 ctx.log | tail -20'
```

It needs the LLM. The OpenAI key is revoked, so it runs off a reverse tunnel to the laptop's
`claude` (`./tunnel.sh` locally, `TABAGENT_LLM_URL=http://127.0.0.1:8791` on the server).
**A fresh API key would remove that dependency and should be the first thing to fix.**

## The experiment now running (previously: written, never run)

`context_bench.py` — the other half of the story. Instead of repairing what the model *learned
from*, repair what it *sees*. The model is frozen; `fit()` loads a context, it does not train.
So the agent acts at inference time, under a credit budget where **1 credit = 1 real labelled
row**:

| action | price |
|---|---|
| derive a feature from what the column names MEAN | 1.0 (the LLM call) |
| curate which rows are in the context | 0.5 |
| synthesise rows to fill a diagnosed gap | 2.0 + 0.01/row |
| **buy a real labelled row** | **1.0/row** |

Arms: `raw` / `buy-unc` (active learning, no LLM) / `caafe` (LLM features only) / `agent`.

It already works on one task. On `credit-g`, budget 100:

```
0.7652 -> 0.7789   spending 33 of 100 credits on
  monthly_installment_burden      (val 0.6955 -> 0.7016)
  repayment_strain                (      0.7016 -> 0.7028)
  liquidity_adjusted_burden       (      0.7028 -> 0.7054)
  + 30 uncertainty-sampled real rows  ( 0.7054 -> 0.7225)
```

The LLM read the column names, understood that this is a credit table, and invented debt-burden
ratios. That is world knowledge doing work the prior cannot supply — TabICL never sees column
names.

Run it:

```bash
ssh $GPU_HOST 'cd ~/tabagent && CUDA_VISIBLE_DEVICES=1 setsid nohup \
  .venv/bin/python -u context_bench.py --budget 100 --seed 0 > ctx.log 2>&1 < /dev/null &'
```

It is resumable — each (task, arm) result is appended to `context_bench.jsonl` and skipped on
a rerun.

## Why the two halves belong in one paper

> **Don't repair the prior. Repair the context.**
>
> *Negative:* post-hoc repair of a TFM's synthetic prior fails, by every route we found —
> LLM-designed priors, random search, and keeping the original as an anchor. C2ST shows the
> prior is perfectly distinguishable from real data (AUC 1.000), and closing that gap makes
> things *worse*. The unrealism is domain randomisation; narrowing it destroys the coverage
> the model depends on.
>
> *Positive:* the same agent, moved to inference time, wins — because in an in-context learner
> the prior is a finished asset and **the context is the real handle**.

The negative is what makes the positive interesting. Without it, "an agent picks features" is
CAAFE. With it, it is an argument about where the leverage in a tabular foundation model
actually lives.

## Open questions, in the order worth attacking

1. **Does `context_bench` hold up across TabArena's 24 tasks?** One task is an anecdote. This
   is the only experiment that can turn the project positive, and it has never been run.
2. **Does prior repair ever cross zero with enough steps?** The trend is monotone: 150 steps →
   −0.0163; 1200 steps → −0.0060. Extrapolating is not evidence, but 5000 steps is one
   overnight run and would settle it.
3. **Is this TabICL-specific?** TabPFN v2's prior is built differently. If prior repair works
   there, the story is about TabICL, not about tabular foundation models.
4. **Semantic vs non-semantic split.** The context agent's whole edge is reading column names.
   It should therefore win on `credit-g` (`checking_status`, `purpose`) and do nothing on
   `phoneme` (`V1..V5`). That test isolates world knowledge from search, and `data.py` already
   has `anonymize()` for it.

## Traps already stepped in — do not repeat

- `lr=1e-4` destroys the pre-trained weights. It is not a prior problem, it is an optimiser
  problem. Use 3e-5 with **alpha=128** — a high LoRA scale lets the adapter act through a
  *small* B and leaves the backbone intact. alpha=32 costs −0.010 for the same training.
- `batch_size` must stay **4**. 8 trips a CUDA "invalid configuration argument" inside TabICL's
  flattened-batch SDPA; 16 OOMs at 38GB.
- The bottleneck is **CPU, not GPU** — SCM sampling plus the LLM's numpy realiser. The A100
  sits at 5–20% and raising the batch does not help. Parallelising 6 runs made it *slower*
  (they fight over cores). Three at a time is about right on 64 cores.
- **C2ST saturates at 1.000** and carries no gradient. Optimise the per-axis mean KS distance
  instead (`close_gap.gap_score`), and exclude axes the generator cannot move (`n_rows`,
  `n_features` — those are set by our own row cap, not by the prior).
- Launch long jobs with `setsid nohup ... < /dev/null &`. A plain `ssh 'bash script.sh'` dies
  with the network, and this network drops.
- Kill with `pkill -9 -f <name>` **and then verify** — a silent failure once left three grids
  running against one results file until they OOM'd the card.

## Layout

| file | what it is |
|---|---|
| `agent_loop.py` | the prior-repair loop: audit → agent proposes priors + generator code → LoRA → TabArena → error analysis → repeat. Arms: `agent` / `random` / `anchored` |
| `context_bench.py` | **the next experiment**: the same agent at inference time, under a credit budget |
| `agent.py` | that inference-time agent (features / context curation / synthesis / buying real rows) |
| `analysis2.py` | C2ST two-sample test — is the prior distinguishable from real tables, and *what gives it away* |
| `close_gap.py` | the LLM writes the data generator, graded by KS distance |
| `realism.py` | compiles and sandboxes the LLM-authored `realize(X, y, rng)` |
| `priortrain.py` | the 19 prior knobs, `PriorDataset`, LoRA injection, training loop |
| `tabarena.py` | the benchmark: 36 TabICL-runnable classification tasks, DEV 12 / TEST 24, cached |
| `sig.py` | paired significance, one difference per (task, seed) |

## Environment

- A100 box: `$GPU_HOST`, work in `~/tabagent`, venv at `.venv`, **use GPU 1**
  (`CUDA_VISIBLE_DEVICES=1`; GPU 0 belongs to someone else). Network drops — always detach.
- LLM: OpenAI, key read from `~/ralphton/.env` (gitignored). No SSH tunnel needed any more (the old
  `llm_server.py` + reverse-tunnel path is still there as a fallback, and it dropped calls
  every time the session died).
- Everything the LLM says is cached by prompt hash under `cache/llm/`, so reruns are free and
  a crashed run resumes where it stopped.
