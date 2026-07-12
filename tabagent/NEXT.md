# Where this stands, and what to do next

Read `RESULTS.md` first for the measured findings. This file is the handoff: what is running,
what is known, what is still open, and the exact commands to pick it back up.

## The one-line state

The **prior-repair** hypothesis is dead or dying — every arm loses to leaving the checkpoint
alone, and the mechanism (domain randomisation) explains why. The **context-repair** idea is
the live one: same agent, applied at inference time, already beat the baseline on `credit-g`
and has never been run at scale. That is the next experiment, and it is written and waiting.

## What is running right now

A full-scale prior run (`full.sh`): 3 arms × 3 seeds × 3 rounds × 3 priors × 1200 steps.
Seed 0 is done; seeds 1–2 are in flight. Check it:

```bash
ssh $GPU_HOST 'pgrep -af agent_loop; cd ~/tabagent && grep -hE "gap .*DEV|: 0\.|wins" full_*.log'
```

If it needs relaunching, detach it — a plain `ssh "bash full.sh"` dies with the network, and
this network drops:

```bash
ssh $GPU_HOST 'cd ~/tabagent && setsid nohup bash full.sh > full.out 2>&1 < /dev/null &'
```

### Seed 0, at full scale

Per-round DEV (validation rows of the 12 DEV tasks — the loop's only signal):

| arm | round 0 | round 1 | round 2 |
|---|---|---|---|
| `agent` | gap 0.646 → **−0.0112** | gap 0.666 → −0.0060 | gap 0.521 → −0.0067 |
| `random` | gap 0.672 → −0.0089 | gap 0.590 → −0.0046 | gap 0.492 → −0.0109 |
| `anchored` | gap 0.694 → −0.0063 | gap 0.633 → −0.0094 | gap 0.694 → −0.0085 |

Held-out TabArena TEST, 24 tasks, scored once:

| | TEST AUC | vs released | tasks won |
|---|---|---|---|
| released checkpoint | **0.8171** | — | — |
| `random` | 0.8139 | −0.0032 | 8/24 |
| `anchored` | 0.8137 | −0.0034 | **11/24** |
| `agent` | *(pending)* | | |

Four things worth keeping:

1. **The full budget halves the damage but does not cross zero.** 150 steps → −0.0060 on
   TEST; 1200 steps → −0.0032. The small runs were not hiding a win, they were exaggerating a
   loss.
2. **The realism/performance anti-correlation reproduces at full scale, across arms.** At
   round 0 the ranking by distribution gap is exactly the reverse of the ranking by DEV:
   `anchored` (gap 0.694, *least* realistic) is best; `agent` (gap 0.646, most realistic) is
   worst.
3. **The agent followed the evidence over the intuition.** Told "chase AUC, not the gap —
   closing the gap has been hurting", it *widened* the gap in round 1 (0.646 → 0.666) and
   improved (−0.0112 → −0.0060).
4. **`random` improves across rounds too**, and `random` has no feedback at all — more rounds
   are just more lottery tickets. Any "the agent learns across rounds!" claim has to beat
   that. This is exactly what the control is for.

`anchored` winning **11 of 24 tasks** while losing on the mean is the most interesting loose
thread: prior repair is not uniformly harmful, it is harmful *on average*. Which tasks it
helps, and what they have in common, is unexamined.

## The next experiment (written, never run)

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
