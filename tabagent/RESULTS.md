# Results

Everything here is measured. Nothing is estimated, smoothed, or filled in.

> **Retraction (2026-07-12).** A full-scale run of all three arms was briefly reported in
> `NEXT.md`. Its `agent` and `anchored` arms were **not agent results** — the OpenAI key had
> been revoked, `llm.ask()` swallowed the 401 and returned `""`, and the loop silently
> substituted random priors while still labelling itself `agent`. Those numbers are withdrawn;
> see `NEXT.md` for the full accounting. **The results below are unaffected** — they were
> produced while the LLM was live (verified against the response cache and the run logs), and
> they contain real agent proposals and real LLM-authored generator code. The failure mode has
> been fixed at the root: the LLM now fails loudly, and a run that cannot reach its agent
> refuses to start.

## The setup

TabICL is a tabular foundation model pre-trained on a prior over random structural causal
models — a synthetic distribution, chosen by its authors, frozen when the checkpoint ships.
We ask whether an LLM agent can repair that prior after the fact: diagnose how it differs
from real data, author replacement priors (config **and** generator code), LoRA-adapt the
released checkpoint to them, and win on a real benchmark.

- **Backbone**: released `tabicl-classifier-v2`, never re-pre-trained. LoRA on the ICL
  transformer's FFN — 590K of 28.1M parameters (2.1%).
- **Benchmark**: TabArena, the suite TabICL's own authors evaluate on. 36 classification
  tasks it can run. **DEV 12** (agent sees summary statistics only — never rows, never task
  names, and only the tasks' *validation* rows) / **TEST 24** (never seen; scored once).
- **Control**: `random` draws the same number of priors from the same knob space with the
  same GPU seconds. The only difference is whether an LLM proposes them.

## 1. The prior really is nothing like real data

A classifier separates prior-sampled datasets from real TabArena tables **perfectly**.

```
C2ST AUC = 1.000   (permutation p = 0.000;  0.5 would mean indistinguishable)
```

What gives it away, ranked by the classifier's own feature importance:

| property (median) | TabICL's prior | real TabArena |
|---|---|---|
| column cardinality (normalised) | 1.000 — every value unique | 0.008 — discrete |
| skew | 0.115 — symmetric | 2.199 — heavily right-skewed |
| fraction of categorical-like columns | 0.000 | 0.430 |
| fraction of heavy-tailed columns | 0.000 | 0.473 |
| linear-model AUC (how learnable) | 0.558 — near chance | 0.778 |
| features | 67 | 22 |

Real tables are skewed, discrete, heavy-tailed and *learnable*. The prior's are none of those.

## 2. The LLM writes a real generator, and it closes the gap

Given only those numbers — never a real row — the agent authored a `realize(X, y, rng)`
that rewrites the SCM's marginals: per column it samples one of seven rank-preserving maps
(quantile bins 26%, integer counts 16%, log-normal 20%, power tails 12%, censoring 12%,
binary thresholds 8%, linear 6%), plus a column-redundancy pass. All monotone, so the SCM's
causal signal survives.

```
distribution gap (mean KS over the axes a generator can move)
   0.593  →  0.415        (−30%)
```

## 3. Closing the gap makes the model WORSE

This is the finding.

```
round 0:  gap 0.616  →  downstream −0.0097
round 1:  gap 0.458  →  downstream −0.0117
          (the agent closed the gap, and the model got worse at real tables)
```

And on held-out TEST (24 tasks, scored once, seed 0):

| | TabArena TEST AUC | vs released | tasks won |
|---|---|---|---|
| released checkpoint | **0.8171** | — | — |
| `random` (same budget, no LLM) | 0.8142 | −0.0028 | 8/24 |
| `agent` (LLM-authored priors) | 0.8115 | −0.0056 | 7/24 |
| `anchored` (TabICL's prior kept, agent's added) | 0.8111 | −0.0060 | 7/24 |

**Every arm loses to leaving the checkpoint alone. The agent loses to random search.**

`anchored` was the last hypothesis standing. Every other arm *replaced* the released prior,
and the thing we kept measuring was that its breadth is load-bearing — so we kept it in the
mixture as an anchor and only *added* the agent's priors on top. It did not help either
(−0.0060). The failure is not that we threw the good prior away. Adapting to a revised prior
does not work here, full stop.

## 4. Why: the unrealism is domain randomisation, not a bug

The same pattern held under every adapter setting we swept (paired test, n=72, TabArena TEST,
3 seeds):

```
pretrained  0.8079      base (LoRA on TabICL's own prior)  0.8079   p = 0.85
                        random prior search                0.8079   p = 0.61
```

Adaptation is either inert (lr 1e-5) or destructive (lr 1e-4: −0.006; 1500 steps: −0.010).
Only one setting ever helped — a high LoRA scale (alpha 128), which lets the adapter act
through a *small* B and so does not trample the pre-trained weights: **+0.0023 DEV**.

Read together with §3, the explanation is the one from sim2real: TabICL's prior is a
*randomised simulator*. Its tables are implausible on purpose, and a real table is just one
more sample inside that wide envelope. Making the prior realistic **narrows** it, and the
model loses the coverage it was relying on. Exactly as a robot trained in a photorealistic
but narrow simulator breaks on the real world, while one trained under wild randomisation
does not.

This is a direct, quantitative data point in an open argument: TabForestPFN found unrealistic
priors work; Real-TabPFN found real data helps; Mitra argued diversity beats fidelity. Our
result sides with diversity — and, unlike those, it *measures* the realism gap (C2ST) and
then shows closing it is harmful.

## 5. What we have not shown

- One backbone (TabICL), one adapter family (LoRA on the ICL transformer's FFN). TabPFN v2
  may behave differently; its prior is differently constructed.
- The final runs are small — 2 rounds, 150 steps per prior, one seed on the headline table.
  Enough to establish direction, not to bound an effect size. The one setting that ever
  helped (alpha 128, +0.0023 DEV) was never replicated across seeds.
- We adapt; we never re-pre-train. A prior can only be *repaired* here, not replaced from
  scratch, so we cannot say what a TFM pre-trained on a realistic prior from step zero would
  do. That is the experiment this result argues is worth running — and argues will fail.
- The agent optimises against DEV tasks' validation rows. A larger search would eventually
  overfit them; ours is too short to have done so, which is also why the effect sizes are
  small.

## 6. What this cost

Roughly six hours on one A100, and four mistakes worth recording because each one produced a
result that looked real:

- **lr 1e-4 was destroying the pre-trained weights**, not improving the prior. Every arm lost
  ~0.006 AUC for that reason alone. Swept and fixed (1e-5), and the losses vanished — leaving
  the flat null of §4.
- **Three grid runs ran concurrently** after a `pkill` silently failed; they shared one
  results file and OOM'd the GPU. Discarded.
- **C2ST saturates.** At AUC 1.000 it has no gradient — a single mismatched axis pins it
  there, and one of ours (`n_rows`) is an artefact of our own row cap. Optimising against it
  taught the agent nothing. Replaced with a graded per-axis KS distance.
- **The safety filter rejected the agent's own valid code** — it banned the substring
  `import`, and the model had written `import numpy as np` in a header. Three of five rounds
  were thrown away before we noticed. **We then fixed it in one file and not the other**: the
  identical filter was still live in `agent.py`, silently discarding every feature the
  inference-time agent proposed. In the logs this is indistinguishable from an LLM with no
  ideas. Fixing it turned `caafe` on `diabetes` from +0.0000 (0 features) into +0.0076 (2
  features), instantly.
- **A revoked API key turned the agent into random search, silently, for a whole full-scale
  run.** `ask()` caught the 401 and returned `""`; the caller read that as "no proposal" and
  fell back to a random prior — while still reporting the arm as `agent`. Nothing crashed and
  the numbers looked plausible. Retracted above.

Three of these five are the same mistake wearing different clothes: **a component failing
quietly, and the silence being read as a result.** The bad exception handler, the over-strict
filter, and the dead key were all caught only by reading logs that had been printing the
problem the whole time. Every fallback is now loud or fatal.

## Reproducing

```bash
python analysis2.py 0                      # the C2ST diagnosis (§1)
python close_gap.py --rounds 6             # LLM writes the generator, graded by KS (§2)
python agent_loop.py --arm agent  --rounds 3 --k 3   # the loop (§3)
python agent_loop.py --arm random --rounds 3 --k 3   # the control
python sig.py                              # paired significance (§4)
```
