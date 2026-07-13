---
name: experiment-design
description: Pre-register a single experiment before it runs — hypothesis, falsification criterion, controls, and the failure modes that would fake the result. Use when the user proposes running something, before launching any job, or when asked to design an experiment, ablation, or control.
when_to_use: Trigger on "let's run", "design an experiment", "what's the control", "how do we test whether", "can we ablate", before any experiment launch.
argument-hint: [experiment or hypothesis]
allowed-tools: Read, Grep, Glob, Bash(nvidia-smi*), Bash(git status*)
---

Design the experiment: $ARGUMENTS

Write the design **before** the code, and write it so that it can lose. Produce this specification:

0. **Question** — which permanent question in `QUESTIONS.md` this moves, and how. A design that maps to none of them is out of scope; say so and stop.
1. **Hypothesis** — one falsifiable sentence about representation structure. Not "we investigate whether X"; rather "X holds / does not hold."
2. **Prediction** — the number, sign, and rough magnitude the hypothesis implies, stated before we look.
3. **Falsification criterion** — the observation that would make us reject it, fixed in advance. Choose the statistical test now, not after seeing the data.
4. **Null** — the control that could produce our effect by accident and must not. Pick from, and justify: untrained weights of the same architecture, shuffled labels, permuted rows, dimension-matched random projection, input statistics alone. **A design with no null is rejected.** If the null fires, the pipeline is broken — that is not confirmation of a large effect.
5. **Variance budget** — how many seeds, and what seed spread would swallow the predicted effect. If we cannot afford enough seeds to separate signal from seed, say so now rather than after.
6. **Instrument risk** — which metric (CKA / RSA / CCA / linear probe / retrieval), which centering, kernel, regularization, normalization, and noise ceiling; and how those choices could manufacture the effect. Name at least one alternative metric under which the result must also survive, and commit to it now.
7. **Failure modes that would fake this result.** Be specific to this pipeline, and consult the ones that have already burned this lab:
   - A dead dependency degrading a model to chance while still exiting 0 (a revoked API key already cost this lab one retracted result).
   - Weights that were never actually loaded; the wrong checkpoint, layer, or pooling.
   - A test artifact in the control condition rather than a real effect in the treatment.
   - Multiple comparisons without correction; the metric or seed chosen after seeing results.
   - Leakage between fit and evaluation splits.
   - A noise ceiling that was never estimated, so "low" and "as good as possible" are indistinguishable.
   For each: the assertion or probe that would catch it *inside the run*, not afterward in review.
8. **What gets recorded** — the run directory, the git SHA, the argv, the seed, the checkpoint hash, the layer, the environment, the exit code, and the raw per-item arrays needed to re-derive the headline number without re-running the job.
9. **Cost** — wall clock, GPU (GPU 1 only), and whether it blocks anything else. Prefer the run that can kill the hypothesis in an hour over the run that can support it in a week.

If any of steps 0-7 cannot be filled in, say so and stop. An experiment whose falsification criterion or null you cannot state is not ready to run, and running it will produce a number that means nothing.

Output the specification. Do not launch the job.
