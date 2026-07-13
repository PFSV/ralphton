# Permanent Research Questions

These are not projects. They are the questions the lab exists to answer, and they outlive any
paper, dataset, or model family we happen to be using this year. Every project must name the
question it attacks and the observation that would move it. A project that maps to none of these
is out of scope, however interesting.

Each question is stated so that it can be *wrong*: with the shape of evidence that would settle it
against us.

---

## Q1 — What is actually encoded?

Given a trained system, what information does its representation carry, and in what geometry —
linearly decodable directions, nonlinear manifolds, superposed features, or nothing at all beyond
the decoder we attached to it?

*Moves against us if:* the "structure" we report is recoverable from an untrained network of the
same architecture, or from the input statistics alone.

## Q2 — What survives?

Which parts of a representation persist across training time, seeds, scale, architecture,
fine-tuning, distribution shift, and modality — and which are contingent artifacts of one run?
Persistence is the lab's central object: a representation that does not survive is not a finding
about learning, it is a finding about a seed.

*Moves against us if:* seed variance alone spans the effect we attribute to scale, data, or
objective.

## Q3 — When do independently trained systems converge on the same code?

Different objectives, architectures, modalities, and substrates — including brains — sometimes
arrive at aligned representations. Under what conditions does this happen, and what exactly is
shared: the geometry, the features, the task-relevant subspace, or only the dimensionality?

*Moves against us if:* the measured alignment is explained by shared input statistics, matched
dimensionality, or a random projection baseline.

## Q4 — What does our instrument measure?

Alignment and probing metrics (CKA, RSA, CCA, linear probes, retrieval, noise ceilings) are
instruments with their own biases. Which conclusions in this literature are facts about
representations, and which are facts about the measurement?

*Moves against us if:* a claimed representational property reverses under an equally defensible
metric, preprocessing choice, or noise ceiling.

## Q5 — What causes a representation to take the form it takes?

Objective, data distribution, architecture, optimization, scale: which of these is doing the work?
We want interventional answers, not correlational ones — change one factor, predict the
representational consequence in advance, and check.

*Moves against us if:* the predicted change fails to appear, or appears equally under a control
intervention we believed was inert.

## Q6 — Does representational structure predict anything we care about?

Does a measured property of the representation predict downstream behaviour — generalization,
sample efficiency, robustness, transfer — better than the baselines already available (loss,
scale, task accuracy)?

*Moves against us if:* the structural measure adds nothing over a compute- or loss-matched
baseline.

---

## Standing obligations attached to every question

- Every reported effect is stated against an explicit null (untrained, shuffled, permuted,
  dimension-matched, random-projection).
- Every effect is reported with seed variance, or it is not reported.
- Every claim names the file under `runs/` it came from, and its row in `reports/claims.yaml`.
