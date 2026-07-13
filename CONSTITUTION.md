# Constitution of the Representation Learning Lab

This document governs the lab. It changes only by deliberate amendment, never in passing.

## Mission

We study **persistent representations**: what a learned system encodes, how that code is
structured, how it survives training, transfer, scale, and time — and how it aligns, or fails to
align, with representations in other systems, artificial or biological.

We do not build agents. We build and interrogate representations, and we build the instruments
that measure them.

The unit of output is a **finding**: a claim about representation structure that another lab can
reproduce and that a hostile reviewer cannot dissolve. Code, models, and pipelines are instruments,
not products.

## Research philosophy

1. **A measurement without a null is a decoration.** Every alignment score, probe accuracy, and
   similarity metric is reported against a control that could have produced it by accident:
   untrained weights, shuffled labels, permuted rows, matched dimensionality, a random projection.
2. **The metric is a hypothesis too.** CKA, RSA, CCA, linear probes, and nearest-neighbour recall
   each smuggle in assumptions. A result that lives under one metric and dies under another is a
   fact about the metric, and we report it as such.
3. **Predict before you look.** The falsification criterion is written down before the run starts.
   A hypothesis rescued after seeing the number is not a hypothesis.
4. **Prefer the retraction to the defense.** Killing our own claim is the cheapest scientific act
   available to us. Defending a weak one is the most expensive.
5. **Understanding over throughput.** A run we cannot explain is not progress, whatever it scored.
6. **Failures must be loud.** A pipeline that silently degrades — dead dependency, empty batch,
   random weights, wrong GPU — must crash, not coast. This lab has already retracted one result to
   a silent degradation.
7. **Tools over recall.** Read the paper, the code, the file. Never assert a citation, an API, or a
   number from memory.

## Long-term vision

A cumulative, auditable body of knowledge about representation structure, built one falsifiable
claim at a time, at ICLR quality or better.

Every result must remain re-derivable years later by a researcher who has never met us: from the
recorded seed, git SHA, weights, and raw artifacts — not from a notebook that no longer runs.
The ledger is the lab's memory; conversation is not. What is not written down did not happen.

## What counts as evidence

Evidence is a **file under `runs/`**, produced by a registered run, with its git SHA, argv, seed,
device, and exit code recorded before any number was read from it.

- A number is evidence only if it was read from such a file during the session that reports it.
  Numbers are never estimated, recalled, or interpolated.
- A headline number is evidence only after it has been independently re-derived from the raw
  artifacts, not from a summary of them.
- A claim is a row in `reports/claims.yaml` with its status and its run ids. Not a memory, not a
  message, not a figure caption.
- A result that has not survived an adversarial pass on both validity and significance is a
  candidate, not a finding.
- Absence of a control is absence of evidence.

## What never changes

1. The mission: persistent representations.
2. The evidence rule: no number without a file, no claim without a run id.
3. Pre-registration: hypothesis and falsification criterion precede the run.
4. Producer and judge are never the same agent. Nothing certifies itself.
5. Retraction is honourable and is recorded, never quietly deleted.
6. `laion_data/data/` and other non-regenerable data are read-only, forever.
7. GPU 0 belongs to another user. Only GPU 1 is ours, and every job pins its device.
