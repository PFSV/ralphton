# Module classification

Every module in `reports/reproduction_plan.md`, classified into the four requested
categories. **This classification is post-M0**, and M0 changed it substantially: three
modules the plan treated as hard-blocked are now ready, and one 10–40 GPU-day module left
the critical path entirely.

**Date:** 2026-07-13

---

## Summary

| Class | Modules | Count |
|---|---|---|
| **1. Ready to implement** | M0 ✅, M1 ✅, M2 ✅, M3 ✅, M4 ✅, M5 ✅, M6 ✅, M7 ✅, M8 ✅, M9 ✅, M10 ✅, M12, M15, M16, M17 | 15 |
| **2. Blocked by missing information** | M18 (partial), M19 (partial) | 2 |
| **3. Requires long GPU training** | M14 — **no longer on the critical path** | 1 |
| **4. Depends on unfinished modules** | M11, M13 | 2 |

✅ = implemented, tested and committed in this session.

---

## 1. Ready to implement

| Module | Why ready | Status |
|---|---|---|
| **M0** Gap resolution | NSD S3 is public; the release repo clones; the checkpoints ship their own training code | ✅ **DONE** — all 5 blocking gaps closed |
| **M1** NSD ingest + `N-BETAZ` | No DUA gate on the download | ✅ **DONE** — z-score verified (per-vertex mean 4e-06, s.d. 1.000000); D-515 = 515; counts {10000, 6234, 5445} |
| **M2** COCO/category ingest | COCO 2017 annotations are open; NSD's own metadata names the release year | ✅ **DONE** — 73k captions, multi-hot (73000, 91), 80 columns reachable |
| **M3** MPNet embeddings | HF checkpoint, no gate | ✅ **DONE** — (73000, 768) |
| **M4** Text variants | M2 + M3 | ✅ **DONE** — **C11 reproduced: mean r = 0.9121 vs published 0.91** |
| **M5** fastText/GloVe/multi-hot | vectors are public; the OOV map is recoverable from the release | ✅ **DONE** (fastText); GloVe in flight |
| **M6** ROI patterns | M1 + the `streams` labels (6 KB) | ✅ **DONE** |
| **M7** RDM + 100-image sampler | M6 + any model | ✅ **DONE** — 4,950-length upper triangle; splits 100/62/54, disjoint, paired |
| **M8** Noise ceiling | MI-15 resolved (plain division) | ✅ **DONE** — as-released **and** matched-515 variants |
| **M9** Group stats | MI-21/22 resolved | ✅ **DONE** — released (unpaired) **and** correct (paired) tests |
| **M10** Fracridge encoding | MI-17 resolved (one shared fraction) | ✅ **DONE** — `select_frac` proven equivalent to `FracRidgeRegressorCV` |
| **M12** Functional contrasts | The 15 sentences are in the release, verbatim | Ready — needs M10's fitted model |
| **M15** Activation extraction | **Weights are published** — no training needed | Ready — needs NSD stimuli (40 GB) |
| **M16** Frozen-readout probe | Same | Ready — needs M15 |
| **M17** Competitor zoo | Same, plus public checkpoints | Ready — needs M15 + the 13 competitor models |

## 2. Blocked by missing information

| Module | Blocker | Consequence |
|---|---|---|
| **M18** (control analyses) | **X7** — Supp. Fig. 5's content is cross-referenced four incompatible ways and cannot be determined from the main text. The **SI is still not in the repository.** | E10f (cross-participant encoding) cannot be scored against its published figure. The *other* sub-analyses (E10c, E10d, E10e, E10g, E10h) are **unblocked** — notably **MI-27 is resolved** (the "other LLMs" are named in the code), so **C12 is verifiable after all**, contrary to the plan. |
| **M19** (figures + verdict) | **X1** — the CORnet-S exception ROI (lateral vs parietal) is self-contradictory in the paper and can only be settled against Supp. Fig. 20. | C17's precise wording cannot be scored. Every other claim can. |

**The SI is now the *only* external blocker in the entire project.** Everything the plan
listed as blocking is closed. This is the one item that may need human intervention
(institutional access to the Nature SI PDF).

## 3. Requires long GPU training

| Module | Original estimate | Revised |
|---|---|---|
| **M14** RCNN/ResNet training | 10–40 GPU-days + ~1 GPU-week (ecoset) | **Not required for reproduction.** All 20 RCNN seeds and both ResNet50s are published (14.5 GB). Retraining is now an *extension*, and the most valuable one is hidden assumption **H5** — the category arm is trained with cosine-distance-on-a-sigmoid rather than BCE (confirmed from ground-truth `hparams.txt`), which may handicap the control that claims C16/C17 rest on. |

## 4. Depends on unfinished modules

| Module | Depends on | Note |
|---|---|---|
| **M11** Decoding + caption retrieval | M10 (shared ridge core) + the 3.1M-caption GCC dictionary (a large download + ~1.5–3 GPU-h to embed) | Ready to build once M10 has run. Note MI-17 does **not** bite in this direction — "best fraction per embedding feature" is coherent when the 768 features *are* the targets. |
| **M13** COCO∖NSD training set | Only needed to *retrain* (M14), which is no longer required | Deprioritised. The exclusion filter is **not in the release** (X2: 48,236 vs 48,238 is unresolvable from the code). |

---

## What changed versus the plan, and why it matters

1. **M14 left the critical path.** The plan's single largest compute item (10–40 GPU-days)
   is unnecessary: the authors published the weights. The plan could not know this because
   it (correctly, per its own rules) refused to read code it had not been given.
2. **Three "blocked" modules became ready.** M15/M16/M17 were gated behind M14; they are
   now gated only on downloading images.
3. **C12 moved from "unverifiable" to "verifiable"** — the other LLMs are named in
   `get_name2file_dict.py:64-67`.
4. **Fig. 3a/3c moved from "irreproducible by construction" to "reproducible"** — the OOV
   decisions the paper describes as manual and unlogged are in fact encoded in a 232-entry
   map in `word_lists.py`.
5. **The SI is the only remaining external blocker,** and it blocks strictly less than the
   plan assumed: six claims were thought SI-only; after M0, only C3 (via X7) and the exact
   wording of C17 (via X1) actually require it.
