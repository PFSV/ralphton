# M11 — Figure 2b: decoding captions from brain activity

**Status: REPRODUCED. Claim C5 holds.**
**Date:** 2026-07-13

---

## 1. Result

Brain → MPNet embedding → nearest of **3,334,173** Google Conceptual Captions.
The ridge is the same core as the encoding model, with X and y swapped and the design matrix
restricted to the `streams` visual ROIs (67,696 vertices).

| | |
|---|---|
| **Mean decoding r** | **0.607** |
| 5–95% mass | **[0.398, 0.773]** — the paper's Fig. 2b axis is **0.3–0.7** ✅ |
| Caption noise ceiling | **0.753** |
| Fraction of ceiling reached | **~81%** |

Per subject: 0.629 / 0.618 / 0.590 / 0.592 / 0.643 / 0.617 / 0.617 / 0.548.

## 2. The reconstructions

Retrieved from 3.3M candidate captions, none of which was ever paired with a brain:

| true caption | decoded from brain |
|---|---|
| *"A person in a wetsuit surfing on a turquoise wave."* | **"person on beach with a surfboard."** (subj01)<br>**"person surfing on a surf board."** (subj05) |
| *"a coach and a table in a home living room"* | **"a detail of living room."** (subj01)<br>**"a white dining room, spotted walls, small dining table in the kitchen"** (subj05) |
| *"A man on skis next to a small child"* | **"person in the snow with person"** |
| *"There are two red double decker buses in a road."* | **"in this photo, vehicles are parked."** |

The decoder recovers the **scene semantics** — surfing, living room, snow, vehicles — without
reproducing the wording. This is exactly what C5 claims, and it reproduces.

## 3. Two findings

### The ridge fraction matters here, unlike in encoding
Selected fractions: **0.10 – 0.21**, varying by subject. Compare the encoding model, where the
fraction was **0.05 for all 8 subjects** and R² varied by only 3e-04 across the whole grid.

So the regularization is doing real work in the decoding direction and essentially none in the
encoding direction. **MI-17 (the shared-vs-per-target fraction question) is therefore a live
concern for Fig. 2b and a non-issue for Figs. 1c/1d.** The risk register treated it as one
undifferentiated risk.

### A predicted bug that did not bite
`decoding_extra_analyses.py:219` re-initialises `these_corrs = []` *inside* the caption loop,
so the released caption ceiling is caption-4-only rather than the mean over the 5. We
implemented both. They give **the same value to three decimals (0.753)**.

We had predicted the bug would inflate the ceiling's variance ~5×. On the *mean* — which is
what the Fig. 2b ceiling line draws — it makes no difference. The bug is real but inert.
This is the third prediction this reproduction made that measurement did not support (see
also: the `nanmean` fix recovering zero vertices, and the `baseball-bat` fix moving bars by
0.002).

## 4. Engineering note

The decoding design matrix is the **brain**: (9,485 images × 67,696 vertices). An economy SVD
of that is the wrong decomposition — `torch.linalg.svd` sat at 0% GPU, and precomputing five
fold-SVDs OOM'd a 40 GB A100 (`Vh` alone is 5.1 GB in float64).

For p ≫ n the Gram matrix `X Xᵀ` is only n × n and its eigendecomposition gives `U` and `s`
directly; predictions go through `W = (X_new Xᵀ) U / s`, so `V` is never formed
(`ridge._svd_wide`). That path exposed a trap the narrow path never hits: mean-centring makes
one singular value **exactly zero**, so `s²/(s²+α)` evaluates `0/0 = NaN` at α = 0 and silently
poisons every prediction. `select_frac` survived it (its `nan_to_num` masked it) while
`fit_predict` returned all-NaN. Both are now guarded and pinned by
`test_wide_gram_path_matches_library`.

Retrieval streams the 3.3M-caption dictionary through the GPU in chunks; materialising it as
float32 and centring it costs ~20 GB of transient and was OOM-killed.
