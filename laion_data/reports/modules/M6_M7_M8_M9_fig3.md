# M6–M9 — Figure 3 reproduced, and the paper's three exceptions explained

**Status: REPRODUCED.** Claims C6, C7, C8, C10 all hold. And the paper's three reported
exceptions turn out to be artifacts of the wrong statistical test.
**Date:** 2026-07-13

---

## 1. Figure 3 — noise-ceiling-corrected RSA, mean over 8 participants

| model | early | ventral | lateral | parietal |
|---|---|---|---|---|
| **LLM (full captions)** | **0.163** | **0.350** | **0.404** | **0.358** |
| LLM (nouns only) | 0.137 | 0.305 | 0.308 | 0.306 |
| LLM (category words) | 0.109 | 0.238 | 0.265 | 0.244 |
| LLM (single words, averaged) | 0.113 | 0.239 | 0.257 | 0.251 |
| LLM (verbs only) | 0.048 | 0.134 | 0.167 | 0.121 |
| multi-hot categories | 0.074 | 0.209 | 0.249 | 0.183 |
| fastText (categories) | 0.089 | 0.202 | 0.207 | 0.190 |
| GloVe (categories) | 0.082 | 0.207 | 0.210 | 0.191 |
| fastText (all words) | 0.122 | 0.268 | 0.288 | 0.270 |
| GloVe (all words) | 0.112 | 0.275 | 0.282 | 0.272 |

Noise ceilings (LOSO on the shared 515): early 0.527, ventral 0.723, lateral 0.474,
parietal 0.377.

Values land inside the paper's published ranges (ventral 0.15–0.45, lateral 0.1–0.5,
parietal 0.1–0.4). Full captions win in **every** ROI.

## 2. Claim verdicts (paired t-test across 8 participants, BH-FDR within ROI)

| Claim | Comparison | early | ventral | lateral | parietal | Verdict |
|---|---|---|---|---|---|---|
| **C6** | LLM-categ > multi-hot | +0.035\*\*\* | +0.029\*\*\* | +0.016\*\* | +0.060\*\*\* | ✅ |
| **C6** | LLM-categ > fastText | +0.020\*\*\* | +0.036\*\*\* | +0.059\*\*\* | +0.054\*\*\* | ✅ |
| **C6** | LLM-categ > GloVe | +0.027\*\*\* | +0.031\*\*\* | +0.055\*\*\* | +0.052\*\*\* | ✅ |
| **C7** | captions > LLM-categ | +0.054\*\*\* | +0.112\*\*\* | +0.138\*\*\* | +0.115\*\*\* | ✅ |
| **C8** | captions > nouns | +0.026\*\*\* | +0.045\*\*\* | +0.096\*\*\* | +0.052\*\*\* | ✅ |
| **C8** | captions > verbs | +0.115\*\*\* | +0.216\*\*\* | +0.236\*\*\* | +0.238\*\*\* | ✅ |
| **C10** | captions > single-words | +0.049\*\*\* | +0.110\*\*\* | +0.146\*\*\* | +0.108\*\*\* | ✅ |

Every comparison is significant in every ROI. **The paper, however, reports three
exceptions** — and we initially found none.

## 3. The three exceptions are artifacts of an unpaired test on paired data

The release compares two models with `scipy.stats.ttest_ind`
(`nsd_roi_analyses_figure.py:198`) — an **independent-samples** test — applied to the *same
8 participants* measured under both models. The pairing is discarded, which throws away
exactly the within-subject variance that makes the comparison powerful.

Re-running our comparisons with the released (unpaired) test:

| Comparison | Paper reports | Paired test (correct) | Unpaired test (as released) |
|---|---|---|---|
| LLM-categ vs multi-hot, **lateral** | *fails in lateral* | **SIG** p = 0.0048 | **n.s.** p = 0.186 ✅ |
| LLM-categ vs fastText, **EVC** | *fails in EVC* | **SIG** p = 0.0001 | **n.s.** p = 0.201 ✅ |
| captions vs nouns, **EVC** | *fails in EVC* | **SIG** p = 0.0001 | **n.s.** p = 0.225 ✅ |

**All three of the paper's exceptions reproduce exactly — but only under the released,
incorrect test.**

Two conclusions, pointing in opposite directions:

**(a) This validates the reproduction at the finest available grain.** Matching a paper's
headline bars is weak evidence; matching its three idiosyncratic *failures*, and only when
adopting its exact statistical procedure, is strong evidence that our pipeline is
computing the same quantities the authors computed.

**(b) The exceptions are not real.** They are a consequence of `ttest_ind` on a
within-subject design. With the correct paired test, all three vanish. The authors
**under-sold their own result**: C6 and C8 hold without exception, not with two and one
respectively.

This is a correctable error that makes the paper's conclusions *stronger*, and it is the
kind of thing a Nature reviewer would be expected to catch.

## 4. Hidden assumption H3, made concrete

The noise-ceiling numerator and denominator are computed on different data: the numerator
is a mean over ~100 disjoint 100-image sub-RDMs of the full pool; the denominator is a
single correlation on the full 515×515 RDM. We therefore also compute a **matched-515**
variant (both quantities on the 515) and store it in `reports/figures/fig3_roi.json` under
`matched_515`, so the sensitivity of every bar to this choice is inspectable.

## 5. Deviations to keep in view

- The RSA split seed is unrecoverable (the release has no seed anywhere), so our splits are
  not the authors' splits. C11's seed sweep showed the statistics are seed-insensitive.
- MI-13 confirmed: no voxel selection, no reliability threshold — all ROI vertices, NaNs
  dropped.
- The correction is applied to Fig. 3 **only**. Figs. 1b/4c/4d are on a raw-r scale and
  must never be cross-compared with these numbers.
