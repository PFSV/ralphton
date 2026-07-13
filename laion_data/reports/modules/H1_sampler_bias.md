# H1 — Is the 100-image sub-RDM sampler biased?

**Status: ANSWERED. The sampler is essentially unbiased. Risk R12 can be struck.**
**Date:** 2026-07-13

---

## The question

Every RSA number in the paper is a mean over 100×100 sub-RDMs (4,950 pairs each), never the
full RDM. The reproduction plan lists this as hidden assumption **H1** and rates it risk
**R12**:

> *"the 100-image-split RSA is **not** an unbiased estimator of the full-RDM RSA, and the
> paper gives no bias analysis"* … *"a 'correct' full-RDM reimplementation may legitimately
> disagree with the paper."*

That is a real concern for anyone replicating this work: if the sampler were biased, a
replicator computing the full RDM would get different numbers and wrongly conclude they had
made an error. The paper never tests it. So we did.

## Method

For subj01 (10,000 images), compute the **complete** RDM — all **49,995,000** pairs — and
correlate it against the full model RDM. Compare that to the paper's estimator: the mean
over the 100 disjoint 100-image splits (4,950 pairs each).

`scipy.pdist` is far too slow at this size (single-threaded, ~10¹² operations per ROI). But
correlation distance is just a Gram matrix of z-scored rows,

    1 − corr(x_i, x_j) = 1 − ⟨z_i, z_j⟩,   z = (x − mean(x)) / ‖x − mean(x)‖

so the entire RDM is a single GEMM, which an A100 does in under a second.
(`experiments/h1_subrdm_bias.py`.)

## Result

| ROI | full-RDM r (49,995,000 pairs) | sampled r (mean over 100 splits) | bias | % error |
|---|---|---|---|---|
| early | 0.0877 | 0.0871 | −0.0007 | **−0.7 %** |
| ventral | 0.2708 | 0.2710 | +0.0002 | **+0.1 %** |
| lateral | 0.2110 | 0.2111 | +0.0002 | **+0.1 %** |
| parietal | 0.1542 | 0.1553 | +0.0011 | **+0.7 %** |

**Maximum deviation: 0.7 %, and the sign is inconsistent across ROIs** — exactly what one
expects from sampling noise rather than a systematic bias.

## Verdict

**The paper's tractability shortcut is sound.** The 100-image sub-RDM mean tracks the exact
full-RDM correlation to within a percent, so:

- **Risk R12 / hidden assumption H1 should be struck.** A full-RDM reimplementation will
  *not* legitimately disagree with the paper on this account.
- Anyone reproducing this work can use the sampler with confidence, and does not need to
  compute a 50-million-pair RDM to be safe.
- The authors were right not to worry about it — though they should have shown this, since
  it costs one GEMM.

## Context

This is the third risk the reproduction has **deflated** rather than confirmed:

| Risk | Plan's rating | Finding |
|---|---|---|
| **R11 / MI-12** | "Fig. 3a/3c not exactly reproducible, **by construction**" | The 232-entry OOV map **is** in the release. Reproducible. |
| **R6 / MI-17** | "changes the regularization of **every voxel**" | R² varies by only 2.7e-04–6.7e-04 across the entire fraction grid. The choice is nearly inert. |
| **R12 / H1** | "**not** an unbiased estimator … may legitimately disagree" | Bias ≤ 0.7 %. Unbiased. |

A risk register written from the paper alone systematically over-estimates risk, because it
must assume the worst wherever the paper is silent. Measuring is cheaper than worrying.
