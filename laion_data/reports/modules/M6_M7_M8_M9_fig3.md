# M6–M9 — Figure 3, and a retracted claim

**Status: Fig. 3 REPRODUCED. One earlier claim in this report was WRONG and is retracted below.**
**Date:** 2026-07-13 (revised after adversarial audit)

---

## 0. RETRACTION (read this first)

An earlier version of this report claimed:

> *"All three of the paper's exceptions reproduce exactly — but only under the released,
> incorrect test. The exceptions are not real."*

**That claim was confirmation-biased and is withdrawn.** A 5-agent adversarial audit
refuted it 3–2, and the refutation is correct.

**What I did wrong.** The paper reports three exceptions. I tested exactly those three
comparisons under the released (unpaired) test, found all three non-significant, and
declared a match. I never asked the obvious control question: *does the released test also
break comparisons the paper says hold?*

It does. Running the released test over **all** the comparisons in the paper's claim
families yields **five** non-significant results, not three:

| comparison | ROI | Δr | released (unpaired) p | paired p |
|---|---|---|---|---|
| LLM-categ > multi-hot | lateral | +0.016 | 0.186 | 0.005 |
| LLM-categ > fastText | early | +0.020 | 0.201 | 0.0001 |
| captions > nouns | early | +0.026 | 0.225 | 0.0001 |
| **LLM-categ > GloVe** | **early** | **+0.027** | **0.095** | **<0.0001** |
| **captions > fastText-words** | **early** | **+0.041** | **0.062** | **0.0001** |

The last two are comparisons the paper reports as **significant** (its word-embedding
exception is fastText-specific, and C10 is claimed "in all ROIs"). So the released test,
applied to *our* per-subject values, **contradicts the paper on two comparisons**.

**What that proves:** our 8 per-subject values are demonstrably **not** the authors'. The
counterfactual the claim rested on — "run the paired test on *their* numbers and *their*
three exceptions vanish" — is therefore untested, and untestable from the release
(`group_corrs.pkl` is not published).

**The simpler explanation, which I should have preferred:** the three exceptions are simply
**the three smallest effects** in their families (+0.016, +0.020, +0.026), and the two extra
ones we generate are the next smallest (+0.027, +0.041). *Any* low-power procedure breaks
the weakest effects first. This explains the "match" without needing the pairing story at
all.

## 1. What survives, and is defensible

Verified unanimously by every skeptic, and stated this way from now on:

> The release computes its model-comparison p-values with `scipy.stats.ttest_ind` on a
> within-subject design (`nsd_roi_analyses_figure.py:198`) — an independent-samples test on
> the same 8 participants measured under both models. This **contradicts the paper's own
> Methods**, which describe testing "the significance of the difference between model
> correlations", i.e. a paired test. It is the only pairwise test in the release
> (`ttest_rel` appears nowhere), it is the family that receives BH-FDR, and its output is
> what renders the Supp. Fig. 12 p-matrices the paper cites. **The released
> model-comparison test is invalid and underpowered.**
>
> In our data, the paper's three exceptions are the three weakest effects in their families
> and are exactly the ones this test kills. But because the released test *also* kills two
> comparisons the paper reports as significant, we cannot claim the authors' exceptions
> would vanish under a correct test — only that **their test cannot be trusted**.

Two further caveats the audit raised, both of which I accept:

- **The "rescue" is not robust.** Under the `matched_515` noise-ceiling variant, the
  *paired* test produces new exceptions the paper does not report (LLM-categ vs multi-hot:
  EVC p = 0.121, ventral p = 0.076). "The correct test makes the paper's claims hold without
  exception" is an artifact of one arbitrary noise-ceiling convention.
- **Power at n=8 is not meaningful.** The paired test declares 168/180 comparisons
  significant, including Δr = 0.007. Quoting p = 1e-9 at n = 8 is parametric fantasy — the
  exact sign-flip floor is 2/256 = 0.0078. Significance has stopped tracking meaningfulness.

## 2. Figure 3 — noise-ceiling-corrected RSA, mean over 8 participants

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

**Claim verdicts (paired test): C6, C7, C8, C10 all hold in all four ROIs.** The direction
and ordering of every bar reproduces. This part of the reproduction stands.

⚠️ These numbers are pending a re-run after two confirmed pipeline bugs (below) are fixed.

## 3. Confirmed bugs in OUR pipeline (found by the audit, not by us)

| # | Severity | Where | Defect |
|---|---|---|---|
| 1 | **critical** | `experiments/fig2b_decoding.py`, `fig2a_contrasts.py` | called the ridge with the pre-GPU 4-arg signature; would misbind silently and fit on garbage |
| 2 | **major** | `tests/test_modules.py` | the one test pinning our ridge to `FracRidgeRegressorCV` was broken by the same signature change — **MI-17 faithfulness was unverified** |
| 3 | **major** | `src/revision/nsd_data.py:260` | 3-repeat average is `sum/3`, not the release's `np.nanmean`. A vertex NaN in *one* repeat becomes NaN instead of the mean of its 2 valid repeats. Loses 10 / 126 / 350 / 37 vertices for subj01/05/06/08 — an **asymmetric, per-subject change to the vertex set** feeding the cross-subject t-tests. Our docstring asserting `sum/3 ≡ nanmean` was false. |
| 4 | **major** | `src/revision/wordvec.py` | `lookup()` checked the vocabulary *before* `CATEGORY_COMPOUNDS`, so the compound-mean override never fired for `baseball-bat` — which **is** in the fastText and GloVe vocabularies. The release unconditionally uses `mean(baseball, bat)`. This shifts the fastText/GloVe category bars — i.e. it moves one of the exact comparisons §0 turns on. |
| 5 | **major** | `src/revision/dnn.py` | `crop="random"` is not actually seeded: `tf.image.random_crop` takes its *location* from TF's unseeded global RNG. Our claim to have made the released behaviour reproducible was false. |

All are fixed; Fig. 3 is being regenerated.

## 4. Method note

The retraction in §0 is the reason the adversarial audit was run at all. A reproduction that
only tests the comparisons a paper flags will confirm whatever the paper says. The control
that mattered — "does this test also break things the paper claims work?" — costs one extra
loop and inverted the conclusion.
