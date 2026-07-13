# Review — per-claim verdicts

Doerig et al. 2025, *High-level visual representations in the human brain are aligned with
large language models*, Nature Machine Intelligence 7:1220–1234.

**Date:** 2026-07-13. Claim ids follow `reports/paper_structure.md` (C1–C20).

---

## Verdict summary

| | count |
|---|---|
| **REPRODUCED** | 8 (C1, C2, C5, C6, C7, C8, C10, C11) |
| **PARTIALLY REPRODUCED** | 1 (C4 — people/places yes, food no) |
| **PENDING** (data in flight) | 6 (C13–C18) |
| **UNVERIFIABLE without the SI** | 3 (C3, C9, C12) |
| **NOT ATTEMPTED** | 2 (C19, C20 — aggregate/discussion) |

---

## Reproduced

| Claim | Statement | Result |
|---|---|---|
| **C1** | LLM caption embeddings match brain RDMs across higher visual cortex | ✅ ventral 0.271, lateral 0.211, parietal 0.155 vs EVC 0.087 — higher visual cortex aligns 2–3× more than EVC |
| **C2** | A linear encoding model from LLM embeddings predicts voxel activity broadly, "approaching inter-participant agreement in all ROIs" | ✅ and then some: it **exceeds** IPA in ventral (1.24×), lateral (1.35×), parietal (1.31×). EVC 0.63×, as expected for a region driven by low-level features. Per-vertex max r = 0.78–0.89 |
| **C5** | Scene captions are reconstructable from brain activity alone | ✅ mean r = 0.607, 5–95% [0.398, 0.773] (paper axis 0.3–0.7), ~81% of the caption ceiling (0.753). Decoded captions are semantically correct |
| **C6** | LLM category-word embeddings beat multi-hot and word embeddings | ✅ in all 4 ROIs (paired test) |
| **C7** | Full-caption embeddings beat category-word embeddings, in all ROIs | ✅ in all 4 ROIs |
| **C8** | Full captions beat noun-only and verb-only | ✅ in all 4 ROIs |
| **C10** | Whole-caption embeddings beat averaged single-word embeddings | ✅ in all 4 ROIs |
| **C11** | MPNet is largely insensitive to word order (r = 0.91) | ✅ **0.9121**. The published **s.d. of 0.03 does not reproduce** under any reading (see §Discrepancies) |

## Partially reproduced

| Claim | Result |
|---|---|
| **C4** — the encoding model reproduces category-selective tuning (people/places/food) | ⚠️ **People − Places holds**: People > Places in midventral (+0.351), midlateral (+0.213), lateral (+0.206) — FFA/OFA/EBA/pSTS territory — and Places > People in parietal/midparietal (OPA). Survives FDR.<br>**Food − People does not.** **Zero** vertices survive FDR, and ventral is +0.001, essentially nothing, where the paper predicts food-selective ventral cortex. |

C4 is the one figure in the paper with **no multiple-comparison correction**. Applying the
paper's own standard correction thins People−Places by **58×** (41,666 → 714 vertices) and
erases Food−People entirely (32,652 → 0).

## Unverifiable without the Supplementary Information

| Claim | Blocker |
|---|---|
| **C3** — the encoding model generalizes across participants | SI-only (Supp. Fig. 5, whose content is itself contested — conflict X7) |
| **C9** — adjectives/adverbs/prepositions align poorly | SI-only (Supp. Fig. 9) |
| **C12** — results are not specific to MPNet | SI-only (Supp. Fig. 11). **Note: the plan called this permanently unverifiable because the other LLMs are never named. They ARE named in the code** (`get_name2file_dict.py:64-67`), so C12 becomes verifiable as soon as the SI is available for comparison. |

## Pending (implemented, data in flight)

C13–C18 all depend on the RCNN activations, which need the 39.6 GB NSD stimulus file. The
drivers are written and the 22 released checkpoints are downloaded and verified (6/6 top-1
retrieval on the sanity images). **C17's precise wording cannot be scored at all until X1 is
resolved** — the body text and the Fig. 4 caption disagree about where the single
non-significant CORnet-S comparison lives (lateral vs parietal), and they cannot both hold.

---

## Discrepancies between the paper and the code that produced it

| # | Finding |
|---|---|
| 1 | **C11's error bar does not describe what it says.** We reproduce the mean (0.9121 vs 0.91) but get s.d. = 0.0012, not 0.03. A between-participant s.d. of 0.03 is **unreachable** by this pipeline: all 8 subjects' model RDMs come from one 73k embedding matrix. A per-image reading gives s.d. 0.0365 — but its mean is 0.894. The published mean and s.d. appear to be computed over different units of analysis. |
| 2 | **The ridge fraction is inert in the encoding direction.** R² varies by only 2.7e-04–6.7e-04 across the entire 20-fraction grid, for every subject, so every selected `frac` lands on 0.05 by numerical noise. In the **decoding** direction it does real work (0.10–0.21). MI-17 is therefore a live concern for Fig. 2b and a non-issue for Figs. 1c/1d. |
| 3 | **The ridge fraction is a single shared scalar**, selected by `r2_score(multioutput='uniform_average')` — neither the paper's "per embedding feature" (incoherent when the targets are voxels) nor a per-voxel rule. The frac grid also runs to **1.05**, not the stated 1.00. |
| 4 | **An unseeded random crop is applied at test time** (`tf_dataset_helper_functions.py:32-49`), while the paper says "the largest square crop". The authors' own Fig. 4c–4e activations are not reproducible run to run. |
| 5 | **The two strongest Fig. 4e baselines were fed mis-preprocessed inputs.** `task_helper_functions.py:274` returns `np_batch`, not the resized `out_batch` — so Alexnet and ResNet50 received 128×128 inputs in [−1,1] with **no ImageNet normalisation**. Fig. 4e is the evidence for C17, the headline data-efficiency claim. |
| 6 | **Model comparisons use an unpaired test on paired data** (`ttest_ind` on the same 8 subjects, `nsd_roi_analyses_figure.py:198`), contradicting the paper's own Methods. It is the only pairwise test in the release and it renders the Supp. Fig. 12 p-matrices the paper cites. **Their test cannot be trusted.** (See the retraction in `reports/modules/M6_M7_M8_M9_fig3.md` §0 — we do *not* claim to know what a correct test would have done to *their* numbers.) |
| 7 | **Undocumented training details.** Ground truth from the checkpoints' `hparams.txt`: `batch_size: 256` (the paper says 96), `clip_norm: 500` (mentioned nowhere), and the cosine loss is applied at **all 6 timesteps** (deep supervision). Confirmed as stated: `optim_epsilon = 0.1`, 10⁷× the Adam default — it is real, and anyone who "fixes" it trains a different optimizer. |
| 8 | **The noise ceiling mixes image sets.** Numerator: mean over ~100 sub-RDMs of the full pool. Denominator: one correlation on the full 515×515 RDM. Corrected values can exceed 1. |

## Risks the reproduction DEFLATED

A risk register written from the paper alone must assume the worst wherever the paper is
silent. Three of its top risks were simply wrong, and measuring cost less than worrying:

| Risk | Plan's rating | Measured |
|---|---|---|
| **R11 / MI-12** | "Fig. 3a/3c not reproducible **by construction**" | The 232-entry OOV map **is** in the release. Fully reproducible; we publish the OOV log the original lacks. |
| **R6 / MI-17** | "changes the regularization of **every voxel**" | R² flat to 4 decimals in the encoding direction. Nearly inert. |
| **R12 / H1** | "**not** an unbiased estimator; a correct full-RDM reimplementation may legitimately disagree" | Bias ≤ **0.7 %** against the full 49,995,000-pair RDM. Unbiased. |

## Predictions this reproduction made that measurement did not support

Recorded because they are as informative as the confirmed ones:

1. The `sum/3` → `nanmean` fix would move the EVC comparisons. **It recovered zero vertices.**
2. The `baseball-bat` compound fix would shift the Fig. 3a control bars. **It moved them 0.002–0.003.**
3. The release's caption-ceiling bug would inflate the ceiling's variance ~5×. **It gives the same value to three decimals (0.753).**

## What still needs a human

**The Supplementary Information.** It is the only external blocker left. It is needed to
settle **X1** (the body text says the CORnet-S exception is in *lateral*; the Fig. 4 caption
says *parietal* — they cannot both hold) and **X7**, and to verify C3, C9, C12.
