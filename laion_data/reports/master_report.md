# Master report — re:vision reproduction of Doerig et al. (2025)

*High-level visual representations in the human brain are aligned with large language models*,
Nature Machine Intelligence 7:1220–1234.

**Status:** M0–M12 + M15 implemented, tested, committed. Figs. 1 and 3 reproduced.
**Date:** 2026-07-13 (revised after adversarial audit)

---

## 1. Headline

The paper is **more reproducible than its own reproduction plan predicted, and less
reproducible than the paper itself implies.**

- **More**, because the authors published far more than they describe: NSD is on a public S3
  bucket, and every one of the 22 trained checkpoints ships its *training code and
  ground-truth hyperparameter dump inside it*. The plan budgeted 10–40 GPU-days to retrain a
  network it believed could not even be instantiated. Neither was necessary.
- **Less**, because the code that produced the published figures applies an **unseeded random
  crop at test time**. The authors' own Fig. 4c–4e activations are not reproducible from one
  run to the next — not by us, and not by them.

## 2. Reproduced against the paper

| Quantity | Paper | Ours | Verdict |
|---|---|---|---|
| Shared test set \|D-515\| | 515 | **515** | ✅ exact |
| 3×-seen image counts | 10,000 / 6,234 / 5,445 | **exact, all 8 subjects** | ✅ |
| RSA splits per subject | 100 / 62 / 54 | **exact** | ✅ |
| RDM upper-triangle length | 4,950 | **4,950** | ✅ |
| Per-vertex session z-score | mean 0, s.d. 1 | 4.2e-06, 1.000000 | ✅ |
| **C11** — word-order insensitivity | r = 0.91 | **0.9121** | ✅ mean; s.d. does not (§4.1) |
| **C1** — LLM ↔ higher visual cortex | ventral/lateral/parietal ≫ EVC | **0.271 / 0.211 / 0.155 vs 0.087** | ✅ |
| **C2** — encoding ≈ inter-participant agreement | "approaches it in all ROIs" | **exceeds it** in ventral (1.24×), lateral (1.35×), parietal (1.31×) | ✅ |
| **C6, C7, C8, C10** — Fig. 3 orderings | full captions beat every control | **hold in all 4 ROIs** | ✅ |

## 3. The five blocking gaps — all closed

| Gap | Resolution |
|---|---|
| MI-32 RCNN timesteps | **6** (`hparams.txt`) |
| MI-33 vNet channels | `[32,32,64,64,128,128,256,256,512,512]`, kernels `[7,7,5,5,3,3,3,3,1,1]`, pool at 3/4/9 |
| MI-15 ceiling arithmetic | plain division; **ROI analyses only** — searchlight maps are never ceiling-corrected |
| MI-17 / X3 ridge granularity | **one shared scalar fraction** — neither reading in the paper |
| MI-05 beta I/O | `/300` for volume only; z-score per session per vertex, then `nanmean` the 3 repeats |

X6 confirmed empirically: the pre-readout tensor is exactly **(None, 512)**, and the released
network emits **6 outputs — one per timestep** (undocumented deep supervision).

## 4. Findings

### 4.1 C11's published error bar does not describe what it claims
Mean reproduces (0.9121 vs 0.91); s.d. is 0.0012, not 0.03. A between-participant s.d. of
0.03 is **unreachable** by this pipeline — all 8 subjects' model RDMs come from one 73k
embedding matrix. A per-image reading gives s.d. 0.0365 but mean 0.894. The published mean
and s.d. appear to be computed over different units of analysis.

### 4.2 The ridge fraction is inert
R² varies by only **2.7e-04 to 6.7e-04** across the entire 20-fraction grid, for every
subject. Every selected `frac` lands on 0.05 by numerical noise. The paper's "20 fractions,
5-fold CV" is choosing among numerically indistinguishable options — which **deflates MI-17**,
rated a top-6 risk.

### 4.3 An unseeded random crop is applied at test time
`tf_dataset_helper_functions.py:32-49` takes a *random* crop of *random* size on every split
including test, with no seed — while the paper says "the largest square crop". First-order
reproducibility defect in the figure carrying the headline claim.

### 4.4 The two strongest Fig. 4e baselines were fed mis-preprocessed inputs
`task_helper_functions.py:274` returns `np_batch`, not the resized `out_batch` — the resize is
dead code. Alexnet and ResNet50 received 128×128 inputs in [−1,1] with **no ImageNet
normalisation**. Fig. 4e is the evidence for C17, the paper's headline data-efficiency claim.

### 4.5 Model comparisons use an unpaired test on paired data
`nsd_roi_analyses_figure.py:198` uses `ttest_ind` on the same 8 subjects measured under both
models, **contradicting the paper's own Methods**, which describe a paired test. It is the
only pairwise test in the release and it renders the Supp. Fig. 12 p-matrices the paper cites.
**Their test cannot be trusted.**

⚠️ **See `reports/modules/M6_M7_M8_M9_fig3.md` §0 for a claim we RETRACTED here.** We initially
concluded the paper's three Fig. 3 exceptions were artifacts of this test. An adversarial audit
refuted it: the released test on *our* numbers produces **five** non-significant results, two of
which the paper reports as significant. Our per-subject values are therefore not the authors',
and the counterfactual is untestable. The retraction stands as a methods lesson — we had tested
only the three comparisons the paper flagged, never the control.

### 4.6 Undocumented training details
Ground truth from `hparams.txt`: `batch_size: 256` (paper says 96), `clip_norm: 500` (mentioned
nowhere), cosine loss applied at **all 6 timesteps**. Confirmed as stated: `optim_epsilon = 0.1`
— 10⁷× the Adam default, and real.

### 4.7 The noise ceiling mixes image sets
Numerator: mean over ~100 sub-RDMs of the full pool. Denominator: one correlation on the full
515×515 RDM. Corrected values can exceed 1. We report a matched-515 variant alongside.

## 5. Risks the reproduction DEFLATED

A risk register written from the paper alone must assume the worst wherever the paper is
silent. Three of its top risks were wrong:

| Risk | Plan's rating | Measured |
|---|---|---|
| **R11 / MI-12** | "Fig. 3a/3c not reproducible **by construction**" | The 232-entry OOV map **is** in the release. Reproducible. |
| **R6 / MI-17** | "changes the regularization of **every voxel**" | R² flat to 4 decimals. Nearly inert. |
| **R12 / H1** | "**not** an unbiased estimator; may legitimately disagree" | Bias ≤ **0.7 %** on the full 49,995,000-pair RDM. Unbiased. |

## 6. Bugs found in OUR pipeline (by adversarial audit)

All fixed. Each produced wrong-but-plausible output, not an error:

1. **float32 → 100 % NaN.** The mean-centred MPNet design matrix has condition number 3.5e19;
   `ols = (Uᵀy)/s` divides by a 6e-19 singular value. Now float64.
2. **Two drivers called a stale ridge signature** — Python would have silently misbound the
   arguments and fit on garbage.
3. **The one test pinning our ridge to the library was broken** by that signature change, so
   MI-17 faithfulness was unverified while the suite stayed green.
4. **`sum/3` ≠ `np.nanmean`.** A vertex NaN in one session became NaN outright. Recovered
   5,670 / 80,664 / 205,016 / (subj08) cells — and it hit **only EVC**, exactly where the
   contested comparisons live.
5. **`baseball-bat` compound override never fired** (it *is* in the fastText/GloVe vocab),
   changing 1,373/73,000 category embeddings.
6. **The "seeded" random crop was not seeded** — `tf.image.random_crop` uses TF's global RNG.

## 7. Remaining

| | |
|---|---|
| **Only external blocker** | The **Supplementary Information** — needed for X1 (body text says the CORnet-S exception is *lateral*; the figure caption says *parietal*) and X7. Likely needs institutional access. |
| In flight | Fig. 3 regeneration with both data fixes |
| Next | Fig. 2a/2b (drivers ready); Figs. 4b–4e (needs the 40 GB stimulus file) |

## 8. Highest-value extensions

1. **H5 — retrain the category arm with BCE.** Ground truth confirms the control is trained
   with *cosine distance on a sigmoid*, which is unusual and may handicap it. C16 and C17 both
   rest on that control. Now cheap, since the architecture is fully recovered.
2. **Fix the Fig. 4e preprocessing bug and re-rank the 14 models.** If Alexnet/ResNet50 move
   up, C17 weakens.
3. **Quantify the unseeded-crop defect** — run Figs. 4c–4e with a deterministic centre crop
   and with the released random crop; the gap bounds how much of the result is noise.
