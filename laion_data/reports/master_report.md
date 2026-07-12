# Master report — re:vision reproduction of Doerig et al. (2025)

*High-level visual representations in the human brain are aligned with large language models*,
Nature Machine Intelligence 7:1220–1234.

**Status:** M0–M10 implemented, tested, committed. Brain-side figures in progress.
**Date:** 2026-07-13

---

## 1. The headline

The paper is **more reproducible than its own plan suggested, and less reproducible than the
paper suggests.**

- **More**, because the authors published far more than the paper describes: the NSD data is
  on a public S3 bucket, and every one of the 22 trained checkpoints ships the *training code
  and the ground-truth hyperparameter dump inside it*. The reproduction plan budgeted 10–40
  GPU-days to retrain a network it believed could not even be instantiated. It can be
  instantiated, and it does not need retraining.
- **Less**, because the code that produced the published figures contains an unseeded random
  crop applied at *test* time. The authors' own Fig. 4c–4e activations are therefore not
  reproducible from one run to the next — not by us, and not by them.

## 2. What has been verified against the paper

| Quantity | Paper | Ours | Verdict |
|---|---|---|---|
| Shared test images \|D-515\| | 515 | **515** | ✅ exact |
| 3×-seen image counts | 10,000 / 6,234 / 5,445 | **10,000 / 6,234 / 5,445** | ✅ exact |
| RSA splits per subject | 100 / 62 / 54 | **100 / 62 / 54** | ✅ exact |
| RDM upper-triangle length | 4,950 | **4,950** | ✅ exact |
| Caption embedding matrix | (73000, 768), 224 MB | **(73000, 768), 224 MB** | ✅ exact |
| **C11** — MPNet word-order insensitivity | **r = 0.91** ± 0.03 | **r = 0.9121** ± 0.0012 | ✅ **mean reproduced**; s.d. does not (see §4) |
| Per-vertex z-score | mean 0, s.d. 1 | mean 4.2e-06, s.d. 1.000000 | ✅ exact |

## 3. The five blocking gaps — all closed

Full detail in `reports/modules/M0_gap_resolution.md`.

| Gap | Resolution |
|---|---|
| MI-32 RCNN timesteps | **6** (`hparams.txt`) |
| MI-33 vNet channels | **[32,32,64,64,128,128,256,256,512,512]**, kernels [7,7,5,5,3,3,3,3,1,1], pool at 3/4/9 |
| MI-15 ceiling arithmetic | **plain division**, ROI analyses only — searchlight maps are *never* ceiling-corrected |
| MI-17 / X3 ridge granularity | **one shared scalar fraction** for the whole model — neither the paper's reading nor the plan's |
| MI-05 beta I/O | `/300` for volume only; fsaverage `.mgh` unscaled; z-score per session per vertex, then average the 3 repeats |

X6 is resolved and the plan's prediction was exactly right: layer 9 has 1024//2 = **512**
channels → `GlobalAvgPool2D` → the 512-d pre-readout. Tapping the 768-d readout breaks
Figs. 4c–4e, so we assert `shape[1] == 512`.

## 4. Discrepancies found between the paper and the code that produced it

These are the substantive findings. Each is a defensible criticism, not a nitpick.

### 4.1 The published error bar on C11 does not describe what it says it describes
We reproduce the mean (0.9121 vs 0.91) but get s.d. = 0.0012, not 0.03. A between-participant
s.d. of 0.03 is **not achievable** by this pipeline: all 8 subjects' model RDMs are built
from the same 73k embedding matrix and differ only in which images each subject saw. A
per-image reading *does* give s.d. = 0.0365 — but its mean is 0.894. The published mean and
s.d. appear to be computed over different units of analysis.

### 4.2 The ridge fraction is a single global scalar
`FracRidgeRegressorCV` → `GridSearchCV` → `r2_score(multioutput='uniform_average')` collapses
~327k vertices into one R² and applies the winning fraction to every vertex. The paper says
the fraction was chosen to best predict "each embedding feature", which is incoherent when
the targets are voxels; the plan defaulted to per-voxel. **Both are wrong**, and a per-voxel
reimplementation will not reproduce the paper's numbers. The frac grid also runs to **1.05**,
not the stated 1.00.

### 4.3 An unseeded random crop is applied at test time
`tf_dataset_helper_functions.py:32-49` takes a *random* square crop of *random* size on every
split including test, with no seed — while the paper describes "the largest square crop".
Every activation-extraction run therefore yields different features and different RDMs. This
is a first-order reproducibility defect in the figure carrying the paper's headline claim.

### 4.4 The two strongest baselines in Fig. 4e were fed mis-preprocessed inputs
`task_helper_functions.py:274` returns `np_batch` instead of the resized `out_batch`, so the
resize is dead code: **Alexnet and ResNet50 (brainscore) received 128×128 inputs in [−1,1]
with no ImageNet normalisation.** Fig. 4e is the evidence for C17, the paper's headline
data-efficiency claim, and this bug handicaps two of its main competitors.

### 4.5 Model-vs-model tests are unpaired on paired data
`nsd_roi_analyses_figure.py:198` uses `scipy.stats.ttest_ind` to compare two models measured
on **the same 8 subjects**. We report both the released test and the correct paired test.

### 4.6 Undocumented training details
Ground truth from `hparams.txt`: `batch_size: 256` (the paper says 96), `clip_norm: 500`
(mentioned nowhere), and the cosine loss is applied at **all 6 timesteps** (deep supervision,
mentioned nowhere). Confirmed as stated: `optim_epsilon = 0.1`, 10⁷× the Adam default — it is
real, and anyone who "fixes" it trains a different optimizer.

### 4.7 The noise ceiling's numerator and denominator use different data
Numerator: mean over ~100 disjoint 100-image sub-RDMs of the full pool. Denominator: one
correlation on the full 515×515 RDM. Different image sets, different RDM sizes. Corrected
values can exceed 1 and are not a clean "fraction of explainable variance". We report a
matched-515 variant alongside.

## 5. Corrections to the reproduction plan

- **Space**: ROI RSA, encoding and decoding run on the **fsaverage surface**, not the volume.
  The plan asserted the opposite.
- **Nouns/verbs** are averaged **per word** (one forward pass each), not concatenated;
  **category words** are concatenated. Opposite conventions. Measurable in the embedding norms.
- **C12 is verifiable** — the "other LLMs" are named in the code.
- **Fig. 3a/3c is reproducible** — the OOV decisions the paper calls manual and unlogged are
  encoded in a 232-entry map in `word_lists.py`. We publish the full OOV log the original lacks.
- **The shared-1000 images** are a scattered set from `sharedix` (ids 2951–72949), not the
  1,000 lowest ids.

## 6. What remains

| | |
|---|---|
| **The only external blocker** | The **Supplementary Information**. Needed to settle X1 (the CORnet-S exception ROI: the body text says lateral, the figure caption says parietal — they cannot both hold) and X7 (Supp. Fig. 5 is cross-referenced four incompatible ways). This likely needs human institutional access. |
| In flight | 8-subject beta build; Figs. 1c/1d, 2a, 3 |
| Next | M11 decoding (needs the 3.1M-caption GCC dictionary); M15–M17 (needs the 40 GB NSD stimuli) |

## 7. Highest-value extensions identified so far

1. **H5 — retrain the category arm with BCE.** Ground truth confirms the control is trained
   with *cosine distance on a sigmoid output*, which is unusual and may handicap it. C16 and
   C17 both rest on that control. This is the one experiment that could overturn the paper's
   central comparison, and it is now cheap because the architecture and hyperparameters are
   fully recovered.
2. **Quantify the unseeded-crop defect.** Run Figs. 4c–4e with a deterministic centre crop
   and with the released random crop; the gap bounds how much of the result is preprocessing
   noise.
3. **Fix the Fig. 4e preprocessing bug** and re-rank the 14 models. If Alexnet/ResNet50 move
   up materially, C17 weakens.
4. **H1 — the sub-RDM bias.** Every RSA number is a mean over 100×100 sub-RDMs with no
   published bias analysis. Compute a full RDM for one subject/ROI and measure the bias.
