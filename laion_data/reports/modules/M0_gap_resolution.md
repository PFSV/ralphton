# M0 — External asset acquisition and gap resolution

**Status: COMPLETE.** All five BLOCKING gaps are closed. The gate to M14 is open.
**Date:** 2026-07-13

---

## 1. What changed the project

Two discoveries reshape the entire plan:

**(a) The NSD data is public.** `s3://natural-scenes-dataset` is readable over plain HTTPS
with no credentials and no signed data-use agreement required for the download itself. The
plan treated this as a gated, days-long, DUA-blocked dependency. It is neither gated nor
slow. (The DUA still governs *use and redistribution* — see §6.)

**(b) The authors ship their training code *inside the model weights*.** The release repo
`visuo_llm` contains no model definition and no training loop; the plan correctly flagged
this as fatal for M14 (`MI-32`, `MI-33`). But every one of the 22 released checkpoints
(`s3://natural-scenes-dataset/other/blt_mpnet_weights/`, 14.5 GB total) contains a
`_code_used_for_training/` directory holding `blt_vNet.py`, `layers/blt.py`,
`setup_model.py`, `task.py` (the training loop) and — decisively — `hparams.txt`, the
ground-truth hyperparameter dump from the actual training run.

The consequence is large: **the 10–40 GPU-days of RCNN training in M14 is no longer on the
critical path.** All 20 RCNN seeds and both ResNet50s are downloadable. Retraining becomes
an *extension* (e.g. testing hidden assumption H5, the cosine-vs-BCE question), not a
prerequisite for reproducing Figs. 4b–4e.

---

## 2. The five BLOCKING gaps

| Id | Gap | Verdict | Locator |
|---|---|---|---|
| **MI-32** | RCNN recurrent timesteps | **6** | `hparams.txt: n_recurrent_steps: 6`. Corroborated independently in the analysis repo by `nsd_prepare_modelrdms.py:72` (`l, t = RCNN_LAYER // 6, RCNN_LAYER % 6`), `get_nsd_activations.py:168` (`'LayerNorm_Layer_9_Time_5'`), `plot_brains.m:35` (`DNN_TIMESTEPS = 1:6`). |
| **MI-33** | vNet channel counts / RF gradient | **RESOLVED** — see §3 | `_code_used_for_training/models/blt_vNet.py:45-62` |
| **MI-15** | Noise-ceiling correction arithmetic | **Plain division**, `r_model / r_ceiling(subj)` | `nsd_roi_analyses.py:162` — `all_corrs[subj][model][roi] / roi_noise_ceilings_per_sub[roi][subj]`. No sqrt, no subtraction. |
| **MI-17 / X3** | Encoding ridge-fraction granularity | **ONE SHARED SCALAR** for the whole model | `nsd_llm_encoding_model.py:104` `FracRidgeRegressorCV(...)` → `GridSearchCV` over `fracs` → `r2_score(multioutput='uniform_average')`. **Neither** the paper's "per embedding feature" **nor** the plan's "per voxel" default. |
| **MI-05** | NSD beta I/O, scale factor, fields | **RESOLVED** — see §4 | `nsd_get_data_light.py:154-202` |

### MI-15, in full

The ceiling is a leave-one-subject-out Pearson correlation of a subject's RDM against the
mean RDM of the other 7, computed on the shared **515** images, per ROI, per subject
(`nsd_roi_analyses.py:128-150`). Each subject's model correlation is then divided by *that
subject's own* ceiling.

Two facts the paper never states, both of which we must reproduce faithfully and then
criticise:
- **The searchlight maps are NEVER ceiling-corrected.** Only the ROI analyses are. So
  Figs. 1b/4c/4d live on a raw-r scale and Figs. 3/4e on a normalised one. The plan called
  this "the consistency trap"; the code confirms it exactly.
- **Numerator and denominator are computed on different data.** The numerator is a mean
  over ~100 disjoint 100-image sub-RDMs of the *full* NSD set; the denominator is a single
  correlation on the *full 515×515* RDM. Different image sets, different RDM sizes,
  different pair counts. Ratios can therefore exceed 1 and are not a clean "fraction of
  explainable variance". This is hidden assumption **H3**, and the code makes it concrete.

### MI-17, in full

This is the most consequential correction to the plan. `FracRidgeRegressorCV` wraps
`GridSearchCV`, whose default `scoring` calls the estimator's `score()` → `r2_score` with
`multioutput='uniform_average'`. That collapses ~327k voxels into **one scalar R²**, and
the single fraction maximising it is applied to **every voxel**. A per-voxel implementation
— which is standard NSD practice, and what the plan defaulted to — will *not* reproduce the
paper's numbers.

The frac grid is also not what the paper says. Paper: "0.05 to 1.00 in steps of 0.05".
Code (`nsd_llm_encoding_model.py:25-26`): `np.linspace(1/20, 1+1/20, 20)` = 0.05 → **1.05**,
step 0.05263. The largest fraction exceeds OLS.

---

## 3. The architecture (MI-33, X6)

`blt_vNet.py` with `divide_n_channels=2` ("half_channels"):

| | |
|---|---|
| Layers | 10 `BLTConvLayer` |
| Channels | `[64,64,128,128,256,256,512,512,1024,1024] // 2` = **`[32,32,64,64,128,128,256,256,512,512]`** |
| Kernels | `[7,7,5,5,3,3,3,3,1,1]` (this decreasing kernel size *is* the paper's "foveal receptive-field-size gradient") |
| Pooling | `MaxPool2D(2,2)` on the bottom-up input at layers **3, 4, 9** |
| Lateral | from `activations[t-1][n]` (same layer, previous timestep) |
| Top-down | from `activations[t-2][n+1]` — **note `t-2`, not `t-1`** |
| Last layer | layer 9 is **BL only** (no top-down input) |
| Readout | `GlobalAvgPool2D(activations[t][9])` → `Dense(classes)` |

**X6 — RESOLVED, and the plan's prediction was exactly right.** Layer 9 has `1024//2 = 512`
channels; global-average-pooling over space gives a **512-d pre-readout vector**. The
readout is `Dense(768)` (MPNet arm) or `Dense(91)` (category arm). The tensor to tap is
`layernorm_layer_9_time_5`. Tapping the 768-d readout breaks Figs. 4c–4e.

---

## 4. Ground-truth training hyperparameters

From `hparams.txt`. The mpnet and multihot checkpoints differ in **exactly two lines**:

```
< target_dataset_name: all_mpnet_base_v2_mean_embeddings     > target_dataset_name: img_multi_hot
< model_output_activation: no_model_output_activation        > model_output_activation: sigmoid
```

This **verifies the paper's controlled-comparison claim from ground truth**: same
architecture, same data, same seeds, differing only in objective and readout activation.
It also **confirms hidden assumption H5**: `embedding_loss: cosine` for *both* arms, so the
category control really is trained with **cosine distance on a sigmoid output, not BCE**.
The loss is `(tf.keras.losses.cosine_similarity(y_true, y_pred) + 1)` (`task.py:85-89`).

| Param | Value | vs. paper |
|---|---|---|
| `n_recurrent_steps` | 6 | not stated (MI-32) |
| `learning_rate` | 0.05 | ✅ matches |
| `optim_epsilon` | **0.1** | ✅ matches — 10⁷× the Adam default, and **stated, not a typo** (MI-31 confirmed) |
| `n_epochs` | 200 | ✅ matches |
| `n_warmup_epochs` | 10 | ✅ matches |
| `learning_rate_schedule` | cosine | ✅ matches |
| `image_size` | 128 | ✅ matches |
| `regularize` (L2) | 1e-06 | not stated (MI-23) |
| **`batch_size`** | **256** | ❌ **paper says 96 (RCNN) / 512 (ResNet)** — NEW CONFLICT, see §5 |
| **`clip_norm`** | **500** | ❌ **not mentioned anywhere in the paper** — NEW |
| `dropout_rate` | 0.0 | not stated |
| `use_mixed_precision` | 1 | not stated |

**Deep supervision (NEW, undocumented).** `task.py:91-95` builds a loss dict over *all*
`network_output_layers`, and `blt_vNet.py:109` emits an output at **every one of the 6
timesteps**. So the cosine loss is applied at all 6 timesteps, not just the last. The paper
never mentions this, and it materially changes what the network optimises.

---

## 5. New conflicts found (not in the paper, not in the plan)

| # | Conflict | Evidence |
|---|---|---|
| **X8** | **Batch size 256, not 96.** | `hparams.txt: batch_size: 256`. The paper states 96 for the RCNN. The released weights were trained at 256. `transfer_readout.py:29` uses 96 *for the probe* with the comment "same as we used to train the big models" — which the checkpoint contradicts. |
| **X9** | **`clip_norm: 500`** — gradient-norm clipping is applied (`task.py:113`) and appears nowhere in the paper. | `hparams.txt`, `task.py:112-114` |
| **X10** | **Deep supervision at all 6 timesteps** — undocumented. | `task.py:91-95`, `blt_vNet.py:109` |
| **X11** | **The searchlight brain RDM ignores the `rdm_distance` flag.** It is hardcoded to correlation distance in TF; the flag controls only the *model* RDM and the output path. Setting it to "cosine" would silently compare a cosine model RDM against a correlation brain RDM. | `tf_utils.py:63-65` vs `examples/searchlight_analyses.py:8` |

---

## 6. Silent gaps resolved (previously "will run; will differ")

| Id | Resolution | Locator |
|---|---|---|
| **MI-14** | **PAIRED.** Splits drawn once per subject, saved, reused for all models; the code *raises* if a non-mpnet model finds no saved splits. | `nsd_searchlight_main_tf.py:121-146` |
| **MI-03** | **UNRESOLVABLE.** No RNG seed exists anywhere in the repo. The original splits are unrecoverable. We seed ours and sweep. | (absence) |
| **MI-02** | Correlation distance, both sides. | `tf_utils.py:63`, `nsd_prepare_modelrdms.py:96` |
| **MI-01** | **YES** — encoding, decoding, RSA and the noise ceiling all call the same `load_or_compute_betas_average`. | 6 shared call sites |
| **MI-13** | **No voxel selection, no reliability threshold.** All vertices in the ROI mask; only NaN vertices dropped. | `roi_utils.py:30-44` |
| **MI-11** | The release passes no `normalize_embeddings`, but `all-mpnet-base-v2`'s `modules.json` ships a `Normalize` module, so encoder outputs *are* unit-norm. The per-image **mean** of 5 unit vectors is not (we measure ‖e‖ = 0.839 ± 0.050). | `embedding_models_zoo.py:76` |
| **MI-21** | **Raw r.** No Fisher-z anywhere (`arctanh` → 0 hits). | (absence) |
| **MI-22** | BH-FDR, α=0.05. Searchlight family = all vertices of one map. ROI family = the pairwise comparisons *within one ROI*; **the one-sample ROI tests are not corrected at all**. | `py_plot_brain_utils.py:52`, `nsd_roi_analyses_figure.py:204-213` |
| **MI-07** | **K = 91**, not 80. Verified: the list is ordered by COCO category id, and only 80 columns are ever set. | `word_lists.py:18-110`; `blt_mpnet/example.py:46` |
| **MI-27** | The "other LLMs" **are named**: `multi-qa-mpnet-base-dot-v1`, `all-distilroberta-v1`, `all-MiniLM-L12-v2`, `paraphrase-multilingual-mpnet-base-v2`, `paraphrase-albert-small-v2`, `paraphrase-MiniLM-L3-v2`, `distiluse-base-multilingual-cased-v2`, plus `GUSE_transformer`, `GUSE_DAN`, `USE_CMLM_Base`, `T5`. **C12 is therefore verifiable**, contrary to the plan. | `get_name2file_dict.py:64-67` |
| **MI-12** | The OOV handling is **not** unlogged: `word_lists.py:126-362` contains a 232-entry `verb_adjustments` misspelling map. **Fig. 3a/3c is therefore not "irreproducible by construction"** as the plan concluded. (Caveat: `noun_adjustments` is empty, and the verb map is applied even to MPNet, where it is unnecessary.) | `word_lists.py:126-362` |
| **MI-08** | fastText `crawl-300d-2M.vec`, GloVe `glove.840B.300d.txt`, both 300-d. | `examples/get_embeddings.py:15-16` |
| **MI-09** | NLTK Penn Treebank. Nouns = `NN`,`NNS` **only** (proper nouns excluded). Verbs = all six `VB*`. | `nsd_embeddings_utils.py:120` |
| **COCO year** | **2017.** Not a guess: NSD's own `nsd_stim_info_merged.csv` has a `cocoSplit` column with values `train2017` (70,051) / `val2017` (2,949). | NSD metadata |
| **X1** | **STILL OPEN.** The CORnet-S exception ROI (lateral vs parietal) needs Supp. Fig. 20. | — |

### Corrections to the reproduction plan's own assumptions

- **Space.** The plan asserts "Volume space, not fsaverage — fsaverage is used only for
  visualization". **Wrong.** Searchlight uses `func1pt8mm`; **ROI RSA, encoding and
  decoding all use `fsaverage` surface** (`roi_utils.py:65`, `nsd_llm_encoding_model.py`,
  `nsd_decode_llm.py`). This changes the data we must download and the shape of every
  brain-side array.
- **Nouns/verbs aggregation.** The plan inferred nouns/verbs were *concatenated* into one
  string. They are **averaged per-word** (each word gets its own forward pass). Category
  words *are* concatenated. The two conventions are opposite.

---

## 7. Bugs in the released code (carry to `reports/failure_modes.md`)

These are defects in the code that produced the published figures. Each one is a live
threat to the reproduction *and* a legitimate criticism of the paper.

1. **Unseeded random crop at test time.** `tf_dataset_helper_functions.py:32-49` applies a
   *random* square crop of *random* size (not the "largest square crop" the paper claims)
   on **every** split, including test. There is no seed. **Every activation-extraction run
   therefore yields different features, and hence different RDMs.** This is a first-order
   reproducibility defect in Figs. 4c–4e.
2. **The brainscore preprocessing function discards its own work.**
   `task_helper_functions.py:274` returns `np_batch`, not the resized `out_batch`. So the
   **Alexnet and ResNet50 baselines were fed 128×128 inputs in [-1,1] with no ImageNet
   normalisation** — a serious handicap to two of the strongest competitors in Fig. 4e,
   which is the figure carrying the paper's headline claim (C17).
3. **Caption noise-ceiling bug.** `decoding_extra_analyses.py:219` re-initialises
   `these_corrs = []` *inside* the caption loop, so the reported ceiling is caption-4-only,
   not the mean over the 5. This value is the ceiling line in Fig. 2b.
4. **`n_train_imgs` has 16 entries for 14 models** (`trainingImgs_vs_brainMatch_scatter.py:27-44`);
   `zip` silently truncates. Also, the Fig. 4e y-values are **hand-copied literals**, not
   recomputed.
5. **ROI model-vs-model tests use `ttest_ind`** (independent samples) on the *same 8
   subjects* measured under both models (`nsd_roi_analyses_figure.py:198`). An unpaired test
   on paired data — this is exactly the test a Nature reviewer would flag.
6. Two released scripts are broken on import (`make_noise_ceiling.py:7`,
   `decoding_extra_analyses.py:11` import a module that does not exist in the release).

Item 2 deserves emphasis: it means the competitor ranking in Fig. 4e — the evidence for the
paper's headline data-efficiency claim — was computed with a preprocessing bug that
disadvantages the two strongest supervised baselines.

---

## 8. Assets acquired

| Asset | Status | Size |
|---|---|---|
| `visuo_llm` release repo | ✅ cloned | 116 files |
| `blt_mpnet` inference repo | ✅ cloned | — |
| `blt_rdl_pipeline` (training) | ❌ **does not exist** (404) | — |
| Checkpoint training code + `hparams.txt` | ✅ downloaded (2 arms) | ~50 KB |
| NSD metadata (`expdesign`, `stim_info`, `responses.tsv` ×8, `streams` labels) | ✅ | 43 MB |
| NSD `fsaverage` betas, 8 subjects | ⏳ downloading | 315 GB |
| COCO 2017 annotations | ✅ | 800 MB |
| 22 trained checkpoints | ⬜ available, not yet fetched | 14.5 GB |
| Supplementary Information | ❌ **still missing** | — |

**The SI remains the one unresolved external dependency.** It is still required to settle
X1 and to verify C3, C9, C12, C13, C18. Everything else the plan listed as blocking is now
closed from the code.

---

## 9. Verified against the paper (no brain data needed)

| Check | Result |
|---|---|
| `\|D-515\| == 515` | ✅ **515** |
| 3×-seen image counts | ✅ **{10000, 6234, 5445}** exactly |
| RSA splits per subject | ✅ **100 / 62 / 54** exactly |
| Caption table | ✅ 73,000 images, 99.7% with exactly 5 captions |
| Multi-hot | ✅ (73000, 91), exactly 80 columns ever set |
| MPNet caption embeddings | ✅ (73000, 768), 224 MB — matches the plan's predicted size |
