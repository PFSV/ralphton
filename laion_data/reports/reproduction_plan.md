# Reproduction Plan — Doerig et al. 2025, *High-level visual representations in the human brain are aligned with large language models* (Nature Machine Intelligence 7:1220–1234)

**Sources read for this plan (and only these):**
- `/home/snucsnl/ralphton/laion_data/reports/paper_structure.md` (whole file)
- `/home/snucsnl/ralphton/laion_data/reports/dependency_graph.md` (whole file, 806 lines)

**Date of analysis:** 2026-07-13
**Paper PDF:** `/home/snucsnl/ralphton/laion_data/paper/re_vision.pdf` (not re-read here; both source reports were built from it and are cited in its place)

**Scope.** This document is an implementation roadmap: a decomposition of the paper into independently testable modules, a topological build order, a module→claim map, a risk register, and a milestone schedule. It does **not** re-extract the paper (that is `paper_structure.md`), does **not** re-derive the computational DAG (that is `dependency_graph.md`), and contains **no code**. Every fact below is traceable to a section of one of the two reports; anything I need but neither report supplies is marked **UNKNOWN — needs verification** rather than guessed.

**Citation convention.** `[PS §N]` = `reports/paper_structure.md`, section N. `[DG §N]` = `reports/dependency_graph.md`, section N. Node ids (`N-BETAZ`, `N-RSASAMPLE`, …), pipeline ids (`E1`–`E10`), missing-info ids (`MI-xx`), conflict ids (`X1`–`X7`) and claim ids (`C1`–`C20`) are used exactly as those reports define them, so this plan can be cross-read against them without renaming.

**One structural fact drives the whole plan** [DG §0]: the paper is *one* brain-side artifact (NSD betas → session z-score → 3-rep average) crossed with *one* text encoder (MPNet), joined by *exactly two* mapping families (parameter-free RSA; fractional ridge), plus *one* ANN-training family whose outputs re-enter RSA. There is no third method. The module list below is a merge of that structure, not a per-figure enumeration.

---

## 0. Gate module (must precede everything)

### M0 — External asset acquisition and gap resolution

| Field | Content |
|---|---|
| **Purpose** | Close the five **BLOCKING** information gaps before any implementation begins, and obtain every licensed/gated asset. [DG §9.1] |
| **Inputs** | Supplementary Information for the paper (Supp. Figs. 1–20) — **not in this repository** [PS §12 preamble; DG §11]. Release code: `https://github.com/adriendoerig/visuo_llm`, Zenodo DOI `10.5281/ZENODO.15282176` v1.0 [PS §1 header; DG D20]. NSD data access at `naturalscenesdataset.org` [PS §1 header; DG D1]. Cited refs 63 (Mehrer et al., vNet/ecoset), 115 (Kietzmann et al., BLT recurrence), 54 (Rokem & Kay, fracridge) [DG §9.1]. |
| **Outputs** | (a) `docs/resolved_gaps.md` recording, with a code `file:line` or SI-figure locator for each: MI-32 (RCNN timesteps), MI-33 (vNet channel counts), MI-15 (noise-ceiling arithmetic), MI-17 (encoding ridge-fraction granularity), MI-05 (NSD beta int16 scale factor, HDF5 field names, voxel counts). (b) Adjudication of X1 (CORnet-S exception ROI: lateral vs parietal) against Supp. Fig. 20, and X7 (what Supp. Fig. 5 actually contains). (c) NSD data-use agreement signed; local mirror of `betas_fithrf_GLMdenoise_RR` @1.8 mm, 8 subjects. |
| **Dependencies** | None (root). External: NSD data-use agreement; COCO (CC-BY 4.0 annotations, images under Flickr terms); Google Conceptual Captions terms; the paper itself is CC-BY 4.0 [PS §1]. |
| **Implementation order** | **1** |
| **Success criteria** | (i) All five BLOCKING rows in [DG §9.1] have a resolved value with a locator, **or** are explicitly re-declared as still-missing with a documented default and a planned sensitivity sweep. (ii) X1 resolved to a single ROI. (iii) `betas_fithrf_GLMdenoise_RR` 1.8 mm present for all 8 subjects and readable. |
| **Explicit gate** | [DG §10] states: *"do not start step 9 [RCNN training] until the repo has been read for MI-32/MI-33/MI-15/MI-17"*. This plan enforces that gate at M14. |

---

## 1. Module specifications

Each module below has exactly the six required fields. Ranks are global and form the topological order in §2.

---

### M1 — NSD brain-data ingest and beta normalization (`D-NSD`, `D-515`, `N-BETAZ`)

| Field | Content |
|---|---|
| **Purpose** | Produce the single brain-side artifact that every downstream analysis consumes. [DG §2.1 merge M3] |
| **Inputs** | NSD `betas_fithrf_GLMdenoise_RR`, **1.8 mm volume preparation**, 8 subjects; NSD trial tables; NSD brain masks. Shape per subject: `(n_trials, n_voxels)`, int16, `n_trials = 3 × 9,000–10,000 ≈ 27,000–30,000` [DG §1 D1, DG §3 `D-NSD`]. **Volume space, not fsaverage** — fsaverage is used only for visualization [DG §1 D1]. |
| **Outputs** | Per subject: `betas_z_avg_subjN.npy`, `(n_images_3x, n_voxels)` float32 (or float16 to fit storage), with `n_images_3x ∈ {≈10,000 (×4 subj), 6,234 (×2), 5,445 (×2)}` [DG §1 D5]. Plus `D-515`: the id list of the **515** shared images seen 3× by **all 8** subjects (of the 1,000 shared images; 3 subjects did not complete all trials) [PS §5 D1a; DG §1 D4]. |
| **Dependencies** | M0 (NSD access, MI-05). External: NSD data-use agreement. |
| **Implementation order** | **2** |
| **Success criteria** | Unit tests, per [DG §10 step 1]: (i) after z-scoring, per-voxel per-session mean ≈ 0 and s.d. ≈ 1; (ii) post-averaging image counts land exactly on {≈10,000, 6,234, 5,445}; (iii) `|D-515| == 515`; (iv) leakage check: `D-515` ids never appear in any training index produced downstream. |
| **Transform order (must be exact)** | 1. restrict to stimuli seen 3× by that participant; 2. z-score across single trials **within each scanning session**; 3. average the 3 repetitions [DG §2.2 `N-BETAZ`]. |
| **Known ambiguities** | **MI-06** — the z-scoring axis is stated "for each participant", not literally "per voxel"; default per-voxel-per-session [DG §9.2]. **MI-01 (HIGH)** — the paper states this normalization only in the RSA paragraph; whether the encoding/decoding models use the *same* betas is never restated. Default: share the node [DG §9.2]. |

---

### M2 — Stimulus and annotation ingest (`D-COCO-CAP`, `D-COCO-CAT`, `D-STREAMS`, `D-GCC`)

| Field | Content |
|---|---|
| **Purpose** | Assemble every non-brain input artifact and the NSD↔COCO id mapping. |
| **Inputs** | MS COCO: 5 human captions per image and instance category labels for the 73,000 NSD images [PS §5 D2; DG §1 D6, D7]. NSD `nsd_stim_info` for the NSD↔COCO mapping [DG §1 D6]. NSD 'streams' ROI masks (EVC, Ventral, Lateral, Parietal; maps additionally show Non-visual, Midventral, Midlateral, Midparietal) [DG §1 D9]. Google Conceptual Captions, **3.1 million** captions, text only [DG §1 D8]. Allen et al. 2021 face/place/body ROIs and Pennock et al. 2021 food ROIs — **overlay only, not an analysis input** [DG §1 D10, D11]. |
| **Outputs** | `captions_73k.json` (73,000 × 5 strings); `categories_73k.npy` (73,000 × K binary-ready labels); `streams_roi_subjN.nii.gz` masks; `gcc_captions.txt` (3.1e6 strings); `nsd_to_coco.csv`. |
| **Dependencies** | M0. External: COCO download; GCC terms; Pennock ROI masks are **not distributed with NSD** and must be obtained from ref. 55's supplement [DG §1 D11]. |
| **Implementation order** | **3** |
| **Success criteria** | (i) Every one of the 73,000 NSD images maps to exactly 5 COCO captions; (ii) GCC line count = 3.1e6 ± the vendor's stated count; (iii) 'streams' masks cover all four ROI labels for all 8 subjects. |
| **Known ambiguities** | **MI-07** — K, the number of COCO categories in the multi-hot vector, is never stated; default 80 thing-categories [DG §9.2]. COCO split/year (`train2017`/`val2017`) is **not stated** [DG §1 D6]. Exact NSD filenames (`nsd_stimuli.hdf5`, `streams.nii.gz`) are **not named in the paper** [DG §1 D3, D9]. |

---

### M3 — Text encoder core and caption embeddings (`N-MPNET`, `N-EMB-CAPTION`)

| Field | Content |
|---|---|
| **Purpose** | The single text encoder that serves *every* text-side use in the paper: NSD caption embeddings, category-word strings, noun/verb strings, single words, scrambled sentences, the 15 hand-written contrast sentences, the 3.1M-caption dictionary, and the RCNN training targets [DG §2.1 merge M1]. |
| **Inputs** | `captions_73k.json` from M2; `sentence-transformers/all-mpnet-base-v2` checkpoint [DG §1 D12]. |
| **Outputs** | `mpnet_caption_emb_73k.npy` — `(73000, 768)` float32, **224 MB** [DG §8]. Canonical construction: encode each of an image's 5 captions → `(5,768)` → **mean across the 5** [DG §2.2 `N-EMB-CAPTION`]. Plus a content-addressed embedding cache keyed by `sha1(string)` so all downstream variants dedupe against it [DG §2.1 M1]. |
| **Dependencies** | M2. External: HuggingFace checkpoint. |
| **Implementation order** | **4** |
| **Success criteria** | (i) shape `(73000, 768)`; (ii) re-encoding a fixed caption is bit-identical across runs; (iii) qualitative reproduction of the Supp. Fig. 2 t-SNE structure (objects present / actions / scene type separate) [DG §10 step 2, §6 E10a]. |
| **Known ambiguities** | **MI-11 (HIGH)** — MPNet pooling, `max_seq_len`, and **L2-normalization** of the output are never stated. Normalization changes both the ridge X-scaling *and* the cosine-distance loss geometry of the RCNN. Default: SBERT `encode()` defaults (mean pooling, `normalize_embeddings=False`); **test both** [DG §2.2 `N-MPNET`, §9.2]. **MI-30** — no checkpoint revision hash; pin the HF revision explicitly [DG §9.2]. **H7 (hidden assumption)** — `all-mpnet-base-v2` was itself fine-tuned partly on COCO, so the target representation is not independent of the stimulus corpus [DG §9.3]. |
| **Compute** | ~10 min on 1 GPU at batch 256 [EST, DG §8]. |

---

### M4 — Text-variant representations (`N-EMB-CATWORDS`, `-NOUNS`, `-VERBS`, `-SINGLEWORD`, `-SCRAMBLE`, `-ADJADVPREP`, `-OTHERLLM`, `-SENTENCES15`)

| Field | Content |
|---|---|
| **Purpose** | Build every *text-input* ablation. These are the entire experimental content of Fig. 3 and of Supp. Figs. 9–11: the encoder and the brain side are held fixed and only the string changes, so any difference is attributable to linguistic content [DG §6 E5 "design invariant"]. |
| **Inputs** | M3's encoder + cache; COCO captions and category labels from M2; NLTK POS tagger [DG §1 D15]. |
| **Outputs** | Eight `(73000, 768)` arrays plus one `(15, 768)`: `N-EMB-CATWORDS` (category words **concatenated into one string**, one forward pass); `N-EMB-NOUNS` / `N-EMB-VERBS` (NLTK POS filter, then **also concatenated** — this is [DERIVED], not directly stated, per [DG §2.2 table note]); `N-EMB-SINGLEWORD` (each word of each of the 5 captions fed **separately**, then averaged); `N-EMB-SCRAMBLE`; `N-EMB-ADJADVPREP`; `N-EMB-OTHERLLM` (n_llm × 73000 × d); `N-EMB-SENTENCES15` (the 15 verbatim contrast sentences, listed in full in [PS §6 P6] and [DG §6 E3]). |
| **Dependencies** | M2, M3. |
| **Implementation order** | **5** |
| **Success criteria** | (i) shapes as above; (ii) **`N-EMB-SCRAMBLE` vs `N-EMB-CAPTION`: mean Pearson r = 0.91, s.d. = 0.03 across participants** — a hard, published number and the single best cheap check of the whole text pipeline [PS §8 C11; DG §6 E10c]; (iii) the 15 sentences match the verbatim strings character-for-character. |
| **Known ambiguities** | **MI-09** — NLTK version / POS tagset not stated; default Penn Treebank, nouns = `NN*`, verbs = `VB*` [DG §9.2]. **MI-27** — the identities of the "several other LLMs from the Sentence-Transformers leaderboard" are **never named**, so claim C12 is unverifiable as stated [DG §9.2; PS §12 N5]. Whether nouns/verbs are concatenated or averaged is forced by inference, not stated [DG §2.2]. |

---

### M5 — Non-LLM semantic controls (`N-EMB-WORDVEC`, `N-MULTIHOT`)

| Field | Content |
|---|---|
| **Purpose** | The control models of Fig. 3a/3c: fastText, GloVe, and the multi-hot category vector. |
| **Inputs** | fastText vectors [DG §1 D13], GloVe vectors [DG §1 D14], COCO category labels and captions from M2. |
| **Outputs** | `N-EMB-WORDVEC` × 4 arrays: {fastText, GloVe} × {category labels (Fig. 3a branch), all words of all 5 captions (Fig. 3c branch)}, each `(73000, d)`, combined by **averaging** the word vectors ("word embeddings can be combined additively … so we average the embeddings across words") [DG §2.2 `N-EMB-WORDVEC`]. `N-MULTIHOT`: `(73000, K)` binary [DG §2.2 `N-MULTIHOT`]. Plus a **published OOV decision log** (see below). |
| **Dependencies** | M2. |
| **Implementation order** | **6** |
| **Success criteria** | (i) shapes correct; (ii) images with **no category words** receive the embedding of the word **'something'** (the paper's stated neutral fallback) [DG §2.2]; (iii) the OOV log is complete and machine-readable. |
| **Known ambiguities** | **MI-12** — the paper's OOV handling was **manual and unlogged**: *"we either corrected the misspelling, found a similar word in the fasttext corpus, or removed them"*, with no published list. **Fig. 3a/3c control bars are therefore not exactly reproducible, by construction** [DG §9.2; PS §12 N10]. Mitigation: log every OOV decision in the reimplementation and publish the list the original lacks. **MI-08** — fastText/GloVe dimensionality, corpus, and vector file are not stated; defaults `crawl-300d-2M` and `glove.840B.300d` [DG §9.2]. |

---

### M6 — Brain pattern extraction: searchlight + ROI (`N-SEARCHLIGHT`, ROI masking)

| Field | Content |
|---|---|
| **Purpose** | Turn the normalized beta volume into the two families of activity patterns the RSA consumes. |
| **Inputs** | M1's `betas_z_avg_subjN.npy`; NSD brain masks; `D-STREAMS` ROI masks from M2. |
| **Outputs** | `sphere_index.npz` — for each voxel v, the ids of the voxels in a sphere of **radius 6 voxels** centred at v (= 10.8 mm at 1.8 mm iso; ≤ ~905 voxels [DERIVED, DG §2.2 `N-RDM`]), keeping only spheres with **>50 % of voxels inside the brain**, with out-of-brain voxels **excluded from the sphere** [DG §2.2 `N-RDM` / `N-SEARCHLIGHT`]. Plus ROI pattern indices for {EVC, Ventral, Lateral, Parietal} and the streams-wide voxel mask used by the decoder. Storage ≈ 650 MB [EST, DG §8]. |
| **Dependencies** | M1, M2. |
| **Implementation order** | **7** |
| **Success criteria** | Unit test [DG §10 step 3]: max sphere size ≤ 905; the >50 %-in-brain rule rejects the expected edge voxels; every retained sphere has ≥ 1 voxel. |
| **Known ambiguities** | **MI-13 (MED–HIGH)** — the **ROI** case has *no* specified voxel selection or reliability threshold, unlike the fully specified searchlight case. Default: all voxels in the ROI mask, no thresholding. This changes **all of Fig. 3 and Fig. 4e** [DG §9.2; PS §12 N7]. |

---

### M7 — RDM builder and the 100-image RSA sampler (`N-RDM`, `N-RSASAMPLE`)

| Field | Content |
|---|---|
| **Purpose** | The load-bearing procedural core of the paper: a *single* shared sampler serves Figs. 1b, 3, 4c, 4d, 4e and every supplementary searchlight [DG §2.1 merge M6]. |
| **Inputs** | Brain patterns from M6; any model feature matrix from M3/M4/M5/M15/M17. |
| **Outputs** | Per (subject, model): one scalar r per ROI, and one **searchlight correlation volume**. Procedure, verbatim from [DG §2.2 `N-RSASAMPLE`]: repeatedly sample **100 images without replacement** from the participant's 3×-seen pool; build a 100×100 brain RDM and a 100×100 model RDM **on the same 100 images**; Pearson-correlate the two upper-triangular vectors (**length exactly 4,950**); repeat until the pool is exhausted (**100 / 62 / 54 splits** for the ~10,000 / 6,234 / 5,445-image subjects); **average the resulting correlation volumes**. |
| **Dependencies** | M6 (brain patterns), and at least one model-feature module (M3 minimum). |
| **Implementation order** | **8** |
| **Success criteria** | (i) upper-triangle length is exactly **4,950** [DG §2.2]; (ii) split counts are exactly 100/62/54 by subject, matching the floor of `n_images_3x / 100` [DERIVED check in DG §1 D5]; (iii) **the same 100-image splits are reused across all models for a given subject** (see MI-14 below) — assert this in a test, because every model-vs-model t-test depends on it; (iv) brain RDMs are computed **once** per (subject, split, sphere) and cached for reuse across all models — [DG §8] calls this "the single biggest engineering win". |
| **Known ambiguities** | **MI-14 (HIGHEST-risk silent gap)** — the paper never says whether the splits are **held fixed across models (paired)** or **redrawn per model (unpaired)**. Every model-vs-model t-test in Figs. 3 and 4 depends on it. Default: **fix the splits per subject across all models**; this is the only way the "significance of the difference between model correlations" test at N=8 is well-posed [DG §9.2]. **MI-03** — the sampler's RNG seed is unstated; fix `seed=0` and report sensitivity over ≥3 seeds [DG §9.2]. **MI-02** — the **model-side** distance metric is never stated (only the brain's Pearson correlation distance is); default: same metric [DG §2.1 merge M5]. **H1** — the 100-image-split RSA is *not* an unbiased estimator of the full-RDM RSA, and the paper gives no bias analysis [DG §9.3]. **H2** — subjects with 54 vs 100 splits contribute estimates of unequal noise into an *unweighted* N=8 t-test [DG §9.3]. |
| **Compute** | The dominant cost of the entire paper. [EST, DG §8]: ≈ 6.5×10¹⁴ FLOP for one model × 8 subjects full-brain; several GPU-hours per model per full-brain searchlight; × ~14 competitor models × (layers × timesteps × 10 seeds) for the supplementary sweeps → **plausibly thousands of GPU-hours**. |

---

### M8 — Noise ceilings and ceiling correction (`N-NC-RSA`, `N-NCCORR`)

| Field | Content |
|---|---|
| **Purpose** | The RSA noise ceiling and the "noise-ceiling-corrected" / "noise-normalized" rescaling used in Fig. 3 and Fig. 4e — **and only there** [DG §7]. |
| **Inputs** | M1's betas restricted to `D-515`; M7's uncorrected model correlations. |
| **Outputs** | Per subject per ROI: a ceiling scalar = **the correlation between this participant's RDM and the mean RDM across the other 7 participants**, all RDMs computed on the **shared 515 images** ("pitting the model against the average of seven human participants"); then corrected correlations, which are averaged across participants [DG §2.2 `N-NC-RSA`]. |
| **Dependencies** | M1, M6, M7. |
| **Implementation order** | **9** |
| **Success criteria** | (i) Ceiling computed on 515×515 RDMs, LOO over 7 others, for all 8 subjects. (ii) **Corrected Fig. 3 bars land in the published range and Fig. 4e points land in the published ranges: Ventral y ≈ 0.15–0.45, Lateral ≈ 0.1–0.5, Parietal ≈ 0.1–0.4** [PS §2 Fig. 4e row; DG §6 E9]. (iii) The correction is applied to Figs. 3 and 4e **only** — Figs. 1b, 1c, 4c, 4d are explicitly **not** ceiling-corrected, and the two scales must never be cross-compared [DG §7 "consistency trap"; PS §7]. |
| **Known ambiguities** | **MI-15 (BLOCKING for exact values)** — the correction **arithmetic is never written**. Divide? subtract? divide by sqrt? Default `r_corrected = r_model / r_ceiling`. It rescales every bar in Fig. 3 and every point in Fig. 4e and **can flip marginal significance** [DG §9.1]. Mitigation prescribed by [DG §10 step 7]: run all candidate formulas and report which reproduces the published bar heights. **MI-16 / H3** — the model correlation is computed on the participant's ~10,000 images (100×100 sub-RDMs) but the ceiling on the 515 shared images (515×515 RDM). Different image sets, different RDM sizes; **the paper does not reconcile this**. Implement as literally written and report the mismatch [DG §2.2 `N-NC-RSA`, §9.3]. |
| **Do NOT merge** | `N-NC-RSA` and `N-NC-CAPTION` (M11) are different ceilings with different units, data and purpose. "A single `noise_ceiling()` function here is a bug" [DG §2.1 merge M8]. |

---

### M9 — Group statistics and surface visualization (`N-STATS`, `N-SURF`)

| Field | Content |
|---|---|
| **Purpose** | The terminal statistical node for every analysis **except Fig. 2a** [DG §2.1 merge M7]. |
| **Inputs** | Per-subject scalars (ROI) or maps (searchlight) from M7/M8/M10/M15/M17. |
| **Outputs** | Thresholded group maps and P-matrices. Test: **two-tailed t-test across the 8 participants**; multiple comparisons: **Benjamini–Hochberg FDR at P = 0.05** [DG §2.2 `N-STATS`]. **Two null hypotheses**, both required: (a) model correlation vs 0 (single-model maps); (b) significance of the **difference** between two model correlations (model comparisons) [DG §2.2]. Visualization: average maps → thresholded → projected into FreeSurfer **`fsaverage`** → flatmap [DG §2.2 `N-SURF`]. |
| **Dependencies** | M7 (and M8 where ceiling-corrected). |
| **Implementation order** | **10** |
| **Success criteria** | (i) Reproduces the sign and topography of the published thresholded maps at the stated colorbar limits: Fig. 1b **−0.25…0.25**; Fig. 1c **−0.73…0.73**; Fig. 4c **−0.06…0.06**; Fig. 4d **−0.07…0.07** [PS §2]. (ii) The **Fig. 2a path bypasses this node entirely** — a test must assert that no FDR is applied there [DG §2.1 merge M7]. |
| **Known ambiguities** | **MI-21** — raw r vs Fisher-z-transformed r in the t-test is not stated; default raw r [DG §9.2]. **MI-22** — the FDR *family* (within one map? across maps?) is not stated; default within a single map over all valid spheres [DG §9.2]. |

---

### M10 — Fractional ridge core, encoding model, inter-participant agreement (`N-FRACRIDGE`, `N-ENC`, `N-IPA`)

| Field | Content |
|---|---|
| **Purpose** | The second (and last) mapping family. **One** ridge core serves both encoding and decoding; they differ *only* in which matrix is X and which is y, and in the voxel mask. "Do not write two fitters." [DG §2.1 merge M2] |
| **Inputs** | X = `N-EMB-CAPTION` `(n, 768)` from M3; y = `N-BETAZ` **whole brain** `(n, p_voxels)` from M1 ("we apply this analysis to the full brain") [DG §6 E2]. Test split = the **515 shared images**, held out [DG §1 D4]. |
| **Outputs** | ĥ of shape `(p_voxels × 768)` per subject (≈550 MB/subject at p≈180k [EST, DG §8]); per-voxel test-set Pearson r between predicted and true betas; `N-IPA` = the mean Pearson r between each participant's voxel activities and the average of the other seven, on the test images (the Fig. 1d y-axis) [DG §3 `N-IPA`]. |
| **Ridge hyperparameters (all STATED)** | **20 regularization fractions, 0.05 → 1.00 in steps of 0.05**; **5-fold cross-validation**; best fraction selected per target, then refit [DG §2.2 `N-FRACRIDGE`]. Library: `fracridge` (ref. 54) [DG §1 D16]. |
| **Dependencies** | M1, M3. External: `fracridge`. |
| **Implementation order** | **11** |
| **Success criteria** | (i) **Encoding r approaches inter-participant agreement in all ROIs** — the paper's own stated sanity check [PS §8 C2; DG §10 step 5]. (ii) Fig. 1d scatter: one dot per voxel, axes 0–0.8, ROI-group colouring reproduces the published ordering (Non-visual < EVC < … < Ventral/Lateral/Parietal) [PS §2 Fig. 1d]. (iii) Leakage test: none of the 515 test images appears in any CV fold. |
| **Known ambiguities** | **MI-17 / X3 (BLOCKING for exact values)** — the Methods say the fraction "that best predicted **each embedding feature**" was selected, but in the *encoding* direction the targets are **voxels**, not embedding features; the identical sentence appears in the decoding section where it *is* coherent. Almost certainly a copy-paste. **Default: one fraction per voxel.** This changes the regularization of every voxel and therefore Figs. 1c, 1d and 2a [DG §2.2 `N-FRACRIDGE`; PS §11 X3]. **MI-18** — CV fold construction (random? contiguous? seeded?) unstated. **MI-19/20** — X/y standardization and intercept handling unstated [DG §9.2]. **MI-36** — the number of voxels *p* is never given, so a reimplementer cannot sanity-check the design size [DG §9.2]. |

---

### M11 — Decoding model, caption retrieval, caption noise ceiling (`N-DEC`, `N-EMB-GCC`, `N-RETRIEVE`, `N-NC-CAPTION`)

| Field | Content |
|---|---|
| **Purpose** | Brain → LLM embedding → caption. The **inverse** of M10 through the *same* ridge core [DG §6 E4 "merge note"]. |
| **Inputs** | X = `N-BETAZ` restricted to **all voxels inside the 'streams' visual ROIs** (not whole-brain — this is the one structural difference from M10) [DG §6 E4]; y = `N-EMB-CAPTION` `(n, 768)`. Dictionary: **3.1M Google Conceptual Captions** → MPNet → `(3.1e6, 768)`, 9.5 GB fp32 / 4.8 GB fp16, ~1.5–3 GPU-hours to build [EST, DG §8]. |
| **Outputs** | 8 × 515 predicted embeddings; per-image Pearson r (predicted vs target 768-d embedding) → per-participant **KDE** over the 515 test images (Fig. 2b bottom-left); 8 × 515 reconstructed captions via **argmax Pearson correlation against every dictionary entry**; the caption noise-ceiling line. |
| **Caption noise ceiling (`N-NC-CAPTION`)** | For each of the 5 human captions, correlate its LLM embedding with the **averaged embedding of the other 4**; average the 5 correlations [DG §2.2 `N-NC-CAPTION`]. **This is a different ceiling from M8 — do not merge** [DG §2.1 merge M8]. |
| **Dependencies** | M1, M2, M3, M6 (streams mask), M10 (the shared ridge core). |
| **Implementation order** | **12** |
| **Success criteria** | (i) KDE mass falls in **0.3–0.7** (the Fig. 2b axis) [DG §10 step 6]. (ii) Decoded-caption **ranks** are reproducible in distribution; the paper prints four worked examples at ranks **0/515, 102/515, 255/515, 459/515** for participants 1–4 [PS §2 Fig. 2b; DG §6 E4]. (iii) Qualitative control reproduced: the decoded caption is **not** simply the nearest *training* caption — the nearest training caption must be printed alongside and shown to differ [DG §6 E4(c)]. |
| **Known ambiguities** | In this direction the "best fraction **per embedding feature**" rule *is* coherent (the 768 features are the targets), so MI-17 does **not** bite here [DG §2.2 `N-FRACRIDGE`]. Retrieval is a single matmul after row z-scoring: 515 × 3.1e6 × 768 ≈ 1.2×10¹² FLOP → seconds on GPU [EST, DG §8]. |

---

### M12 — Functional-contrast prediction from novel sentences (`N-CONTRAST`)

| Field | Content |
|---|---|
| **Purpose** | Fig. 2a: feed hand-written sentences through the frozen encoding model and contrast the predicted brain maps. |
| **Inputs** | Frozen ĥ from M10; `N-EMB-SENTENCES15` from M4 — the **15 verbatim sentences** (5 People, 5 Places, 5 Food), reproduced in full in [PS §6 P6] and [DG §6 E3]. |
| **Outputs** | Two group flatmaps: **People − Places** (colorbar −1.52…1.52) and **People − Food** (colorbar −0.94…0.94), in units of "difference in predicted voxel activity" [PS §2 Fig. 2a; DG §6 E3]. Procedure: predict per sentence, **average the 5 predictions within a category**, then contrast. |
| **Dependencies** | M4, M10. |
| **Implementation order** | **13** |
| **Success criteria** | Qualitative overlap of the People map with **FFA1/2, OFA, EBA, pSTS**; the Places map with **PPA, OPA**; the Food map with Pennock et al. 2021 food-selective areas [PS §8 C4; DG §6 E3]. There is no published effect size to hit — the claim is topographic. |
| **Known ambiguities / stated weakness** | **No FDR correction** — the *only* analysis in the paper without multiple-comparison control, stated explicitly: *"unlike all other maps in this Article, there is no correction for false discovery rate"* [DG §6 E3]. And: *"We did not have a precise method for selecting these sentences"* [DG §6 E3]. **H4** — Fig. 2a inherits every assumption of the encoding model and adds two of its own [DG §9.3]. This makes C4 the weakest claim in the paper on evidentiary grounds [PS §13 R8]. It is nonetheless reproducible **as run**, because the sentences are printed verbatim. |

---

### M13 — ANN training-set construction (`N-COCOMINUSNSD`)

| Field | Content |
|---|---|
| **Purpose** | Build the COCO∖NSD image set on which every network in the paper is trained, and guarantee zero NSD leakage. |
| **Inputs** | COCO train+val images; the 73,000 NSD image ids from M2. |
| **Outputs** | **48,236 train / 2,051 validation** images; the **73,000 NSD images become the test set** ("the networks did not see any of the NSD images during training, nor in validation") [DG §1 D17, §6 E6]. Preprocessing: for rectangular images take the **largest square crop** (as was done for the NSD stimuli), then resize to **128 × 128 px** [DG §6 E6]. |
| **Dependencies** | M2. |
| **Implementation order** | **14** |
| **Success criteria** | (i) **Leakage test (hard gate): `set(train) ∩ set(NSD_73k) == ∅` and `set(val) ∩ set(NSD_73k) == ∅`.** (ii) Train/val counts within ±5 of 48,236 / 2,051. |
| **Known ambiguities** | **MI-10 / X2** — the paper says **48,236** train images in one sentence and **48,238** two paragraphs later (both p. 1230); Fig. 4a and the Discussion round to "48,000" [DG §1 D17, §9.4]. Immaterial numerically, but it signals the COCO∖NSD filter is not exactly specified. **Reproduce the filter from the repo; do not hardcode either number** [DG §9.2 MI-10]. **MI-24** — no data augmentation or pixel-normalization statistics are stated; the largest-square-crop + 128×128 resize is the *only* stated preprocessing [DG §9.2]. |

---

### M14 — RCNN / ResNet architecture and shared training loop (`N-RCNN`, `N-RCNN-CAT`, `N-RESNET50-*`, `N-RCNN-ECOSET`, `N-TRAINLOOP`, `N-COSDIST`)

| Field | Content |
|---|---|
| **Purpose** | Train the paper's own models. One training loop serves all arms; only two things vary across them: the **readout activation** (linear vs sigmoid) and the **batch size** (96 RCNN / 512 ResNet) [DG §2.1 merge M9]. |
| **Inputs** | M13's images; targets = `N-EMB-CAPTION` (LLM arm) or `N-MULTIHOT` (category arm). |
| **Architecture** | RCNN = **vNet** (ten-layer conv net with a foveal receptive-field-size gradient, ref. 63) made recurrent with **lateral and top-down** connections following ref. 115 (BLT) [DG §6 E6]. **Pre-readout representation = 512 features** [DG §6 E6; PS §11 X6]. LLM arm: fully connected **768-d linear** readout — *no* softmax or sigmoid, because MPNet embeddings can be negative. Category arm: identical architecture, **sigmoid** readout, multi-hot target [DG §6 E6]. |
| **Loss** | **Cosine distance for both arms** [DG §2.1 merge M10]. The category arm minimizes cosine distance **on a sigmoid output — not BCE**. "This is unusual; implement as written." [DG §2.1 M10] |
| **Training hyperparameters (all STATED, p.1230)** | Adam; **lr = 5×10⁻²**; **Adam ε = 1×10⁻¹**; **200 epochs**; **10-epoch linear warm-up** then **cosine decay**; **batch 96** (RCNN) / **512** (ResNet); input 128×128 [DG §2.2 `N-TRAINLOOP`]. |
| **Instances** | 10 RCNN seeds per arm (LLM, category); **1** ResNet50 each (LLM, category), non-pretrained; **1** RCNN-on-ecoset with **channels doubled** [DG §6 E6]. |
| **Outputs** | 10 × LLM-RCNN, 10 × category-RCNN, 1 × LLM-ResNet50, 1 × category-ResNet50, 1 × ecoset-RCNN checkpoints. |
| **Dependencies** | **M0 (hard gate)**, M3, M5, M13. External: ecoset. |
| **Implementation order** | **15** |
| **Success criteria** | (i) The two arms are trained on **the exact same images, architecture, dimensionality and random seeds**, differing **only** in objective (and the forced readout-activation difference) — this controlled-comparison guarantee is the paper's own [DG §6 E6]. Assert it in a test. (ii) Validation cosine similarity beats the `N-PROBE-FLOOR` baselines of M16. (iii) Pre-readout width is **512**, not 768 (see X6 below). |
| **BLOCKING gaps** | **MI-32** — the **number of recurrent timesteps** is never stated anywhere in Methods, yet "timestep" is load-bearing in Figs. 4c/4d and Supp. Figs. 14/16. **The network cannot be instantiated.** [DG §9.1] Note [DG §9.1] flags that ref. 115 uses 8 timesteps for BLT-style nets **but explicitly warns this is not stated by *this* paper and must not be cited as such.** **MI-33** — vNet per-layer channel counts / RF-size gradient are not given. **The network cannot be instantiated.** [DG §9.1] Both must come from M0. |
| **Other ambiguities** | **MI-31 (HIGH)** — **Adam ε = 1×10⁻¹ is 10⁷× the default (1e-8)** and effectively damps Adam's adaptive scaling toward SGD-with-momentum at lr 0.05. It is *stated*, not a typo we may silently correct. **Anyone who "fixes" it to 1e-8 trains a materially different optimizer and will not reproduce Figs. 4b–e** [DG §9.2 MI-31; PS §4.2 note]. **MI-23** weight decay, **MI-24** augmentation, **MI-25** Adam β1/β2 — all unstated [DG §9.2]. **H5** — the category control's cosine-loss-on-sigmoid may handicap it; the paper never tests BCE, which weakens C16/C17 [DG §9.3]. |
| **Compute** | [EST, DG §8]: 20 RCNN runs (10 seeds × 2 objectives) at ~0.5–2 GPU-days each → **10–40 GPU-days**, and **the unknown timestep count T multiplies this linearly**. + 2 ResNet50 runs (~0.5 GPU-day each) + 1 ecoset RCNN (**~1 GPU-week**). |

---

### M15 — Activation extraction for our models (`N-ACT`, ours)

| Field | Content |
|---|---|
| **Purpose** | Extract the representations that re-enter the RSA family (Figs. 4c/4d) and feed the readout probe (Fig. 4b). |
| **Inputs** | M14's checkpoints; the 73,000 NSD images. |
| **Outputs** | Per (model, layer, timestep, seed): `(73000, 512)` float32 = **150 MB** each [EST, DG §8]. For Figs. 4c/4d: **last layer, last timestep**. |
| **Dependencies** | M13, M14. |
| **Implementation order** | **16** |
| **Success criteria** | (i) **The extracted tensor is the 512-d pre-readout layer, NOT the 768-d readout.** This is X6: the paper never puts "pre-readout" and "512" in one sentence, and "a reimplementer who taps the 768-d readout will not reproduce Figs. 4c–e" [DG §2.2 `N-ACT`; PS §11 X6]. Assert `shape[1] == 512` in a test. (ii) **Seed handling: compute individual RDMs per seed, then average the CORRELATIONS with brain data across seeds — not the RDMs** [DG §2.2 `N-ACT`, verbatim]. Assert this ordering in a test; averaging RDMs first is a different (and wrong) estimator. |
| **Known ambiguities** | Full RCNN sweeps (10 layers × T timesteps × 10 seeds × 2 objectives) do not fit in cache; stream per (layer, timestep) [DG §8]. |

---

### M16 — Frozen-readout probe and its noise floor (`N-PROBE`, `N-PROBE-FLOOR`) → Fig. 4b

| Field | Content |
|---|---|
| **Purpose** | Test the asymmetric-subsumption claim: can category labels be read out of an LLM-trained net, and vice versa? |
| **Inputs** | M15's last-layer/last-timestep activations over the **entire NSD dataset** — collecting activations this way is *equivalent to freezing the weights* but avoids recomputing activations each epoch [DG §6 E7]. Split: **first 71,000 NSD images = train, last 2,000 = test** [DG §6 E7]. |
| **Outputs** | Four bars: decode category {from LLM-trained, from category-trained}; decode LLM {from LLM-trained, from category-trained}. Metric = **test cosine similarity**, averaged over the 10 seeds; error bars = **s.d. across the 10 instances** [PS §2 Fig. 4b]. Two baseline lines from `N-PROBE-FLOOR`: the mean LLM embedding (resp. mean multi-hot) across the **48,238** training images, and its mean cosine distance to the **2,051** validation targets [DG §6 E7]. |
| **Training** | Reuses `N-TRAINLOOP` verbatim: "the readout activation, optimizer and training hyperparameters were also the same as for training the full RCNN" [DG §2.1 merge M9]. |
| **Dependencies** | M14, M15. |
| **Implementation order** | **17** |
| **Success criteria** | (i) The **asymmetry** reproduces: category ← LLM works (matching the category-trained ceiling); LLM ← category does **not** [DG §6 E7]. This *is* the claim (C15). (ii) All four bars land in the published band **y ≈ 0.40–0.70** [PS §2 Fig. 4b]. |
| **Known ambiguities** | **MI-34** — the exact bar values appear **nowhere** in text or caption; they are readable only off the axis. C15 can therefore be reproduced *qualitatively* but **not checked numerically** [DG §9.2]. **MI-29** — "first 71,000 / last 2,000" in *what index order*? Default: NSD's canonical 73k index order [DG §9.2]. Note also the train/test counts here (71,000/2,000) are over **NSD** images while the floor is computed over the **48,238 COCO** training images — two different sets, as written. |

---

### M17 — Competitor model zoo and Fig. 4e benchmark (`N-ACT` for 13 published ANNs)

| Field | Content |
|---|---|
| **Purpose** | The 14-model benchmark that carries the paper's headline data-efficiency claim (C17). |
| **Inputs** | The 13 published checkpoints, per [DG §6 E9] / [PS §4.3]: **Supervised (object)** `Cornet-s` (thingsvision), `Alexnet` (brainscore), `Alexnet-gn-sv`, `Resnet50` (brainscore), `nf_resnet50` (timm), `rcnn_ecoset` (ours, doubled channels); **Supervised (scene)** `Resnet50_Places365` (trained by the authors), `Taskonomy_scene_cat`; **Semi-supervised** `Resnext101_32x8d` (WSL, **914 M images**), `CLIP_RN50_imgs`, `CLIP_ViT_imgs`; **Unsupervised** `Alexnetgn_ipcl`, `Google_simclr_rn50`; plus **`LLM trained (ours)`**. |
| **Outputs** | Three scatter plots (Ventral / Lateral / Parietal): x = **training-dataset size in images, log scale, 10⁴…10⁹**; y = **noise-ceiling-normalized Pearson r**. Layer used: the **pre-readout layer** for every model, **except CLIP, where the final embedding is used** [DG §6 E9]. Pairwise corrected P values → Supp. Fig. 20. |
| **Dependencies** | M6, M7, M8, M9, M14 (ours + ecoset), M15. External: thingsvision, brainscore, timm, torch.hub WSL, openai/CLIP, google-research/simclr, StanfordVL/taskonomy [DG §1 D19]. |
| **Implementation order** | **18** |
| **Success criteria** | (i) `LLM trained (ours)` significantly outperforms the other models in **ventral and parietal**, with **one** non-significant exception — but **where that exception lives is a live paper-internal contradiction (X1)**: the body text says lateral, the Fig. 4 caption says parietal/CORnet-S. **These cannot both hold. Resolve against Supp. Fig. 20 before scoring this criterion** [PS §11 X1; DG §9.4]. (ii) y-values land in the published ranges (Ventral 0.15–0.45, Lateral 0.1–0.5, Parietal 0.1–0.4). (iii) The **COCO-subset control** reproduces: RCNNs trained on ecoset categories are *not* outperformed by RCNNs trained on COCO categories, and **both** are outperformed by the LLM-trained RCNN [DG §6 E9]. |
| **Known ambiguities** | **MI-04 (HIGH)** — the per-model image preprocessing table is **not given**; the Methods say only *"we preprocess stimuli to match the input range expected by each model."* **Fig. 4e's entire ranking — i.e. claim C17 — hangs on this one clause** [DG §9.2; PS §12 N8]. **MI-26 (HIGH)** — the per-model "pre-readout layer" identity is not given per model [DG §9.2]. **MI-35** — the per-model training-set sizes plotted on the x-axis are given numerically for only three models (COCO 48k, ImageNet/ecoset ">1M", WSL 914M), so **the x-axis cannot be reconstructed exactly** [DG §9.2]. **H6** — the x-axis counts **images only**; the LLM's own text-corpus training (which produced the *targets*) is excluded from the "training set size" of the LLM-trained RCNN. The authors flag this as out of scope — but it is precisely the axis on which the headline data-efficiency claim is made [DG §9.3]. |

---

### M18 — Control and auxiliary analyses (`E10a`–`E10h`)

| Field | Content |
|---|---|
| **Purpose** | The robustness checks on which six of the paper's claims rest **entirely** [PS §13 R1]. |
| **Inputs** | Outputs of M3–M11 re-run with variant inputs. |
| **Sub-analyses (all from [DG §6 E10])** | **E10a** t-SNE of the caption embedding space → Supp. Fig. 2 (qualitative, no statistic reported). **E10b** caption descriptive stats → Supp. Fig. 1. **E10c** scrambled-sentence control → **mean r = 0.91, s.d. 0.03**. **E10d** other-LLM control → "none of the statistical comparisons among LLM models was significant". **E10e** adjective/adverb/preposition control → "very low alignment". **E10f** cross-participant encoding (train on one subject, test on the other seven). **E10g** category-word encoding+decoding rerun → both get worse. **E10h** RCNN layer × timestep searchlight sweep → early layers ↔ lower visual areas, higher layers ↔ higher visual areas. |
| **Outputs** | Supp. Figs. 1, 2, 5, 8, 9, 10, 11, 13, 14, 16 equivalents. |
| **Dependencies** | M3, M4, M7, M9, M10, M11, M15. |
| **Implementation order** | **19** |
| **Success criteria** | (i) **E10c hits 0.91 ± 0.03** — the one exact published number available for the text pipeline. (ii) E10h reproduces the layer→hierarchy ordering. (iii) E10g shows a *drop* for category-word embeddings in both encoding and decoding. |
| **Known ambiguities** | **MI-27** — the "other LLMs" are never named → **C12 is unverifiable as stated** [DG §9.2]. **MI-28** — t-SNE perplexity/seed/metric unstated [DG §9.2]. **X7** — Supp. Fig. 5 is cross-referenced **four mutually incompatible ways** across pp. 1222/1223/1224/1228 (other-LLM comparison / t-SNE of train-test split / cross-participant encoding); **its actual content is undetermined from the main text** and must be resolved by fetching the SI [DG §9.4]. |

---

### M19 — Figure reproduction and reporting

| Field | Content |
|---|---|
| **Purpose** | Render the four main figures and the supplementary set, and produce the reproduction verdict. |
| **Inputs** | Artifacts from M7–M18; the figure→pipeline reverse index in [DG §7]. |
| **Outputs** | Fig. 1 (a schematic, b/c/d), Fig. 2 (a/b), Fig. 3 (a/b/c × 4 ROI columns), Fig. 4 (a schematic, b/c/d/e); `reports/review.md` with a per-claim verdict. |
| **Dependencies** | all of M7–M18. |
| **Implementation order** | **20** |
| **Success criteria** | Each figure matches its **stated colorbar/axis range** and its **noise-ceiling / FDR status** exactly, per the reverse index [DG §7]: Fig. 1b no ceiling + FDR; Fig. 1c no ceiling + FDR; Fig. 1d no ceiling, no test; **Fig. 2a no ceiling, NO FDR**; Fig. 2b caption ceiling; **Fig. 3 ceiling-corrected + FDR**; Fig. 4b cosine floor; Fig. 4c/4d **no ceiling**, FDR on the difference; **Fig. 4e ceiling-normalized + pairwise FDR**. |
| **Hard rule** | **The consistency trap** [DG §7]: noise-ceiling correction is applied to Figs. 3 and 4e **only**. Numbers from Figs. 1b/1c/4c/4d live on a different scale and **must not be cross-compared.** Any reported comparison across those two families is a bug in the report, not a finding. |

---

## 2. Global implementation order (topological)

| Rank | Module | Blocking dependencies | Est. wall-clock | Est. compute |
|---|---|---|---|---|
| 1 | **M0** Asset acquisition + gap resolution (SI, repo, NSD DUA) | — | days (download-bound; [DG §8] notes NSD download can take days on a slow link) | ~100–150 GB storage [EST] |
| 2 | **M1** NSD ingest + `N-BETAZ` | M0 | 1–2 days | minutes–1 h compute; 86–173 GB storage [EST] |
| 3 | **M2** COCO / GCC / ROI ingest | M0 | 1–2 days | download-bound |
| 4 | **M3** MPNet + caption embeddings | M2 | hours | ~10 GPU-min; 224 MB [EST] |
| 5 | **M4** Text variants | M2, M3 | 1 day | GPU-minutes |
| 6 | **M5** fastText / GloVe / multi-hot | M2 | 1 day (+ manual OOV work) | CPU |
| 7 | **M6** Searchlight + ROI patterns | M1, M2 | 1 day | minutes; ~650 MB [EST] |
| 8 | **M7** RDM + `N-RSASAMPLE` | M6, M3 | 1 week to get right | **dominant cost**: several GPU-h per model per full-brain searchlight; ~6.5×10¹⁴ FLOP/model [EST, DG §8] |
| 9 | **M8** Noise ceilings + correction | M1, M6, M7 | days | cheap |
| 10 | **M9** `N-STATS` + `N-SURF` | M7 (M8 where corrected) | days | seconds–minutes |
| 11 | **M10** fracridge + encoding + IPA | M1, M3 | days | minutes/subject (X is only 768-wide) [EST] |
| 12 | **M11** Decoding + GCC dictionary + retrieval | M1, M2, M3, M6, M10 | days | dictionary build 1.5–3 GPU-h; decode SVD tens of min/subject; retrieval seconds [EST] |
| 13 | **M12** Functional contrasts (Fig. 2a) | M4, M10 | hours ("cheap once step 5 exists", [DG §10]) | negligible |
| 14 | **M13** COCO∖NSD training set | M2 | 1 day | CPU |
| 15 | **M14** RCNN/ResNet training | **M0 (GATE)**, M3, M5, M13 | weeks | **10–40 GPU-days** (RCNNs) + ~1 GPU-day (ResNets) + ~1 GPU-week (ecoset) [EST] |
| 16 | **M15** Activation extraction (ours) | M13, M14 | days | inference-bound, hours |
| 17 | **M16** Frozen-readout probe (Fig. 4b) | M14, M15 | days | modest |
| 18 | **M17** Competitor zoo + Fig. 4e | M6–M9, M14, M15 | weeks | RSA × 14 models — the second-largest cost |
| 19 | **M18** Control analyses | M3, M4, M7, M9, M10, M11, M15 | weeks | RSA sweeps × layers × timesteps |
| 20 | **M19** Figures + verdict | all | 1 week | negligible |

**The order above is exactly the build order recommended in [DG §10]**, expanded to name the gate (M0) and the competitor zoo (M17) as first-class modules.

**Stop-loss checkpoint, explicitly prescribed by [DG §10 step 4]:** after M7, reproduce **Fig. 1b**. *"If Fig. 1b's ventral/lateral/parietal hot-spot pattern does not appear, stop; nothing downstream will work."*

**Hard gate, explicitly prescribed by [DG §10]:** do not begin M14 until M0 has resolved **MI-32, MI-33, MI-15, MI-17** from the release repo. Four of the five blocking gaps are ANN- or RSA-normalization-related and are all resolvable there.

---

## 3. Module → claim map

Claims are cited by their `paper_structure.md` ids (C1–C20) [PS §8]. "Support" = the figure or number the claim rests on. Where a claim rests *entirely* on the Supplementary Information — which is **not in this repository** [PS §13 R1] — it is marked **SI-ONLY**.

| Claim | Statement (short) | Verified by module(s) | Supporting figure / number |
|---|---|---|---|
| **C1** | LLM caption embeddings match brain RDMs across higher visual cortex | M3, M6, M7, M9 | Fig. 1b (colorbar −0.25…0.25); Supp. Figs. 3, 11 |
| **C2** | A linear encoding model from LLM embeddings predicts voxel activity broadly | M10, M9 | Fig. 1c (−0.73…0.73), Fig. 1d; "approaches inter-participant agreement in all ROIs" |
| **C3** | The encoding model generalizes across participants | M18 (E10f), M10 | **SI-ONLY** — Supp. Fig. 5 (whose content is itself contested; see X7) |
| **C4** | The encoding model reproduces category-selective tuning (people/places/food) | M12 | Fig. 2a; Supp. Fig. 6. **Caveat: no FDR; sentences ad-hoc** |
| **C5** | Scene captions are reconstructable from brain activity alone | M11 | Fig. 2b: KDE vs caption ceiling; example ranks 0/102/255/459 of 515 |
| **C6** | LLM category-word embeddings beat multi-hot and word embeddings | M4, M5, M7, M8, M9 | Fig. 3a; pairwise P → Supp. Fig. 12. **Two distinct exceptions:** multi-hot fails in **lateral**; word-embedding fails only for **fastText in EVC** [PS §8 C6] |
| **C7** | Full-caption embeddings beat category-word embeddings, in all ROIs | M4, M7, M8, M9; M18 (E10g) | Fig. 3 reference bar vs 3a; Supp. Fig. 8 |
| **C8** | Full captions beat noun-only and verb-only | M4, M7, M8, M9 | Fig. 3b (exception: nouns in EVC) |
| **C9** | Adjectives/adverbs/prepositions align poorly | M18 (E10e) | **SI-ONLY** — Supp. Fig. 9 |
| **C10** | Whole-caption embeddings beat averaged single-word embeddings | M4, M5, M7, M8, M9 | Fig. 3c |
| **C11** | MPNet is largely insensitive to word order | M4, M18 (E10c) | Supp. Fig. 10; **mean r = 0.91, s.d. 0.03** — the one exact reproducible number for the text pipeline |
| **C12** | Results are not specific to MPNet | M18 (E10d) | **SI-ONLY** — Supp. Fig. 11. **Currently unverifiable: the other LLMs are never named (MI-27)** |
| **C13** | LLM-trained RCNN activations predict brain responses across the visual system | M14, M15, M7, M9 | **SI-ONLY** — Supp. Figs. 13, 14 |
| **C14** | The LLM-trained RCNN aligns *better* with the brain than the LLM embeddings it was trained on | M14, M15, M7, M9 | Fig. 4c (−0.06…0.06); Supp. Fig. 15. Counter-confound: RCNN 512-d < LLM 768-d |
| **C15** | Category labels are decodable from LLM-trained RCNNs, but not vice versa | M16 | Fig. 4b (asymmetry). **Numerically uncheckable — MI-34** |
| **C16** | LLM-trained RCNNs beat category-trained RCNNs (matched arch/data/dims/seeds) | M14, M15, M7, M9 | Fig. 4d (−0.07…0.07); Supp. Figs. 16, 17; ResNet50 replication Supp. Fig. 18 |
| **C17** | LLM-trained RCNNs beat ~13 SOTA ANNs despite orders of magnitude less data | M17 | Fig. 4e; Supp. Fig. 20. **⚠ X1: the location of the single non-significant exception is self-contradictory in the paper** |
| **C18** | The advantage is not a COCO-subset artifact | M14 (ecoset arm), M17 | in-text; **SI-ONLY** for the ResNet50 version — Supp. Fig. 19 |
| **C19** | The advantage is not a feature-count artifact | M15 (512-d assertion), M7 | RSA is parameter-free; 512 vs 2,048 features |
| **C20** | (Discussion synthesis) Higher visual cortex converges on an LLM-alignable format | M19 | aggregate of C1–C19 |

**Consequence for planning:** **six claims (C3, C9, C12, C13, C18, and part of C17) rest *entirely* on the Supplementary Information**, which is absent from this repository [PS §13 R1]. Until M0 fetches the SI, those claims cannot be verified at all — not "not yet verified", but **unverifiable**.

---

## 4. Risks and unknowns

### 4.1 Ranked risk register

Ranked by probability of **silently** producing different numbers (none of these throws an error). This merges [PS §13] and [DG §9], keeping their ids.

| Rank | Risk | Module hit | Why it matters | Mitigation |
|---|---|---|---|---|
| **R1** | **The Supplementary Information is not in this repo.** ~half the paper's statistical support (all pairwise-P matrices, all per-participant maps, all robustness checks) is unavailable. | M0, M18 | Six claims are unverifiable. | **Fetch the SI before anything else.** [PS §13 R1] |
| **R2** | **MI-14: RSA split pairing unstated.** Fixed-across-models (paired) vs redrawn (unpaired). | M7 | **Determines whether every model-vs-model t-test in Figs. 3 and 4 is paired — directly changes every reported P.** Rated HIGHEST in [DG §9.2]. | Fix splits per subject across models; document; run the unpaired variant as a sensitivity check. |
| **R3** | **MI-31: Adam ε = 1×10⁻¹**, 10⁷× the default. Stated, not a typo. | M14 | Anyone who "fixes" it to 1e-8 trains a materially different optimizer → all of Figs. 4b–e change → C13–C17 change. | Use 1e-1 **as written**; verify against the repo; report both if the repo differs. |
| **R4** | **MI-32 + MI-33: the RCNN architecture is underspecified** (timesteps; per-layer channels). | M14 | **The headline result rests on a network that cannot be reconstructed from the paper.** The repo is mandatory, not optional. [PS §13 R3] | M0 gate. |
| **R5** | **MI-15: noise-ceiling correction arithmetic unstated.** | M8 | Rescales every bar in Fig. 3 and every point in Fig. 4e; **can flip marginal significance.** | Run all candidate formulas; report which reproduces the published bar heights [DG §10 step 7]. |
| **R6** | **MI-17 / X3: encoding ridge-fraction granularity is incoherent as written** ("per embedding feature" when the targets are voxels). | M10, M12 | Changes the regularization of **every voxel** → Figs. 1c, 1d, 2a. | Default per-voxel; confirm at the `fracridge` call site in the repo. |
| **R7** | **X6: 512-d pre-readout vs 768-d readout.** The paper never puts "pre-readout" and "512" in one sentence. | M15 | Tapping the wrong tensor breaks Figs. 4c–e. | Assert `shape[1] == 512`. |
| **R8** | **MI-04 + MI-26: per-model preprocessing and layer identity for the 13 competitors are not given.** | M17 | **Fig. 4e's entire ranking — claim C17, the paper's headline — hangs on one Methods clause.** | Use each model's canonical transform; publish the table the paper lacks; sweep alternatives. |
| **R9** | **MI-11: MPNet L2-normalization unstated.** | M3, M14 | Changes both the ridge X-scaling *and* the RCNN's cosine-loss geometry. | Test both settings. |
| **R10** | **R7 [PS §13]: NSD beta-version sensitivity.** `betas_fithrf_GLMdenoise_RR` @1.8 mm is stated, but NSD encoding scores are known to move materially across beta versions. | M1 | Any deviation shifts *all* brain-side results. | Pin the exact volume prep; record checksums. |
| **R11** | **MI-12: fastText/GloVe OOV handling was manual and unlogged.** | M5 | **Fig. 3a/3c control bars are not exactly reproducible even in principle.** | Log and publish every OOV decision. |
| **R12** | **MI-13: ROI-level voxel selection unspecified** (unlike the searchlight case). | M6 | Changes all of Fig. 3 and Fig. 4e. | All voxels, no threshold; sweep a reliability threshold as a sensitivity check. |
| **R13** | **R8 [PS §13]: Fig. 2a is uncorrected and uses 15 hand-picked sentences** chosen by an admittedly ad-hoc procedure. | M12 | C4 is the weakest claim on evidentiary grounds. Reproducible *as run* (sentences are verbatim), but with no correction to fall back on. | Reproduce as run; add an FDR-corrected version as an extension. |
| **R14** | **MI-30: MPNet checkpoint drift** (pinned by name only). | M3 | Every embedding in the paper shifts silently. | Pin the HF revision hash; record it. |
| **R15** | **X1: the CORnet-S exception ROI is self-contradictory** (body = lateral; caption = parietal). | M17 | Cannot both hold. C17's precise wording is unsettled. | Resolve against Supp. Fig. 20 at M0. |

### 4.2 Hidden assumptions to test as first-class experiments

These are stated as assumptions, not gaps, in [DG §9.3]. Each is an alternative the authors did not argue for; each is a candidate extension.

| # | Assumption | Why it exists (what it buys) | How to vary it |
|---|---|---|---|
| **H1** | Every RSA number is a mean over 100×100 **sub**-RDMs, not the full ~10,000² RDM. | A tractability workaround for the RDM's quadratic scale [DG §2.2 `N-RSASAMPLE`]. | Compute the full RDM for one subject/ROI and compare. The paper gives **no bias analysis**. |
| **H2** | Subjects with 54 vs 100 splits feed an **unweighted** N=8 t-test. | Simplicity. | Inverse-variance-weight the group test; check whether any marginal result flips. |
| **H3** | The model correlation (on ~10,000 images) is divided by a ceiling measured on the **515** shared images. Different data, different RDM sizes. | The 515 set is the only set common to all 8 subjects, so it is the only place a between-subject ceiling can be computed. | Compute both quantities on the 515 set and compare the corrected values. |
| **H4** | Fig. 2a inherits all of the encoding model's assumptions and adds hand-picked sentences and **no FDR**. | It is a qualitative demonstration, not a test. | Generate sentences programmatically; apply FDR. |
| **H5** | The category control minimizes **cosine distance on a sigmoid output**, not BCE. | Keeps the loss identical across arms (a fair-comparison argument). | Train the category arm with BCE. **If the control is handicapped by an unnatural loss, C16 and C17 weaken.** The paper does not test this. |
| **H6** | The Fig. 4e x-axis counts **images only** — the LLM's own text-corpus training, which produced the *targets*, is excluded. | It makes the data-efficiency claim expressible on one axis. Authors flag it as out of scope. | Add the LLM's token count to the x-axis and re-plot. **This is precisely the axis the headline claim is made on.** |
| **H7** | `all-mpnet-base-v2` was itself **fine-tuned partly on COCO**, the corpus NSD is built from. | It is the best off-the-shelf sentence encoder for this exact use. | Re-run with an encoder never trained on COCO. |
| **H8** | Session z-scoring + 3-rep averaging is the **only** denoising: no voxel-reliability threshold, no PCA, no GLMsingle refit. | Simplicity and neutrality. | Add a reliability threshold and re-run Figs. 1b/3. |

### 4.3 Data-access gates and licences

| Asset | Gate | Locator |
|---|---|---|
| NSD betas | Data-use agreement at `naturalscenesdataset.org` | [PS §1 header; DG §1 D1] |
| Pennock et al. 2021 food ROIs | **Not distributed with NSD**; must come from ref. 55's supplement | [DG §1 D11] |
| Google Conceptual Captions | Google's terms; text only, images not needed | [DG §1 D8] |
| ecoset | Codeocean release | [DG §1 D18] |
| Paper + code | Paper CC-BY 4.0; code at `github.com/adriendoerig/visuo_llm`, Zenodo `10.5281/ZENODO.15282176` | [PS §1] |
| Competitor checkpoints | thingsvision, brainscore, timm, torch.hub WSL, openai/CLIP, google-research/simclr, StanfordVL/taskonomy | [DG §1 D19] |

**UNKNOWN — needs verification:** neither report states the licence terms of the competitor checkpoints (in particular the WSL ResNeXt weights and CLIP), nor whether NSD's DUA permits redistribution of derived beta artifacts. Check the DUA text and each model card before publishing any derived artifact.

### 4.4 Compute and storage budget

All figures below are marked **[EST]** in [DG §8] — they are that report's derivations from paper-stated quantities, **not paper claims**. Hardware baseline there: 1× A100-40GB + 32-core CPU host.

| Item | Storage | Compute |
|---|---|---|
| NSD download | ~100–150 GB [EST] | download-bound (days) |
| `N-BETAZ` output | 173 GB fp32 / 86 GB fp16 [EST] | minutes–1 h |
| MPNet on 73k × 5 captions | 1.1 GB raw → 224 MB averaged | ~10 GPU-min |
| GCC dictionary (3.1e6 × 768) | 9.5 GB fp32 / 4.8 GB fp16 | 1.5–3 GPU-h |
| Searchlight index | ~650 MB | minutes |
| **`N-RSASAMPLE` (dominant)** | streamed | ~6.5×10¹⁴ FLOP per model × 8 subjects; **several GPU-h per model per full-brain searchlight**; × 14 models × (layers × timesteps × 10 seeds) for the supplementary sweeps → **plausibly thousands of GPU-hours** |
| Ridge (encoding) | ~550 MB/subject at p≈180k | minutes/subject |
| Ridge (decoding) | — | tens of min/subject (wide X; SVD per fold) |
| Retrieval | — | seconds on GPU |
| **RCNN training** | checkpoints | **10–40 GPU-days** for 20 runs — **and the unknown timestep count T multiplies this linearly (MI-32)** |
| ResNet50 ×2 | — | ~0.5 GPU-day each |
| ecoset RCNN | — | **~1 GPU-week** |
| Activation cache | 150 MB per (model, layer, timestep, seed) — the full RCNN grid does **not** fit; stream it | inference-bound, hours |

**The single biggest engineering win** [DG §8]: compute the brain RDMs **once** per (subject, split, sphere) and reuse them across **all** models. This is the entire payoff of merge decision M6 and should be designed in from the start of M7, not retrofitted.

---

## 5. Milestone / checkpoint schedule

Each milestone has a binary "done" condition. A milestone is not done until its condition is *demonstrated*, not merely coded.

| Milestone | Modules | "Done" means |
|---|---|---|
| **M1 — Gaps closed, data on disk** | M0 | The SI is in `paper/`; the release repo is vendored under `code/`; all five BLOCKING items (MI-32, MI-33, MI-15, MI-17, MI-05) have a resolved value **with a `file:line` or SI-figure locator**, or are re-declared missing with a documented default and a planned sweep; **X1 resolved to a single ROI**; NSD betas for 8 subjects readable. |
| **M2 — Brain and text substrates verified** | M1, M2, M3 | Per-session per-voxel z-scored betas pass the mean≈0/sd≈1 test; image counts hit {≈10,000, 6,234, 5,445}; `|D-515| == 515`; `mpnet_caption_emb_73k.npy` is `(73000, 768)` and bit-reproducible; the Supp. Fig. 2 t-SNE structure is qualitatively reproduced. |
| **M3 — Text pipeline validated against a published number** | M4, M5 | **Scrambled-vs-original caption embedding correlation = 0.91, s.d. 0.03** [PS §8 C11]. This is the only exact number available to validate the text side; if it does not reproduce, the encoder configuration (MI-11 normalization, MI-30 checkpoint) is wrong and must be fixed **before** anything downstream. |
| **M4 — RSA core + STOP-LOSS GATE** | M6, M7, M9 | Upper-triangle length is exactly 4,950; split counts are exactly 100/62/54; brain RDMs are cached once and reused. **Then: Fig. 1b reproduces its ventral/lateral/parietal hot-spot pattern at colorbar −0.25…0.25.** [DG §10 step 4]: *if it does not, stop — nothing downstream will work.* |
| **M5 — Ridge family** | M10, M11, M12 | Fig. 1c reproduces (colorbar −0.73…0.73) and **encoding r approaches inter-participant agreement in all ROIs**; Fig. 1d scatter reproduces with ROI-group ordering; Fig. 2b KDE mass falls in 0.3–0.7 and decoded captions are shown to differ from the nearest *training* caption; Fig. 2a reproduces the people/places/food topography — **with the no-FDR status explicitly preserved, not silently corrected.** |
| **M6 — Noise ceiling + Fig. 3** | M8, M18 (E10e, E10g) | All candidate MI-15 formulas have been run, and the one reproducing the published Fig. 3 bar heights is documented as the working assumption. Fig. 3a/3b/3c reproduce, including **both stated exceptions** (multi-hot fails in lateral; fastText fails in EVC; nouns fail in EVC). Claims C6, C7, C8, C10 have a verdict. |
| **M7 — ANN training (post-gate)** | M13, M14 | The COCO∖NSD leakage test passes with an empty intersection; 10 LLM-RCNN + 10 category-RCNN + 2 ResNet50 + 1 ecoset-RCNN are trained with **Adam ε = 1e-1 as written**; the two arms provably differ only in objective + readout activation; validation cosine beats the noise floor. |
| **M8 — Fig. 4b/4c/4d** | M15, M16, M18 (E10h) | Pre-readout tensors assert `shape[1] == 512`; correlations (not RDMs) are averaged across the 10 seeds; the **Fig. 4b asymmetry** reproduces (category ← LLM works; LLM ← category does not); Figs. 4c and 4d reproduce at colorbars −0.06…0.06 and −0.07…0.07. Claims C14, C15, C16 have a verdict. |
| **M9 — Fig. 4e benchmark** | M17 | All 14 models extracted with a **published preprocessing + layer table** (the table the paper does not have, MI-04/MI-26); the three ROI scatters land in their published y-ranges; the ecoset/COCO-subset control reproduces; **C17 is scored against the X1-resolved wording, not against the paper's contradictory text.** |
| **M10 — Verdict and extensions** | M19 | Every claim C1–C20 carries one of: REPRODUCED (with the number), REPRODUCED-QUALITATIVELY (where no number was published — C15), FAILED (with a diagnosis), or **UNVERIFIABLE** (C12, and any claim still SI-blocked). The eight hidden assumptions H1–H8 each have a documented sensitivity result or an explicit "not tested". `reports/review.md` and `reports/experiment_ideas.md` written. |

---

## Missing information

Items required for reproduction that **neither source report could supply from the paper**, collected. Ids preserved from [DG §9] / [PS §12] so this section is a cross-index, not a restatement.

**Blocking (cannot instantiate / cannot run):**
- **MI-32** — number of RCNN recurrent timesteps. Never stated. The network cannot be built. Blocks M14, M15, M16, M17.
- **MI-33** — vNet per-layer channel counts / RF-size gradient. Never stated (only "ten-layer", "foveal RF gradient", "channels doubled" for the ecoset variant, and "512 features" pre-readout). Blocks M14.
- **MI-15** — the noise-ceiling correction *arithmetic*. Never written. Blocks exact values in M8, and therefore Figs. 3 and 4e.
- **MI-17 / X3** — encoding ridge-fraction selection granularity; the Methods sentence is incoherent in the encoding direction. Blocks exact values in M10, M12.
- **MI-05** — NSD int16 scale factor, HDF5 field names, voxel counts. Blocks I/O in M1.

**Silent (will run; will differ):** MI-01 (are the encoding/decoding betas the same as the RSA betas?), MI-14 (RSA split pairing — **highest-risk**), MI-03 (sampler seed), MI-02 (model-RDM metric), MI-13 (ROI voxel selection), MI-16 (ceiling measured on a different image set than the model correlation), MI-11 (MPNet pooling/normalization), MI-30 (MPNet revision hash), MI-18/19/20 (CV folds, standardization, intercept), MI-21 (raw r vs Fisher-z), MI-22 (FDR family), MI-06 (z-score axis), MI-04 (competitor preprocessing), MI-26 (competitor layer identity), MI-07 (K), MI-08 (fastText/GloVe dims and corpora), MI-12 (unlogged OOV handling — **irreproducible by construction**), MI-09 (NLTK POS tagset), MI-10 (48,236 vs 48,238), MI-23/24/25 (weight decay, augmentation, Adam betas), MI-27 (identities of the "other LLMs" — **makes C12 unverifiable**), MI-28 (t-SNE params), MI-29 ("first 71,000" in what order?), MI-34 (Fig. 4b bar values, published nowhere), MI-35 (Fig. 4e per-model training-set sizes), MI-36 (voxel counts p).

**Not supplied by either report, and needed by this plan:**
- **UNKNOWN — needs verification:** the licence/redistribution terms of the competitor checkpoints and of NSD-derived artifacts. Neither report addresses redistribution.
- **UNKNOWN — needs verification:** which COCO release (`train2017`/`val2017` vs 2014) the authors used. [DG §1 D6] explicitly flags the split/year as not stated.
- **UNKNOWN — needs verification:** whether the release repo actually contains the RSA sampler, the fracridge call site, and the vNet definition — i.e. whether M0 can in fact close the blocking gaps. Neither report read the repo ([DG §11]: *"No code was found in this repository"*; the release code was **not read**). If the repo turns out to be incomplete, M14 and M17 are unreachable and the plan must be re-scoped to the RSA + ridge families only (M1–M12, M18-partial), which still cover C1–C12.
- **UNKNOWN — needs verification:** wall-clock estimates in §2 are my scheduling judgement, not derived from either report; only the compute/storage figures in §4.4 are traceable ([DG §8], and those are themselves marked [EST]).

## Conflicts

Carried forward from [PS §11] and [DG §9.4]. Each must be resolved at M0 or explicitly reported as unresolved in the final review.

| # | Conflict | Sources | Effect on this plan |
|---|---|---|---|
| **X1** | **CORnet-S exception ROI.** Body text (p.1225) places the single non-significant comparison in the **lateral** ROI; the Fig. 4 caption (p.1225) places it in the **parietal** ROI. **These cannot both hold.** | both p.1225 | **M17's success criterion cannot be written until this is resolved** against Supp. Fig. 20. C17's precise wording is unsettled. |
| **X2** | **48,236 vs 48,238** RCNN training images, stated on the same page. | p.1230 (both) | M13: reproduce the COCO∖NSD filter from the repo; **do not hardcode either number.** |
| **X3** | Encoding ridge fraction selected "**per embedding feature**" — incoherent when the targets are voxels; the identical sentence is coherent in the decoding section. | p.1229 (both) | M10 default = per voxel; must be confirmed at the `fracridge` call site. |
| **X6** | **512-d pre-readout vs 768-d readout.** The paper never puts "pre-readout" and "512" in one sentence. | p.1225 vs p.1230 | M15 must assert 512. Tapping the 768-d readout breaks Figs. 4c–e. |
| **X7** | **Supp. Fig. 5 is cross-referenced four mutually incompatible ways** (other-LLM comparison / t-SNE of train-test split / cross-participant encoding) across pp. 1222/1223/1224/1228. | four pages | **Supp. Fig. 5's actual content is undetermined from the main text.** C3's evidence base is therefore uncertain until the SI is fetched. |
| **(resolved)** | **X4** — Adam ε is **1×10⁻¹**, not 1×10⁻⁴. An earlier revision of the dependency-graph report recorded 1e-4 and was wrong. | p.1230 | **If any earlier plan or code was seeded from that file, it is wrong.** Tracked as MI-31. |
| **(resolved)** | **X5** — C6's multi-hot exception is "except in the **lateral** ROI"; only the *word-embedding* exception is fastText-specific (in EVC). An earlier revision merged the two. | p.1222 | M6's Fig. 3a success criterion must check **both** exceptions separately. |

## Reproduction risks

Ranked; one sentence each. (Full detail in §4.1.)

1. **The Supplementary Information is absent** — six claims (C3, C9, C12, C13, C18, part of C17) currently cannot be verified at all, so M0 is not optional overhead but the precondition of the project.
2. **MI-14 (RSA split pairing)** — whether the 100-image splits are shared across models decides whether every model-comparison t-test in Figs. 3 and 4 is paired, and therefore changes every reported P value.
3. **Adam ε = 1×10⁻¹** — it is stated and is 10⁷× the default; anyone who "corrects" it silently trains a different optimizer and forfeits Figs. 4b–e.
4. **RCNN architecture underspecified (MI-32, MI-33)** — the paper's headline network cannot be instantiated from the paper, making the release repo mandatory rather than a convenience.
5. **Noise-ceiling arithmetic unstated (MI-15)** — it rescales every bar in Fig. 3 and every point in Fig. 4e and can flip marginal significance.
6. **Encoding ridge granularity incoherent (MI-17/X3)** — the wrong reading changes the regularization of every voxel in Figs. 1c, 1d and 2a.
7. **512-d pre-readout vs 768-d readout (X6)** — one tensor-selection error silently breaks Figs. 4c–e.
8. **Competitor preprocessing and layer choice (MI-04, MI-26)** — Fig. 4e's ranking, and thus the paper's headline claim C17, rests on a single unelaborated Methods clause.
9. **MPNet configuration drift (MI-11, MI-30)** — output normalization and checkpoint revision jointly perturb every embedding, the ridge scaling, and the RCNN loss geometry.
10. **NSD beta-version sensitivity** — encoding scores are known to move materially across beta preparations, so any deviation from `betas_fithrf_GLMdenoise_RR` @1.8 mm shifts all brain-side results.
11. **fastText/GloVe OOV handling was manual and unlogged (MI-12)** — the Fig. 3a/3c control bars are not exactly reproducible even in principle.
12. **The 100-image sub-RDM sampler (H1)** — every RSA number in the paper is a mean over 100×100 sub-RDMs with no published bias analysis, so a "correct" full-RDM reimplementation may legitimately disagree with the paper.agentId: a0951f0674f2d231a (use SendMessage with to: 'a0951f0674f2d231a', summary: '<5-10 word recap>' to continue this agent)
<usage>subagent_tokens: 100873
tool_uses: 3
duration_ms: 360281</usage>