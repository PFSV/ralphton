# Paper Structure — Reverse Engineering Extraction

**Source:** `paper/re_vision.pdf` (18 pages: pp. 1–15 article, pp. 16–18 Nature Portfolio Reporting Summary)

**Title:** High-level visual representations in the human brain are aligned with large language models
**Authors:** Adrien Doerig^1,2,3,9; Tim C. Kietzmann^2,9; Emily Allen^4,5; Yihan Wu^6; Thomas Naselaris^7; Kendrick Kay^4,10; Ian Charest^8,10 (✉ ian.charest@umontreal.ca)
**Affiliations:** 1 FU Berlin (Psych & Education); 2 Univ. Osnabrück (Inst. Cognitive Science); 3 Bernstein Center for Computational Neuroscience Berlin; 4 CMRR, Dept. Radiology, Univ. Minnesota; 5 Dept. Psychology, Univ. Minnesota; 6 Grad. Program in Cognitive Science, Univ. Minnesota; 7 Dept. Neuroscience, Univ. Minnesota; 8 cerebrUM, Dép. de Psychologie, Univ. de Montréal. 9 = equal contribution (A.D., T.C.K.); 10 = joint supervision (K.K., I.C.)
**Venue:** Nature Machine Intelligence, Vol. 7, August 2025, pp. 1220–1234
**DOI:** 10.1038/s42256-025-01072-0
**Dates:** Received 2024-08-19; Accepted 2025-06-04; Published online 2025-08-07
**License:** CC BY 4.0 (Open Access)
**Code:** https://github.com/adriendoerig/visuo_llm (ref. 120; Zenodo v1.0, DOI 10.5281/ZENODO.15282176)
**Data:** http://naturalscenesdataset.org

This document extracts structure only. No interpretation, no summary of arguments.

---

## 1. Sections

| # | Section | Subsections | Pages |
|---|---------|-------------|-------|
| — | Abstract | — | 1220 |
| — | Introduction (untitled lead) | — | 1220–1221 |
| 1 | Results | (a) *A linear mapping from LLM embeddings captures brain responses to natural scenes*<br>(b) *LLMs integrate complex information contained in scene captions that is important to match brain activities*<br>(c) *LLM-trained RCNNs outperform other models of visual responses* | 1221–1226 |
| 2 | Discussion | — | 1227–1228 |
| 3 | Methods | *NSD*; *LLM embeddings for NSD stimuli*; *Word embeddings for NSD stimuli*; *Category labels for NSD stimuli*; *ANN activations for NSD stimuli*; *Quantifying model–brain representational agreement using RSA*; *Encoding model*; *Encoding-model-based brain activity predictions*; *Decoding of LLM embeddings from brain data*; *RCNNs*; *RCNN fine-tuning*; *Other ANNs* (Supervised object category / Supervised scene category / Semi-supervised / Unsupervised); *Predicting brain activations from ANN activations*; *Reporting summary* | 1228–1230 |
| 4 | Data availability | — | 1230 |
| 5 | Code availability | — | 1230 |
| 6 | References | 120 refs | 1230–1233 |
| 7 | Acknowledgements / Author contributions / Competing interests / Additional information | — | 1233 |
| 8 | Reporting Summary (Nature Portfolio form) | Statistics; Software and code; Data; Human participants; Field-specific reporting; Life sciences study design; Materials/systems/methods | 1235–1237 (unpaginated) |

---

## 2. Figures

### Main figures (4)

| Fig | Panels | Content | Method producing it |
|-----|--------|---------|---------------------|
| **1** — *A mapping from LLM embeddings captures visual responses to natural scenes* | **1a** Schematic: image → (7T fMRI → voxel betas) and (COCO caption → LLM → embedding) → RSA branch (Fig. 1b) + encoding-model branch (Figs. 1c, 2a) | pipeline diagram | — |
| | **1b** Searchlight RSA map, group-average Pearson r (LLM ↔ brain), NOT noise-ceiling corrected. Flatmap. Colorbar −0.25…0.25 | Searchlight RSA | two-tailed *t*-test across participants (N=8), BH-FDR, P=0.05 |
| | **1c** Voxel-wise linear encoding model performance (LLM → brain), group-average Pearson r, not noise-ceiling corrected, predicted vs actual betas on test set. Colorbar −0.73…0.73 | Encoding model (fractional ridge) | two-tailed *t*-test across N=8, BH-FDR, P=0.05 |
| | **1d** Scatter: encoding-model performance (x) vs inter-participant agreement (y), one dot per voxel, coloured by ROI group (Non-visual, EVC, Midventral, Ventral, Midlateral, Lateral, Midparietal, Parietal). Axes 0…0.8 | Encoding model vs noise reference | — |
| **2** — *LLM-based linear prediction and decoding of brain activities* | **2a** Functional-contrast maps predicted by the encoding model: People vs Places (left, colorbar −1.52…1.52) and People vs Food (right, colorbar −0.94…0.94), from 5 novel sentences per category. ROI outlines from Allen et al. 2021 (left) / Pennock et al. 2021 (right) | Encoding-model-based brain activity prediction | two-tailed *t*-test across N=8, P=0.05, **no FDR correction** |
| | **2b top** Schematic: brain activity → linear decoding → predicted LLM embedding → nearest-neighbour look-up in 3.1M Google Conceptual Captions → decoded caption | Decoding model | — |
| | **2b bottom-left** Kernel density estimate of embedding prediction performance (Pearson r) per participant (8 curves), with noise-ceiling line | Decoding model on 515 shared test images | noise ceiling = consistency across the 5 human captions/image |
| | **2b bottom-right** Example decoded captions for participants 1, 2, 3, 4 with ranks 102/515, 0/515, 459/515, 255/515 — showing [Human], [Decoder], [Nearest training] captions | qualitative | — |
| **3** — *The match of LLMs to visually evoked brain activities derives from their ability to integrate complex information contained in scene captions* | Layout: 4 columns of ROIs (Early visual, Ventral, Lateral, Parietal); top row = "LLM caption" reference bar in every column. Inset: 'streams' ROI definitions on flatmap. Bars = noise-ceiling-corrected Pearson r, error bars = s.e.m., dots = individual participants (N=8) | ROI-wise RSA | two-tailed *t*-tests across N=8, BH-FDR; stars = 'LLM caption' significantly outperforms the control model (P<0.05) |
| | **3a** *Category information*: LLM, Multi-hot, Fasttext, Glove | | |
| | **3b** *Word types*: LLM nouns, LLM verbs | | |
| | **3c** *Single word average*: LLM, Fasttext, Glove | | |
| **4** — *LLM-trained deep recurrent convolutional networks outperform other models in predicting brain activity* | **4a** Schematic: MS COCO subset (48,000 images) → BLT RCNN (10 recurrent conv layers; bottom-up purple, lateral green, top-down orange; FC readout) → training objective: LLM embedding of caption ("a dog standing on a boat") **or** category multi-hot {dog, boat, …} | architecture diagram | — |
| | **4b** Bar chart: validation performance (cosine similarity) for 4 conditions — Decode category {from LLM-trained RCNN, from category-trained RCNN}; Decode LLM {from LLM-trained RCNN, from category-trained RCNN}. Horizontal reference lines = Category baseline, LLM baseline. y ≈ 0.40–0.70 | RCNN fine-tuning (frozen-weight readouts) | mean over 10 seeds, error bars |
| | **4c** Searchlight RSA contrast map: LLM-trained RCNN (last layer/timestep) **vs** LLM embeddings. Colorbar −0.06…0.06. Inset scatter by ROI group | Searchlight RSA contrast | two-tailed *t*-test across N=8, BH-FDR, P=0.05 |
| | **4d** Searchlight RSA contrast map: LLM-trained **vs** category-trained RCNN (last layer/timestep). Colorbar −0.07…0.07. Inset scatter by ROI group | Searchlight RSA contrast | same |
| | **4e** Three scatter plots (Ventral, Lateral, Parietal): x = training dataset size (# images, log scale, 10^4…10^9); y = representational agreement with ROI (Pearson r, noise-normalized). ~14 models plotted, colour-coded by training-type legend: LLM trained (ours) / Supervised (category) / Supervised (scene) / Unsupervised / Semi-supervised. Ventral y-range 0.15–0.45; Lateral ~0.1–0.5; Parietal ~0.1–0.4 | ROI-wise RSA (pre-readout layer vs ROI RDMs) | noise-ceiling corrected; two-tailed *t*-tests across N=8, BH-FDR, P=0.05; pairwise P values → Supp. Fig. 20 |

**Note on figure images:** for copyright reasons, real COCO images are replaced by copyright-free lookalikes in Figs. 1a, 2b, 4a.

### Supplementary figures (referenced in main text; not in this PDF — 20 total)

| Supp. Fig | Referenced content |
|-----------|--------------------|
| 1 | Descriptive statistics of the COCO captions |
| 2 | 2D t-SNE projection of MPNet-embedded NSD captions |
| 3 | Individual-participant searchlight RSA maps (Fig. 1b) |
| 4 | Individual-participant encoding-model maps (Fig. 1c) |
| 5 | Cross-participant encoding approach; t-SNE of train/test split; comparison of several different LLMs |
| 6 | Additional functional contrasts (Fig. 2a) |
| 7 | (referenced implicitly in Fig. 3 chain) |
| 8 | Encoding + decoding analyses (as Fig. 2a,b) based on LLM embeddings of **category words** → worse performance |
| 9 | Adjectives, adverbs, prepositions → very low alignment with brain representations |
| 10 | LLM embeddings of **scrambled sentences** vs original (mean Pearson r = 0.91, s.d. 0.03) |
| 11 | Other LLMs from the Sentence-Transformers leaderboard; reproduction of Fig. 1b with different LLMs |
| 12 | All pairwise model comparisons for Fig. 3 (corrected P values) |
| 13 | Searchlight RSA of LLM-trained RCNN layer activations |
| 14 | Searchlight maps of all layers and timesteps (early layers ↔ lower visual areas; higher layers ↔ higher visual areas) |
| 15 | Individual participants for Fig. 4c,d |
| 16 | Searchlight contrast maps between all layers and timesteps of LLM- vs category-trained RCNNs |
| 17 | Individual participants for Fig. 4d |
| 18 | Reproduction of Fig. 4d result with a **ResNet50** architecture |
| 19 | ResNet50 reproduction of the ecoset category-control comparison |
| 20 | BH-FDR-corrected P values for all pairwise model comparisons in Fig. 4e |

---

## 3. Tables

**None.** The article contains zero numbered tables. All quantitative results are conveyed via figures. (The Reporting Summary contains form checkboxes, not data tables.)

---

## 4. Models

### 4.1 Language / embedding models (produce target representations)

| ID | Model | Role | Spec / source |
|----|-------|------|---------------|
| M1 | **MPNet** (`all-mpnet-base-v2`) | Primary LLM sentence encoder | 768-d embedding; fine-tuned for sentence-length embeddings; https://www.sbert.net/docs/pretrained_models.html; ref. 39 |
| M2 | Other Sentence-Transformers LLMs | Generality check (Supp. Fig. 11) | from sbert.net leaderboard; "none of the statistical comparisons among LLM models was significant" |
| M3 | **fastText** | Word-embedding control | refs 58, 59; context-of-words based |
| M4 | **GloVe** | Word-embedding control | ref. 60; co-occurrence based |
| M5 | **Multi-hot category vector** | Base/control model | binary vector over COCO category labels |

### 4.2 Trained-in-this-paper ANNs

| ID | Model | Architecture | Training objective | Instances |
|----|-------|--------------|--------------------|-----------|
| M6 | **LLM-trained RCNN** | BLT RCNN, derived from **vNet** (10-layer conv, foveal RF-size gradient, ref. 63) + lateral & top-down recurrence (Kietzmann et al., ref. 115) | minimize **cosine distance** to MPNet caption embedding; 768-d linear readout (no softmax/sigmoid) | 10 random seeds |
| M7 | **Category-trained RCNN** (control) | identical vNet/BLT architecture | minimize cosine distance to multi-hot COCO category encoding; **sigmoid** activation on readout | 10 random seeds |
| M8 | **LLM-trained ResNet50** | ResNet50 (non-pretrained, ref. 74) | LLM embeddings | 1 each |
| M9 | **Category-trained ResNet50** | ResNet50 (non-pretrained) | category labels | 1 each |
| M10 | **RCNN on ecoset** (category control, larger dataset) | same RCNN, **channels doubled** | object classification on ecoset (ref. 63) | — |

**Shared training hyperparameters (M6–M9):** Adam optimizer; lr = 5×10⁻²; **ε = 1×10⁻¹**; 200 epochs; warm-up of 10 epochs with linear lr increase, then cosine decay; batch size **96** for RCNNs, **512** for ResNets; input 128×128 px (largest square crop for rectangular images). All STATED, p. 1230.

> **Note (ε):** the paper says "a learning rate of 5 × 10⁻² and an epsilon of 1 × 10⁻¹" (p. 1230). ε = 1×10⁻¹ is 10⁷× the Adam default (1×10⁻⁸) and effectively damps the adaptive scaling toward plain SGD-with-momentum at lr 0.05. It is stated, not a typo we may silently correct — but it must be verified against the released code before reproduction. See *Reproduction risks* R2.

**Not stated anywhere in the paper (MISSING, required to build the network):** number of recurrent timesteps; per-layer channel counts/widths of vNet; weight decay; data augmentation. The only width given is the "512 features" of the extracted pre-readout representation (p. 1225). Sources to check: ref. 63 (Mehrer et al., ecoset/vNet), ref. 115 (Kietzmann et al., BLT recurrence), and the release repo.

### 4.3 Baseline / comparison ANNs (~13 previously published models, Fig. 4e)

| Category | Models | Source |
|----------|--------|--------|
| Supervised (object category) | CORnet-S (ref. 116, ImageNet, via thingsvision ref. 117); AlexNet (ImageNet, via brainscore); AlexNet-gn (ImageNet, ref. 65); RCNN-on-ecoset (ours, above); ResNet50 (ImageNet, brainscore); **NF-ResNet50** (ImageNet, best-performing CNN at predicting NSD data, ref. 79, via `timm` ref. 119) | — |
| Supervised (scene category) | **ResNet50 trained on Places365** (ref. 82; trained by authors); **ResNet50 trained on Taskonomy scene-cat** (ref. 83, github.com/StanfordVL/taskonomy) | — |
| Semi-supervised | **ResNeXt101_32x8d_wsl** (ref. 84; 914M public images; best brainscore downloadable model; pytorch.org/hub/facebookresearch_WSL-Images_resnext/); **CLIP_RN50_imgs** (visual stream, ResNet50 backbone, WebImageText, ref. 43); **CLIP_ViT_imgs** (visual stream, ViT backbone, github.com/openai/CLIP) | — |
| Unsupervised | **AlexNet** trained with instance-prototype contrastive learning on ImageNet (ref. 65); **ResNet50 + SimCLR** on ImageNet (ref. 85, github.com/google-research/simclr) | — |

Figure-4e legend names observed: `Cornet-s`, `rcnn_ecoset`, `Alexnet`, `Alexnet-gn-sv`, `nf_resnet50`, `Resnet50`, `Alexnetgn_ipcl`, `Google_simclr_rn50`, `Resnet50_Places365`, `Taskonomy_scene_cat`, `Resnext101_32x8d`, `CLIP_RN50_imgs`, `CLIP_ViT_imgs`, `LLM trained (ours)`.

**Layer used for comparison:** pre-readout layer for all models (last layer + last timestep for RCNNs), **except CLIP**, where the final embedding is used.

---

## 5. Datasets

| ID | Dataset | Use | Key numbers |
|----|---------|-----|-------------|
| D1 | **NSD** (Natural Scenes Dataset, ref. 46) | Brain data | 8 participants; 7T fMRI; gradient-echo EPI; **1.8 mm** isotropic; **1.6 s** TR; 30–40 scan sessions/participant; 9,000–10,000 distinct images each; **73,000 unique images total**; 3 repetitions/image; stimulus 3 s on, 1 s gap; continuous recognition task; fixation central. Preprocessing: 1 temporal interpolation (slice timing) + 1 spatial interpolation (head motion), then GLM → single-trial betas. **Volume preparation used: `betas_fithrf_GLMdenoise_RR` (1.8 mm)** |
| D1a | NSD **shared-515** subset | Test set for encoding & decoding | 1,000 images shared across participants; only **515** seen 3× by all 8 participants (3 participants did not complete all trials) |
| D2 | **MS COCO** (refs 47, 48) | Source of NSD images + captions + category labels | 5 human captions/image; object category labels/image |
| D2a | **COCO training subset for RCNNs** | ANN training | **48,236** train / **2,051** validation; the 73,000 NSD images removed from train+val and used as test set |
| D3 | **Google Conceptual Captions** (ref. 57) | Decoder dictionary look-up | **3.1 million** captions → embedded with MPNet → dictionary **D** of 3.1M × 768 |
| D4 | **ecoset** (ref. 63) | Category-control ANN training | — |
| D5 | ImageNet (ref. 81), Places365 (ref. 82), Taskonomy (ref. 83), WebImageText/CLIP (ref. 43), 914M-image WSL set (ref. 84) | Training sets of comparison ANNs | dataset sizes span 10⁴–10⁹ images (x-axis of Fig. 4e) |

**ROI definitions used:** NSD 'streams' ROIs — Early visual cortex (EVC), Ventral, Lateral, Parietal (plus Non-visual, Midventral, Midlateral, Midparietal in maps). Additional ROI sets: Allen et al. 2021 (face/place/body areas: FFA1/2, OFA, EBA, PPA, OPA, pSTS, FFA anterior/posterior sections) and Pennock et al. 2021 (ref. 55; food-selective areas).

---

## 6. Pipelines

### P1 — LLM embedding extraction (Methods: *LLM embeddings for NSD stimuli*)
```
for each NSD image:
    collect 5 COCO human captions
    for each caption: MPNet(all-mpnet-base-v2) -> 768-d vector
    embedding(image) = mean over the 5 caption embeddings
```
Variants:
- Fig. 3a: retrieve COCO **category words** per image → concatenate into a string → feed string to LLM ("LLM" in figure)
- Fig. 3b: extract all **nouns** / all **verbs** of captions using **NLTK** POS tagging (ref. 111) → "LLM nouns" / "LLM verbs"
- Fig. 3c: feed **each word separately** to the LLM → average → "single-word-wise LLM"
- Supp. Fig. 5: repeat with several different LLMs, same 5-caption averaging

### P2 — Word-embedding controls (Methods: *Word embeddings for NSD stimuli*)
```
fastText / GloVe embeddings, additively combined (averaged) across words
Fig. 3a: average word embeddings of the COCO category labels
Fig. 3c: mean embedding across all words of all 5 COCO captions
OOV handling: correct misspelling -> else find similar word in fastText corpus -> else remove.
             If an image ends with no category words: use embedding of 'something' (neutral).
```

### P3 — Brain RDM construction / RSA (Methods: *Quantifying model–brain representational agreement using RSA*)
```
1. restrict to stimuli seen 3x by participant
2. z-score betas within each scanning session, per voxel
3. average the 3 repetitions -> one response estimate per image
4. searchlight: for voxel v, sphere radius = 6 voxels; keep spheres with >50% in-brain voxels;
   exclude out-of-brain voxels from the sphere
5. RDM entries = Pearson correlation distance between activity patterns of stimulus pairs
6. SAMPLING PROCEDURE (scale workaround):
      repeat until the participant's ~10,000 images are exhausted:
          sample 100 images WITHOUT replacement from the remaining pool
          build 100x100 brain RDM  -> upper triangle, length 4,950
          build 100x100 model RDM for the same 100 images (per model / per RCNN layer & timestep)
          Pearson-correlate the two upper triangular vectors, per ROI / per searchlight sphere
      -> 100 independent correlation volumes -> average them
   (N splits = 100 for the 4 participants who completed NSD; 62 or 54 splits for the
    two participants with 6,234 images and the two with 5,445 images)
7. ROI analyses: noise-correct each participant's result
```

### P4 — Noise ceiling (ROI RSA)
```
participant-wise noise ceiling = correlation between this participant's RDM and the
                                 MEAN RDM of the other 7 participants
                                 (all RDMs computed on the shared 515 images)
-> "pitting the model against the average of seven human participants"
-> participant-wise noise-ceiling-corrected correlations are then averaged
```

### P5 — Encoding model: LLM → brain (Methods: *Encoding model*; Fig. 1c)
```
y = brain activity          (n_images x p_voxels)      # whole brain
X = MPNet embeddings        (n_images x 768)
h = fractional ridge regression weights (ref. 54)      (768 x p_voxels)
test set: shared 515 images (seen 3x by all participants)
20 regularization fractions: 0.05 .. 1.0 step 0.05
5-fold cross-validation; best fraction selected PER EMBEDDING FEATURE   # <- verbatim, but see X3
evaluation: Pearson r between predicted and true test-set activities, per voxel
stats: two-tailed t-test across the 8 participants, BH-FDR, P = 0.05
```
**⚠ AMBIGUITY (X3, see §11).** p. 1229 says, of the **encoding** model: "The fraction that best predicted each embedding feature after cross-validation was identified, and used as the final model." But in the encoding direction the regression *targets* are voxels (**y** = brain, **X** = embeddings); there is nothing to select per embedding feature. The identical sentence appears in the **decoding** section (p. 1229), where targets *are* the 768 embedding features and the phrasing is coherent. Most likely a copy-paste from decoding → encoding, in which case the intended rule is **one fraction per voxel**. Do not silently assume either reading: check `fracridge` usage in the release repo. The choice changes every voxel's regularization and therefore Figs. 1c/1d/2a.

### P6 — Encoding-model-based brain activity prediction (Fig. 2a)
```
write a novel sentence -> MPNet -> feed to trained encoding model -> predicted voxel activities
5 sentences per category; average predictions within category; contrast categories on brain maps
Categories/sentences (verbatim, Methods):
  People: 'Man with a beard smiling at the camera.' / 'Some children playing.' /
          'Her face was beautiful.' / 'Woman and her daughter playing.' /
          'Close up of a face of young boy.'
  Places: 'A view of a beautiful landscape.' / 'Houses along a street.' /
          'City skyline with blue sky.' / 'Woodlands in the morning.' /
          'A park with bushes and trees in the distance.'
  Food:   'A plate of food with vegetables.' / 'A hamburger with fries.' /
          'A bowl of fruit.' / 'A plate of spaghetti.' / 'A bowl of soup.'
NOTE: NO FDR correction on these maps (unlike every other map in the paper).
NOTE: sentences were not selected by a precise method (stated explicitly).
```

### P7 — Decoding model: brain → LLM → caption (Methods: *Decoding of LLM embeddings from brain data*; Fig. 2b)
```
voxels: all voxels inside the 'streams' visual ROIs (NSD-provided)
y = caption embeddings   (n_images x 768)
X = brain activity       (n_images x p_voxels)
h = fractional ridge regression   (p_voxels x 768)
test set: shared 515 images; 20 fractions 0.05..1 step 0.05; 5-fold CV; best fraction per feature
evaluation: Pearson r (predicted embedding vs target embedding)
noise ceiling: for each of the 5 human captions, correlate its LLM embedding with the
               averaged embedding of the other 4 -> average the 5 correlations
caption reconstruction: dictionary D = 3.1M Google Conceptual Captions x 768 (MPNet)
                        predicted embedding -> Pearson r with every dictionary entry
                        -> argmax = reconstructed caption
```

### P8 — RCNN training (Methods: *RCNNs*)
```
architecture: vNet (10 conv layers, foveal RF gradient) + lateral + top-down recurrence (BLT)
input: 128x128 px, largest square crop
train/val: 48,236 / 2,051 COCO images (73,000 NSD images excluded)
readout: 768-d linear (no softmax/sigmoid; MPNet embeddings can be +/-)
loss: cosine distance(network output, MPNet caption embedding)
10 instances with different random seeds
Adam, lr 5e-2, eps 1e-1, 200 epochs, 10-epoch linear warm-up, cosine decay, batch 96   # eps as stated p.1230
Category-control twin: identical architecture, sigmoid readout, cosine distance to multi-hot COCO categories
```

### P9 — RCNN fine-tuning / linear readout probe (Methods: *RCNN fine-tuning*; Fig. 4b)
```
collect last-layer/last-timestep activities of each of the 10 instances over the whole NSD dataset
(== freezing weights; no need to recompute activations)
train split: first 71,000 NSD images; test split: last 2,000
train a LINEAR readout to decode multi-hot categories from LLM-trained nets (and vice versa:
   LLM embeddings from category-trained nets); loss = cosine distance
same readout activation / optimizer / hyperparameters as full-RCNN training
average test performance across the 10 seeds
noise floor: mean LLM embedding (resp. mean multi-hot vector) over the 48,236 training images,
             mean cosine distance to the 2,051 validation images' targets
```

### P10 — ANN ↔ brain RSA (Methods: *ANN activations for NSD stimuli*, *Predicting brain activations from ANN activations*)
```
extract activations of layer L at timestep T for all NSD images
preprocess stimuli to each model's expected input range
build model RDMs -> correlate with brain RDMs, ROI-wise and searchlight-wise (same P3 procedure)
for the 10-seed RCNNs: compute individual RDMs per seed, then AVERAGE the correlations with brain data across seeds
Fig. 4e comparison layer: pre-readout layer (CLIP: final embedding)
```

---

## 7. Evaluation Metrics

| Metric | Where used | Notes |
|--------|-----------|-------|
| **Pearson correlation (RSA)** between upper-triangular model RDM and brain RDM | Figs. 1b, 3, 4c, 4d, 4e | called "representational agreement" |
| **Pearson correlation distance** | RDM construction | dissimilarity measure between activity patterns |
| **Noise-ceiling normalization** | Fig. 3 ("noise-ceiling-corrected"), Fig. 4e ("noise-normalized") | leave-one-participant-out mean RDM on shared 515 images. Figs. 1b, 1c, 4c, 4d explicitly **NOT** noise-ceiling corrected |
| **Pearson correlation (encoding)** predicted vs actual voxel betas on test set | Figs. 1c, 1d, 2a | per voxel |
| **Inter-participant agreement** | Fig. 1d | mean Pearson r between a participant's voxel activities and the average of the other 7 on the test images |
| **Pearson correlation (decoding)** predicted vs target 768-d embedding | Fig. 2b | per test image; KDE across the 515 test images |
| **Rank / prediction score** (0 = best, 514 = worst) | Fig. 2b examples | rank of the true caption among the 515 test predictions |
| **Cosine similarity / cosine distance** | Fig. 4b; all ANN training losses | training objective + validation metric |
| **Training-dataset size (# images, log scale)** | Fig. 4e x-axis | 10⁴ … 10⁹ |

### Statistical procedures
- **Test:** two-tailed *t*-test across the **N = 8** NSD participants (parameter-free RSA, participant-level statistics).
- **Multiple comparisons:** Benjamini–Hochberg FDR (ref. 114), **P = 0.05** (Fig. 3 states P < 0.05).
- **Two null hypotheses tested:** (a) individual model maps → model correlation vs 0; (b) model comparisons → significance of the *difference* between two model correlations.
- **Exception:** Fig. 2a functional contrasts have **no FDR correction**.
- **Visualization:** thresholded group maps projected into FreeSurfer's `fsaverage` surface space, shown on a flatmap.
- **Error bars:** s.e.m. across participants (Fig. 3); s.d. across the 10 network instances (Fig. 4b).

---

## 8. Claims

Numbered for downstream verification. Each is stated as extracted, with its evidence locus.

### C1 — LLM embeddings of scene captions match brain RDMs across higher visual cortex
> "the LLM embeddings are able to predict visually evoked brain responses across higher level visual areas in the ventral, lateral and parietal streams"
**Evidence:** Fig. 1b (searchlight RSA), Supp. Fig. 3 (individual), Supp. Fig. 11 (other LLMs).

### C2 — A linear encoding model from LLM embeddings predicts voxel activity across large parts of the visual system
**Evidence:** Fig. 1c, Fig. 1d, Supp. Fig. 4. Encoding performance "approaches the interparticipant agreement in all ROIs."

### C3 — The encoding model generalizes across participants
> "train the model on one participant and test it on the other participants"
**Evidence:** Supp. Fig. 5.

### C4 — The encoding model reproduces known category-selective tuning (people / places / food)
> "Such a contrast revealed classical tuning properties associated with people- and place-areas (FFA, OFA, EBA vs PPA and OPA) as well as food-selective areas"
**Evidence:** Fig. 2a, Supp. Fig. 6. **Caveat stated in paper:** no FDR correction; sentences chosen without a precise method.

### C5 — Scene captions can be reconstructed from brain activity alone (decoding)
> "accurate scene captions can be reconstructed from brain activity"; "The decoder is not simply looking up the closest training item, but instead provides another adequate caption."
**Evidence:** Fig. 2b (KDE vs noise ceiling; example captions at ranks 0, 102, 255, 459 / 515).

### C6 — LLM embeddings of category words beat multi-hot and word embeddings
> "LLM embeddings of category words showed significantly better alignment with brain representations than multi-hot vectors (except in the lateral ROI) and word embeddings (except fasttext in EVC)" — p. 1222, verbatim.
**Evidence:** Fig. 3a; pairwise P values in Supp. Fig. 12.
**Two distinct exceptions:** multi-hot comparison fails in **lateral**; word-embedding comparison fails only for **fasttext in EVC**.

### C7 — Full-caption LLM embeddings beat category-word LLM embeddings, in all ROIs
> "the LLM embeddings of full captions better predicted brain activities in all ROIs by far"
**Evidence:** Fig. 3 (LLM caption reference bar vs Fig. 3a bars); Supp. Fig. 8 (same result for encoding + decoding analyses).

### C8 — Full-caption embeddings beat noun-only and verb-only embeddings
> "the full caption embeddings significantly outperform the noun- and verb-based embeddings across all ROIs, except for noun-based embeddings in EVC"
**Evidence:** Fig. 3b.

### C9 — Adjectives, adverbs, prepositions align poorly with brain representations
**Evidence:** Supp. Fig. 9.

### C10 — Whole-caption embeddings beat the average of individual word embeddings (contextualization matters)
> "in all ROIs, the embeddings of whole captions aligned significantly better with brain data than averaged embeddings of the individual caption words"
**Evidence:** Fig. 3c (LLM, fastText, GloVe single-word averages).

### C11 — MPNet is largely insensitive to word order
> "LLM embeddings from scrambled sentences ... highly correlated with LLM embeddings from the original sentences (mean Pearson correlation across all participants, 0.91; s.d. 0.03)"
**Evidence:** Supp. Fig. 10. **Stated caveat:** scrambled sentences fall outside the LLM's training distribution; may not hold for more complex sentences.

### C12 — Results are not specific to MPNet
> "we tested several other LLMs from the Sentence-Transformers leaderboard and found that they all perform similarly to MPNet ...; none of the statistical comparisons among LLM models was significant"
**Evidence:** Supp. Fig. 11.

### C13 — LLM-trained RCNN activations significantly predict brain responses across the entire visual system
**Evidence:** Supp. Fig. 13; layer/timestep hierarchy in Supp. Fig. 14 (early layers ↔ lower visual areas; higher layers ↔ higher visual areas).

### C14 — The LLM-trained RCNN aligns *better with the brain* than the LLM embeddings it was trained to predict
> "the LLM-trained RCNNs align significantly better with the brain data than the LLM embeddings they were trained to predict"
**Evidence:** Fig. 4c, Supp. Fig. 15.
**Stated counter-argument to dimensionality confound:** RCNN features = 512 dims < LLM's 768 dims.

### C15 — Category labels are decodable from LLM-trained RCNNs, but LLM embeddings are NOT decodable from category-trained RCNNs (asymmetric subsumption)
> "category labels could successfully be read out from LLM-trained RCNNs ... However, the reverse was not true"
**Evidence:** Fig. 4b.

### C16 — LLM-trained RCNNs outperform category-trained RCNNs (matched architecture, data, dimensionality, seeds)
**Evidence:** Fig. 4d, Supp. Fig. 16 (all layers/timesteps), Supp. Fig. 17 (individual participants).
**Replication with different architecture:** Supp. Fig. 18 (ResNet50).

### C17 — LLM-trained RCNNs outperform ~13 state-of-the-art ANNs despite orders-of-magnitude less training data
> "significantly outperform every single other model in the ventral and parietal ROIs, and all but one (which is worse, but not significantly) in the lateral ROI"
> Training data: LLM-trained RCNN = **48,000** COCO images (after removing NSD) vs >1M (ecoset/ImageNet) or hundreds of millions (ResNeXt101_32x8d_wsl, CLIP)
**Evidence:** Fig. 4e; pairwise corrected P values in Supp. Fig. 20.
**⚠ PAPER-INTERNAL CONTRADICTION (X1, see §11).** The body text (p. 1225) places the single non-significant exception in the **lateral** ROI: "significantly outperform every single other model in the ventral and parietal ROIs, and all but one (which is worse, but not significantly) in the lateral ROI." The Fig. 4 caption (p. 1225) places it in the **parietal** ROI: "Our RCNN model significantly outperforms all other models (except CORnet-S, which is not significantly worse in the parietal ROI)." These cannot both hold. Resolve against Supp. Fig. 20 (the pairwise-P matrix) before citing C17.

### C18 — The advantage is not explained by training on a COCO subset rather than the LLM objective
> "we verified that RCNNs trained to predict category labels on ecoset are not outperformed by our RCNNs trained to predict category labels on our subset of COCO (and both are outperformed by our LLM-trained RCNN)"
**Evidence:** in-text; reproduced with ResNet50 in Supp. Fig. 19.

### C19 — The advantage is not explained by feature-count / dimensionality
**Evidence:** (a) RSA is parameter-free, not directly biased by feature dimension (refs 70–73); (b) LLM-trained models have 512 features vs ResNet's 2,048.

### C20 (Discussion-level synthesis) — The visual system converges, across higher-level regions, onto a representational format alignable with LLM embeddings of scene descriptions, driven by the LLM's ability to integrate complex information across entire captions.

### Explicitly acknowledged limitations (Discussion)
- L1: NSD's continuous-recognition task may have induced **internal captioning** by participants → alignment could be task-driven. Not fully ruled out; no NSD data at other tasks available.
- L2: LLM embeddings are not fully interpretable; which elements of scene captions drive the match remains open.
- L3: LLM training-data size is not factored into the Fig. 4e x-axis (only image counts are). "Whether these data need to be factored into the training set size estimates is an open question that is beyond the scope of this Article."
- L4: Results do not imply visual representations have recursivity/syntax.

---

## 9. Dependency Summary (what must exist to reproduce)

| Requirement | Specific artifact |
|-------------|-------------------|
| Brain data | NSD `betas_fithrf_GLMdenoise_RR`, 1.8 mm volume prep; 8 participants; NSD 'streams' ROIs; Allen et al. 2021 + Pennock et al. 2021 ROI definitions; FreeSurfer `fsaverage` for flatmaps |
| Stimuli/annotations | COCO images + 5 captions/image + category labels; NSD↔COCO id mapping |
| External corpora | Google Conceptual Captions 3.1M (decoder dictionary); fastText & GloVe vectors; ecoset; ImageNet; Places365; Taskonomy |
| Models to download | `all-mpnet-base-v2`; CORnet-S (thingsvision); AlexNet/ResNet50 (brainscore); NF-ResNet50 (timm); ResNeXt101_32x8d_wsl (torch.hub); CLIP RN50 + ViT; SimCLR RN50; AlexNet-gn IPCL |
| Libraries named | Sentence-Transformers; NLTK (ref. 111); fracridge (fractional ridge regression, ref. 54); thingsvision; timm; PyTorch |
| Compute | 10 RCNN seeds × 2 objectives × 200 epochs @128×128 on 48,236 images; + ResNet50 replications; + full-brain searchlight RSA × 8 participants × 100 splits × all model/layer/timestep combinations |
| Code | https://github.com/adriendoerig/visuo_llm |

---

## 10. Reporting Summary — extracted facts

- Statistics: sample size, distinctness of measurements, test names/sidedness, covariates, multiple-comparison adjustment, central tendency + variation, test statistic/CI/effect size/P values, effect sizes → all **confirmed present**. Bayesian analysis and hierarchical designs → **n/a**.
- Data collection: "data collection was described previously in Allen et al. 2022".
- Data analysis: "the complete codebase is available at https://github.com/adriendoerig/visuo_llm".
- Data availability: "data is openly available at naturalscenesdataset.org".
- Ethics: "the ethics was approved by the universite de montreal CPREC."
- Sample size justification: "details are published in the Allen et al. 2022 manuscript."
- **Blank fields** in the Life-sciences study-design section: Data exclusions, Replication, Randomization, Blinding.
- Field: Life sciences. MRI-based neuroimaging box: marked **n/a** (checkbox in the "n/a" column).

---

## 11. Conflicts

Places where the paper disagrees with itself, or where an earlier version of this report disagreed with the paper. Every row was checked against the PDF page cited.

| # | Conflict | Sources | Status |
|---|----------|---------|--------|
| **X1** | **CORnet-S exception ROI.** Body text puts the one non-significant model comparison in the **lateral** ROI; the Fig. 4 caption puts it in the **parietal** ROI. | Body p. 1225 vs Fig. 4 caption p. 1225 | **Unresolved.** Paper-internal. Resolve against Supp. Fig. 20. Affects the precise wording of C17. |
| **X2** | **RCNN training-set size.** "This resulted in 48,236 COCO images for training, 2,051 for validation" vs, later on the same page, "the mean LLM embedding … across the 48,238 images used to train the RCNNs". Fig. 4a and the Discussion both round to "48,000". | p. 1230 (both) ; Fig. 4a p. 1226 ; p. 1225 | **Unresolved.** Paper-internal, 2-image discrepancy. Immaterial to results, but signals the COCO-minus-NSD filter is not exactly specified — reproduce the filter from the repo, do not hardcode either number. |
| **X3** | **Encoding-model ridge fraction selected "per embedding feature"** — incoherent in the encoding direction, where targets are voxels. Identical sentence is coherent in the decoding section. | p. 1229 (Encoding model) vs p. 1229 (Decoding) | **Unresolved.** Likely authorial copy-paste; intended rule is probably per-voxel. See §6/P5. |
| **X4** | *(corrected in this report)* Adam **ε = 1×10⁻¹**, not 1×10⁻⁴. A prior revision of this report recorded 1×10⁻⁴. | p. 1230 | **Fixed.** Paper is unambiguous. |
| **X5** | *(corrected in this report)* C6's multi-hot exception is "**except in the lateral ROI**", with no "fasttext" qualifier; only the *word-embedding* exception is fasttext-specific (in EVC). A prior revision merged the two. | p. 1222 | **Fixed.** |
| **X6** | **512 vs 768 features.** RCNN readout is 768-d (p. 1230), but the representations compared to brain data are stated as 512-d (p. 1225). | p. 1225 vs p. 1230 | **Resolved by inference.** The 512-d is the **pre-readout** layer — consistent with Fig. 4e's stated use of pre-readout activations (p. 1230). A reimplementer who uses the 768-d readout instead will not reproduce Figs. 4c–e. |

---

## 12. Missing information

Required to reproduce, absent from the main article + Reporting Summary. **The Supplementary Information is not in this repository** (`paper/` holds only `re_vision.pdf`), so everything below marked *(supp?)* may in fact be stated there and should be checked first.

**Blocking — cannot build the model without these**

| # | Missing | Why it blocks | Where to look |
|---|---------|---------------|---------------|
| N1 | **Number of RCNN recurrent timesteps.** "Timestep" is load-bearing in Figs. 4c/4d and Supp. Figs. 14/16, but no count appears anywhere in Methods. | The network cannot be instantiated. | ref. 115 (Kietzmann et al.); release repo |
| N2 | **vNet per-layer channel counts / RF-size gradient.** Only "ten-layer", "foveal RF gradient", and "we doubled the number of channels" (for the ecoset variant) are given. | The network cannot be instantiated. | ref. 63 (Mehrer et al.); repo |
| N3 | **Noise-ceiling normalization formula.** Figs. 3 and 4e report "noise-ceiling-corrected"/"normalized" Pearson r; the arithmetic (divide? subtract? clip at 1?) is never written. | Rescales every bar in Fig. 3 and every point in Fig. 4e; can flip marginal significance. | repo |
| N4 | **Encoding-model ridge selection granularity** — see X3. | Changes regularization of every voxel. | repo |

**Non-blocking but reproduction-affecting**

| # | Missing | Effect |
|---|---------|--------|
| N5 | Identities of the "several other LLMs from the Sentence-Transformers leaderboard" (p. 1224) — never named. *(supp?)* | C12 unverifiable. |
| N6 | Whether the 100-image RSA splits are **held fixed across models**, or resampled per model (p. 1228–1229 does not say). | Determines whether the model-vs-model *t*-tests in Figs. 3/4 are paired. Directly affects every reported *P*. |
| N7 | How **ROI-level** RDMs are built (voxel selection / reliability threshold within an ROI). The *searchlight* case is fully specified (radius 6 voxels, >50% in-brain); the ROI case is not. | Changes all of Fig. 3 and Fig. 4e. |
| N8 | Per-model **image preprocessing** for the 13 comparison ANNs — "we preprocess stimuli to match the input range expected by each model" (p. 1228), no table. | Changes the Fig. 4e ranking, i.e. C17. |
| N9 | Per-model **layer identity** in Fig. 4e beyond "pre-readout layer (CLIP: final embedding)". | Same as N8. |
| N10 | fastText/GloVe **OOV handling was manual and unlogged**: "we either corrected the misspelling, found a similar word in the fasttext corpus, or removed them" (p. 1228). No list of corrections. | Fig. 3a/3c control bars are not exactly reproducible, by construction. |
| N11 | **NLTK version / POS tagset** used to split nouns vs verbs (p. 1228). | Changes Fig. 3b. |
| N12 | **MPNet checkpoint revision hash.** `all-mpnet-base-v2` is pinned by name only. | A different snapshot silently changes every embedding in the paper. |
| N13 | Number of voxels *p* in the encoding ("full brain") and decoding ("all voxels in streams ROIs") models. | Not needed to run, but needed to sanity-check a reimplementation. |
| N14 | Fig. 4b exact bar values — readable only off the axis; no numbers in text or caption. | C15 can be reproduced qualitatively but not checked numerically. |
| N15 | Training-set sizes plotted on the Fig. 4e x-axis for each comparison model (only COCO 48k, ImageNet >1M, WSL 914M are given numerically). | The x-axis of Fig. 4e cannot be reconstructed exactly. |

---

## 13. Reproduction risks

Ranked by probability of silently producing different numbers. "Silently" is the operative word: none of these throws an error.

| Rank | Risk | Consequence |
|------|------|-------------|
| **R1** | **The Supplementary Information is not in this repo.** ~half the paper's statistical support (all pairwise-P matrices, all per-participant maps, all robustness checks) is unavailable, and 6 of the 20 claims (C3, C9, C12, C13, C18, and part of C17) rest *entirely* on it. | Those claims currently cannot be verified at all. **Fetch the SI before anything else.** |
| **R2** | **Adam ε = 1×10⁻¹** (p. 1230). Stated, but 10⁷× the default. Anyone who "fixes" it to 1e-8 — or who read the earlier, wrong 1e-4 in this report — trains a materially different optimizer. | Changes all RCNN results (Figs. 4b–e), i.e. the paper's headline claims C13–C17. |
| **R3** | **RCNN architecture is underspecified** (N1, N2). The headline result rests on a network whose depth-per-timestep and widths cannot be reconstructed from the paper. | Figs. 4c–e not reproducible from the paper alone. Repo is mandatory, not optional. |
| **R4** | **The 100-image RSA sampling scheme** (100/62/54 splits, averaged) is unusual, and its pairing structure across models is unstated (N6). | Changes both point estimates and every *t*-test in Figs. 3 and 4. |
| **R5** | **Noise-ceiling normalization formula unstated** (N3). | Rescales Fig. 3 and Fig. 4e; can flip marginal significance. |
| **R6** | **512-d pre-readout vs 768-d readout** (X6). The paper never says "pre-readout" and "512" in the same sentence; the reader must infer it. | Getting it wrong changes Figs. 4c/4d/4e. |
| **R7** | **NSD beta version.** `betas_fithrf_GLMdenoise_RR` @1.8 mm is stated, as are session-wise z-scoring and 3-repetition averaging — but NSD encoding scores are known to move materially across beta versions. | Any deviation shifts all brain-side results. |
| **R8** | **Fig. 2a is uncorrected and uses 15 hand-picked sentences** selected by an admittedly ad-hoc procedure ("We did not have a precise method", p. 1229). The sentences are given verbatim, so this is reproducible *as run* — but there is no correction to fall back on and no principled way to vary them. | C4 is the weakest claim in the paper on evidentiary grounds. |
| **R9** | **Comparison-ANN layer + preprocessing choices** (N8, N9). Fig. 4e's entire ranking — the basis of C17 — depends on one clause of Methods. | C17 is the most sensitive claim to reimplementation detail. |
| **R10** | **Manual, unlogged OOV handling** for fastText/GloVe (N10). | Fig. 3a/3c control bars are not exactly reproducible even in principle. |
| **R11** | **MPNet checkpoint drift** (N12). | Every embedding in the paper shifts silently. |

---

## 14. Provenance of this report

- **Source read:** `paper/re_vision.pdf`, all 18 pages (journal pp. 1220–1234 + Reporting Summary).
- **Sources NOT read, and therefore not verified here:** the Supplementary Information (Supp. Figs. 1–20 — referenced ~25× in the main text, absent from `paper/`); the release code (https://github.com/adriendoerig/visuo_llm, Zenodo 10.5281/ZENODO.15282176); the NSD data itself; all 120 cited references.
- **Method:** independent re-extraction from the PDF, diffed against the previous revision of this report. Every disagreement between the two was adjudicated by re-reading the cited PDF page — see §11.
- **Classification convention** used in §§11–13: *stated* = written in the paper (page cited); *inferred* = compelled by what is written (premises + reasoning given); *missing* = required for reproduction, absent from every source read. §§1–10 are extraction-only and cite pages/figures throughout.
