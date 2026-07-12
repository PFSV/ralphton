"""Central configuration for the re:vision reproduction (Doerig et al. 2025, NMI).

Every constant here is traceable to either (a) the paper, (b) the release repo
`code/visuo_llm`, or (c) the ground-truth `hparams.txt` shipped inside the released
checkpoints. The `SOURCE:` tag on each block says which, so that no value in this
pipeline is a magic number.

Where the paper and the released code disagree, the CODE value is used (the code is
what produced the published figures) and the disagreement is recorded in
`reports/modules/M0_gap_resolution.md`.
"""

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
NSD_DIR = DATA / "nsd"
COCO_DIR = DATA / "coco"
DERIV = DATA / "derivatives"  # everything this pipeline computes
REPORTS = ROOT / "reports"
CODE = ROOT / "code"

NSD_S3 = "https://natural-scenes-dataset.s3.amazonaws.com"

# --------------------------------------------------------------------------------------
# NSD structure
# SOURCE: NSD dataset; visuo_llm/src/nsd_visuo_semantics/utils/nsd_get_data_light.py
# --------------------------------------------------------------------------------------
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]

# Sessions actually completed per subject. NSD ran 40 sessions for subj 1,2,5,7 and
# fewer for the rest. `get_conditions_515` in the release code uses n_sessions=40 for
# ALL subjects (nsd_get_data_light.py:434), which is only safe because missing sessions
# simply contribute no trials. We follow the release code and cap at the true count.
# SOURCE: NSD data release (behav/responses.tsv SESSION column is the ground truth;
# these values are asserted against it in tests/test_m1_nsd_betas.py).
N_SESSIONS = {
    "subj01": 40, "subj02": 40, "subj03": 32, "subj04": 30,
    "subj05": 40, "subj06": 32, "subj07": 40, "subj08": 30,
}

# The beta preparation named in the paper (Methods, "fMRI data").
BETA_VERSION = "betas_fithrf_GLMdenoise_RR"

# Two target spaces are used by the paper, for DIFFERENT analyses.
# SOURCE (correction to reports/reproduction_plan.md, which claimed fsaverage was only
# for visualisation):
#   - searchlight RSA  -> func1pt8mm  (nsd_searchlight_main_tf.py:26)
#   - ROI RSA, encoding, decoding -> fsaverage (nsd_roi_analyses.py:19, roi_utils.py:65,
#     nsd_llm_encoding_model.py, nsd_decode_llm.py)
TARGETSPACE_SEARCHLIGHT = "func1pt8mm"
TARGETSPACE_ROI = "fsaverage"

# int16 -> percent-signal-change scale factor. Applied ONLY for func1pt8mm; the
# fsaverage .mgh betas are already scaled.
# SOURCE: nsd_get_data_light.py:190 `zscore(img/300., axis=cond_axis)`  [MI-05]
BETA_SCALE_FUNC1PT8MM = 300.0

# fsaverage vertex count (lh+rh concatenated).
N_VERTICES_FSAVERAGE = 327684

# 'streams' ROI labels.
# SOURCE: nsddata/freesurfer/fsaverage/label/streams.mgz.ctab (downloaded; verified)
STREAMS_LABELS = {
    0: "Unknown", 1: "early", 2: "midventral", 3: "midlateral",
    4: "midparietal", 5: "ventral", 6: "lateral", 7: "parietal",
}
# The four ROIs the paper actually reports (Fig. 3, Fig. 4e).
STREAMS_MAIN_ROIS = ["early", "ventral", "lateral", "parietal"]

# Number of times a stimulus must have been seen to enter the analysis.
# SOURCE: nsd_get_data_light.py:266 `np.sum(conditions == x) == 3`
N_REPEATS = 3

# --------------------------------------------------------------------------------------
# RSA  (M6, M7, M8)
# --------------------------------------------------------------------------------------
# SOURCE: nsd_searchlight_main_tf.py:130 `np.random.choice(all_conditions, 100, replace=False)`
RSA_SAMPLE_SIZE = 100
RSA_UPPER_TRI_LEN = RSA_SAMPLE_SIZE * (RSA_SAMPLE_SIZE - 1) // 2  # == 4950

# Splits are drawn ONCE per subject and REUSED for every model (paired design).
# The release code has NO RNG seed (MI-14 / MI-03) -- the original splits are therefore
# unrecoverable. We seed explicitly so that OUR pipeline is deterministic, and we sweep
# seeds as a sensitivity analysis.
# SOURCE: nsd_searchlight_main_tf.py:121-146 (no seed anywhere in that file)
RSA_SPLIT_SEED = 0
RSA_SEED_SWEEP = [0, 1, 2]

# Distance metric. Brain RDM is hardcoded to correlation distance in the release TF
# code (tf_utils.py:63-65); model RDM takes the `rdm_distance` argument, which both
# example scripts set to "correlation".
# SOURCE: examples/searchlight_analyses.py:8, examples/roi_analyses.py:15  [MI-02]
RDM_METRIC = "correlation"

# Searchlight geometry.
# SOURCE: nsd_searchlight_main_tf.py:24 `radius = 6`; :80 `RSASearchLight(mask, radius=radius, thr=.5)`
# NOTE: radius is in VOXELS (=10.8 mm at 1.8 mm iso) and the comparison is STRICT `<`.
SEARCHLIGHT_RADIUS_VOXELS = 6
SEARCHLIGHT_MIN_IN_BRAIN_FRAC = 0.5

# --------------------------------------------------------------------------------------
# Ridge  (M10, M11, M12)
# --------------------------------------------------------------------------------------
# SOURCE: nsd_llm_encoding_model.py:25-26
#   n_alphas = 20; fracs = np.linspace(1/n_alphas, 1+1/n_alphas, n_alphas)
# NOTE this is NOT the paper's stated "0.05 to 1.00 in steps of 0.05": the actual grid
# runs to 1.05 with step 0.05263.  [X3-adjacent; recorded in M0 report]
RIDGE_N_FRACS = 20
RIDGE_FRAC_MIN = 1 / RIDGE_N_FRACS          # 0.05
RIDGE_FRAC_MAX = 1 + 1 / RIDGE_N_FRACS      # 1.05

# fracridge's FracRidgeRegressorCV leaves cv=None -> sklearn 5-fold KFold(shuffle=False),
# i.e. CONTIGUOUS folds over NSD-id-sorted rows, and selects ONE SHARED fraction for the
# whole model via r2_score(multioutput="uniform_average").  [MI-17 / X3 -- RESOLVED]
RIDGE_CV_FOLDS = 5
RIDGE_FIT_INTERCEPT = True

# --------------------------------------------------------------------------------------
# Text encoder  (M3, M4, M5)
# --------------------------------------------------------------------------------------
# SOURCE: examples/get_embeddings.py:27
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
# The release code pins no revision (MI-30). We pin one so our embeddings are stable.
MPNET_REVISION = "9a3225965996d404b775526de6dbfe85d3368642"
MPNET_DIM = 768
# The release calls `embedding_model.encode(sentences)` with all defaults
# (embedding_models_zoo.py:76). all-mpnet-base-v2's own modules.json contains a Normalize
# module, so outputs ARE unit-norm regardless. [MI-11 -- RESOLVED: normalised]
MPNET_NORMALIZE = True

# COCO categories: the release uses the 91-entry "things" superset, NOT the 80-class
# subset.  SOURCE: visuo_llm/src/nsd_visuo_semantics/get_embeddings/word_lists.py:18-110
# (len == 91), corroborated by blt_mpnet/example.py:46 `OUTPUT_DIM = ... else 91`.
# [MI-07 -- RESOLVED: K = 91]
N_COCO_CATEGORIES = 91

# POS tagsets (NLTK / Penn Treebank).
# SOURCE: nsd_embeddings_utils.py:120. Note nouns EXCLUDE proper nouns (NNP/NNPS).
POS_TAGS = {
    "noun": ["NN", "NNS"],
    "verb": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "adjective": ["JJ", "JJR", "JJS"],
    "adverb": ["RB", "RBR", "RBS"],
    "preposition": ["IN"],
}
# Fallbacks when an image yields no word of the requested type.
# SOURCE: get_nsd_noun_embeddings_simple.py:104 ("something"),
#         get_nsd_verb_embeddings_simple.py:107 ("is"),
#         get_nsd_sentence_embeddings_categories_simple.py:75 ("something")
FALLBACK_NOUN = "something"
FALLBACK_VERB = "is"
FALLBACK_CATEGORY = "something"

# The scrambling control (Supp. Fig. 10 / claim C11) is unseeded in the release
# (nsd_embeddings_utils.py:146-150). We seed it.
SCRAMBLE_SEED = 0

# Published target for the scrambled-vs-original control: mean r = 0.91, s.d. = 0.03.
# This is the ONLY exact published number that validates the text pipeline with no
# brain data, so it is our first hard gate.  SOURCE: paper, claim C11.
SCRAMBLE_TARGET_R = 0.91
SCRAMBLE_TARGET_SD = 0.03

# --------------------------------------------------------------------------------------
# RCNN  (M14, M15, M16) -- ground truth from the released checkpoints' hparams.txt
# SOURCE: code/checkpoint_code/blt_vNet_half_channels_mpnet_Dec23_seed1/hparams.txt
#         code/checkpoint_code/.../_code_used_for_training/models/blt_vNet.py
# --------------------------------------------------------------------------------------
RCNN_TIMESTEPS = 6                       # [MI-32 -- RESOLVED]
RCNN_BASE_CHANNELS = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
RCNN_DIVIDE_CHANNELS = 2                 # "half_channels"
RCNN_CHANNELS = [c // RCNN_DIVIDE_CHANNELS for c in RCNN_BASE_CHANNELS]  # [MI-33 -- RESOLVED]
RCNN_KERNELS = [7, 7, 5, 5, 3, 3, 3, 3, 1, 1]
RCNN_POOL_LAYERS = [3, 4, 9]
RCNN_N_LAYERS = 10
# Layer 9 has 1024//2 = 512 channels -> GlobalAvgPool2D -> 512-d pre-readout. [X6 -- RESOLVED]
RCNN_PREREADOUT_DIM = RCNN_CHANNELS[-1]  # == 512
RCNN_PREREADOUT_LAYER = f"layernorm_layer_{RCNN_N_LAYERS - 1}_time_{RCNN_TIMESTEPS - 1}"

# Training hyperparameters, verbatim from hparams.txt.
RCNN_TRAIN = {
    "optimizer": "adam",
    "learning_rate": 0.05,
    "optim_epsilon": 0.1,      # [MI-31] 1e7x the Adam default. STATED, not a typo.
    "clip_norm": 500,          # NOT mentioned anywhere in the paper.
    "regularize": 1e-06,       # L2 on conv+dense kernels. [MI-23 -- RESOLVED]
    "dropout_rate": 0.0,
    "n_epochs": 200,
    "n_warmup_epochs": 10,
    "learning_rate_schedule": "cosine",
    "batch_size": 256,         # PAPER SAYS 96. The checkpoint says 256. See M0 report.
    "image_size": 128,
    "image_normalization": "[-1,1]",
    "norm_type": "LN",
    "activation": "relu",
    "embedding_loss": "cosine",  # BOTH arms. Category arm = cosine on a sigmoid, not BCE.
    "use_mixed_precision": 1,
}
# The loss is applied at EVERY timestep (deep supervision) -- task.py:91-95 loops over
# all `network_output_layers`. The paper does not mention this.
RCNN_DEEP_SUPERVISION = True
