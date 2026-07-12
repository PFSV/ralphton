"""Tests for M1-M4 and the M7 sampler.

Every assertion here checks a value that is either (a) published in the paper, or (b)
forced by the released code. A failure means the reproduction has diverged from one of
those two sources -- not merely that a number moved.

Run:  python -m pytest tests/test_modules.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.revision import config as C
from src.revision import nsd_data, rsa, stimuli

# --------------------------------------------------------------------------------------
# M1 -- NSD conditions. These need only the NSD metadata (no betas).
# --------------------------------------------------------------------------------------
PAPER_IMAGE_COUNTS = {10000, 6234, 5445}   # paper, Methods
PAPER_SPLIT_COUNTS = {100, 62, 54}         # paper, Methods


def test_d515_is_exactly_515():
    """The shared test set is 515 images. [paper; DG D4]"""
    assert len(nsd_data.get_conditions_515()) == 515


def test_shared_1000_is_not_the_first_1000_ids():
    """Guards a real bug we hit: the shared-1000 are a scattered set from `sharedix`,
    NOT the 1,000 lowest 73k ids."""
    s1000 = nsd_data.get_conditions_1000()
    assert len(s1000) == 1000
    assert not np.array_equal(s1000, np.arange(1000))
    assert s1000.min() > 1000  # they start at 73k-id 2950 (0-indexed)


def test_three_repeat_counts_match_paper():
    """Post-averaging image counts must land exactly on {10000, 6234, 5445}."""
    counts = {s: len(nsd_data.get_conditions_3rep(s)[1]) for s in C.SUBJECTS}
    assert set(counts.values()) == PAPER_IMAGE_COUNTS
    assert sum(v == 10000 for v in counts.values()) == 4  # 4 subjects completed 40 sessions
    assert sum(v == 6234 for v in counts.values()) == 2
    assert sum(v == 5445 for v in counts.values()) == 2


def test_every_kept_condition_has_exactly_three_repeats():
    """SOURCE: nsd_get_data_light.py:266 -- `np.sum(conditions == x) == 3` (exactly 3)."""
    for subj in C.SUBJECTS:
        trials, keep = nsd_data.get_conditions_3rep(subj)
        _, counts = np.unique(trials[np.isin(trials, keep)], return_counts=True)
        assert (counts == C.N_REPEATS).all()


def test_d515_is_a_subset_of_every_subjects_pool():
    """The 515 must be held out per subject, so it must be *in* every subject's pool."""
    c515 = nsd_data.get_conditions_515()
    for subj in C.SUBJECTS:
        _, keep = nsd_data.get_conditions_3rep(subj)
        assert np.isin(c515, keep).all()


# --------------------------------------------------------------------------------------
# M7 -- the RSA sampler
# --------------------------------------------------------------------------------------
def test_split_counts_match_paper():
    """100 / 62 / 54 splits, i.e. floor(n_images / 100)."""
    counts = {s: len(rsa.make_splits(s)) for s in C.SUBJECTS}
    assert set(counts.values()) == PAPER_SPLIT_COUNTS
    for subj, n in counts.items():
        assert n == len(nsd_data.get_conditions_3rep(subj)[1]) // C.RSA_SAMPLE_SIZE


def test_splits_are_disjoint_and_100_wide():
    """The release samples WITHOUT replacement and removes each draw from the pool
    (`np.setdiff1d`), so the splits partition the pool -- they are not bootstraps.
    SOURCE: nsd_searchlight_main_tf.py:130-134."""
    for subj in ("subj01", "subj04"):
        splits = rsa.make_splits(subj)
        assert splits.shape[1] == C.RSA_SAMPLE_SIZE
        flat = splits.ravel()
        assert len(np.unique(flat)) == len(flat), "splits overlap -- not a partition"


def test_upper_triangle_length_is_4950():
    """The RDM vector compared against the brain must be exactly 4,950 long."""
    x = np.random.default_rng(0).normal(size=(C.RSA_SAMPLE_SIZE, 32))
    assert len(rsa.rdm(x)) == C.RSA_UPPER_TRI_LEN == 4950


def test_splits_are_paired_across_models():
    """MI-14: the splits must be identical for every model, or the model-vs-model t-tests
    are not well-posed. Our sampler is a pure function of (subject, seed), which enforces
    this by construction. The release enforces it by refusing to run a non-mpnet model
    when no saved split file exists (nsd_searchlight_main_tf.py:136-144)."""
    for subj in ("subj01", "subj03"):
        assert np.array_equal(rsa.make_splits(subj, seed=0), rsa.make_splits(subj, seed=0))


def test_corr_rdms_is_pearson():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=4950), rng.normal(size=4950)
    assert rsa.corr_rdms(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-6)


# --------------------------------------------------------------------------------------
# M2 -- stimuli
# --------------------------------------------------------------------------------------
def test_captions_cover_all_73k_images():
    caps = stimuli.load_captions()
    assert len(caps) == 73000
    assert all(len(c) >= 5 for c in caps), "every NSD image must have >= 5 COCO captions"


def test_multihot_is_91_wide_with_80_active():
    """MI-07: K = 91 (the COCO 'things' superset), of which only the 80 annotated
    categories can ever be set. The 11 deprecated ids are structurally always zero."""
    mh = stimuli.load_multihot()
    assert mh.shape == (73000, C.N_COCO_CATEGORIES)
    assert (mh.sum(0) > 0).sum() == 80


def test_category_list_is_ordered_by_coco_id():
    """Pins the index convention: coco_categories_91[i] is the name of COCO category i+1.
    Checked on the deprecated ids, which is where an off-by-one would show up."""
    names = stimuli.coco_categories_91()
    assert len(names) == 91
    assert names[0] == "person"          # coco id 1
    assert names[9] == "traffic-light"   # coco id 10
    assert names[11] == "signpost"       # coco id 12 == the deleted "street sign"


# --------------------------------------------------------------------------------------
# M3 / M4 -- embeddings
# --------------------------------------------------------------------------------------
def test_caption_embeddings_shape():
    from src.revision import embeddings as E

    e = E.load("captions")
    assert e.shape == (73000, C.MPNET_DIM)


def test_single_pass_embeddings_are_unit_norm_but_means_are_not():
    """MI-11: all-mpnet-base-v2 ships a Normalize module, so a single forward pass is
    unit-norm. The per-image MEAN of several such vectors is not, and the release does not
    re-normalise it. Both facts must hold or the ridge scaling and the RCNN's cosine-loss
    geometry are wrong."""
    from src.revision import embeddings as E

    cat = np.linalg.norm(E.load("categories"), axis=1)  # one string -> one forward pass
    cap = np.linalg.norm(E.load("captions"), axis=1)    # mean over ~5 captions
    assert cat.mean() == pytest.approx(1.0, abs=1e-4)
    assert cap.mean() < 0.95


def test_c11_scrambled_caption_correlation():
    """The paper's ONE exactly-published number for the text pipeline: MPNet is largely
    insensitive to word order, mean r = 0.91 (claim C11).

    We reproduce the MEAN. We do NOT reproduce the published s.d. of 0.03 under any
    reading we could construct -- see reports/modules/M3_M4_text_pipeline.md. The mean is
    the load-bearing part of the claim, so we gate on it.
    """
    from src.revision import embeddings as E

    o, s = E.load("captions"), E.load("scrambled")
    rs = np.array([rsa.model_vs_model(o, s, subj) for subj in C.SUBJECTS])
    assert rs.mean() == pytest.approx(C.SCRAMBLE_TARGET_R, abs=0.015)


# --------------------------------------------------------------------------------------
# M10 -- the ridge core
# --------------------------------------------------------------------------------------
def test_frac_grid_is_the_code_grid_not_the_paper_grid():
    """The paper says 0.05..1.00 step 0.05. The release computes
    np.linspace(1/20, 1+1/20, 20) = 0.05..1.05, step 0.05263 -- the top fraction exceeds
    OLS. SOURCE: nsd_llm_encoding_model.py:25-26."""
    from src.revision.ridge import FRACS

    assert len(FRACS) == 20
    assert FRACS[0] == pytest.approx(0.05)
    assert FRACS[-1] == pytest.approx(1.05)
    assert FRACS[-1] > 1.0


def test_shared_frac_matches_gridsearchcv():
    """MI-17: the release selects ONE shared fraction for the whole model via
    r2_score(multioutput='uniform_average'), not one per voxel. Our chunked, fraction-
    vectorised reimplementation must return exactly what FracRidgeRegressorCV returns, or
    the tractability shortcut is not faithful."""
    from fracridge import FracRidgeRegressorCV

    from src.revision.ridge import FRACS, select_frac

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 40))
    Y = X @ rng.normal(size=(40, 60)) + rng.normal(size=(300, 60)) * 3.0

    mine, _ = select_frac(X, Y, chunk=17)  # chunk < n_targets, to exercise chunking
    ref = FracRidgeRegressorCV(jit=True, fit_intercept=True).fit(X, Y, frac_grid=FRACS)
    assert mine == pytest.approx(float(ref.best_frac_))
