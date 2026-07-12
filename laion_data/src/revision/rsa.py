"""M7 -- RDM builder and the 100-image RSA sampler (`N-RDM`, `N-RSASAMPLE`).

Purpose
-------
The load-bearing procedural core of the paper. A single shared sampler serves Figs. 1b, 3,
4c, 4d, 4e and every supplementary searchlight.

Procedure (verbatim from the release)
-------------------------------------
Repeatedly draw 100 images WITHOUT replacement from the participant's 3x-seen pool,
removing them from the pool each time (`np.setdiff1d`), until the pool is exhausted. This
yields floor(n_images / 100) DISJOINT splits -- a partition, not independent bootstraps:
100 / 62 / 54 splits for the 10,000 / 6,234 / 5,445-image subjects. For each split, build a
100x100 brain RDM and a 100x100 model RDM on the SAME 100 images, Pearson-correlate their
upper triangles (length exactly 4,950), and average the resulting correlations.

SOURCE: nsd_searchlight_main_tf.py:121-146.

MI-14 (the paper's highest-risk silent gap) -- RESOLVED
    The splits are PAIRED: drawn once per subject, saved to disk, and reused for every
    model. The release enforces this so aggressively that any model other than mpnet which
    finds no saved split file raises FileNotFoundError with the comment "we try to load the
    100x100 samples used for the original MPNET sampling procedure, and reapply them for
    subsequent models, for fair comparisons".

MI-03 (sampler seed) -- UNRESOLVABLE
    The release calls `np.random.choice` with NO seed anywhere in the repository. The
    original splits are therefore recoverable only from the authors' saved .npy files,
    which are not published. We seed explicitly (config.RSA_SPLIT_SEED) and sweep seeds to
    show the published statistics are seed-insensitive.

MI-02 (model-side distance metric) -- RESOLVED
    Correlation distance on both sides. The brain RDM is HARDCODED to correlation distance
    in the release's TF searchlight (tf_utils.py:63-65); the model RDM takes an argument
    which both example scripts set to "correlation".
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from . import config as C
from . import nsd_data


def make_splits(subj: str, seed: int = C.RSA_SPLIT_SEED) -> np.ndarray:
    """Disjoint 100-image splits for one subject. -> (n_splits, 100) int, POSITIONAL.

    Indices are positions into the subject's sorted 3x-seen condition list, matching the
    release (which samples `range(subj_n_images)`, not raw 73k ids).
    """
    _, keep = nsd_data.get_conditions_3rep(subj)
    n = len(keep)
    n_splits = n // C.RSA_SAMPLE_SIZE

    rng = np.random.default_rng(seed)
    pool = np.arange(n)
    splits = []
    for _ in range(n_splits):
        choice = rng.choice(pool, C.RSA_SAMPLE_SIZE, replace=False)
        choice.sort()
        splits.append(choice)
        pool = np.setdiff1d(pool, choice)
    return np.asarray(splits)


def rdm(patterns: np.ndarray, metric: str = C.RDM_METRIC) -> np.ndarray:
    """Condensed RDM (upper triangle) for (n_items, n_features). -> (n_items*(n_items-1)/2,)"""
    return pdist(patterns, metric=metric).astype(np.float32)


def corr_rdms(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two condensed RDMs. Matches `corr_rdms` in utils/utils.py:22-26."""
    az = (a - a.mean()) / a.std()
    bz = (b - b.mean()) / b.std()
    return float(np.dot(az, bz) / len(az))


def model_vs_model(
    emb_a: np.ndarray, emb_b: np.ndarray, subj: str, seed: int = C.RSA_SPLIT_SEED
) -> float:
    """Mean RDM correlation between two 73k-indexed model embeddings, for one subject.

    Uses that subject's own image pool and the shared 100-image splits, exactly as the
    paper's model-vs-brain analyses do -- which is what makes the resulting number
    comparable to the paper's published "across participants" statistics.
    """
    _, keep = nsd_data.get_conditions_3rep(subj)
    a = emb_a[keep]
    b = emb_b[keep]
    rs = []
    for idx in make_splits(subj, seed=seed):
        ra = rdm(a[idx])
        rb = rdm(b[idx])
        assert len(ra) == C.RSA_UPPER_TRI_LEN, f"upper-tri length {len(ra)} != {C.RSA_UPPER_TRI_LEN}"
        rs.append(corr_rdms(ra, rb))
    return float(np.mean(rs))
