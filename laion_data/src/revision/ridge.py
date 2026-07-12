"""M10 + M11 + M12 -- Fractional ridge: encoding, decoding, and functional contrasts.

One ridge core serves both directions. They differ ONLY in which matrix is X and which is
y, and in the voxel mask:

  ENCODING  (Figs. 1c, 1d, 2a) : X = MPNet embedding (n, 768)   y = betas, WHOLE BRAIN
  DECODING  (Fig. 2b)          : X = betas, 'streams' ROIs only  y = MPNet embedding (n, 768)

Test set for both: the 515 images shared by all 8 subjects, held out.
SOURCE: nsd_llm_encoding_model.py, nsd_decode_llm.py.

MI-17 / X3 -- RESOLVED, and it is neither reading in the paper
--------------------------------------------------------------
The paper says the fraction "that best predicted each embedding feature" was selected,
which is incoherent in the encoding direction (the targets are voxels). The plan defaulted
to one fraction per voxel. **Both are wrong.** The release calls
`FracRidgeRegressorCV(jit=True, fit_intercept=True)`, which wraps `GridSearchCV` with the
default `scoring=None`. That calls `FracRidgeRegressor.score` -> `r2_score` with
`multioutput='uniform_average'`, collapsing all ~327k voxels into ONE scalar R². The single
fraction maximising it is applied to EVERY voxel. `best_frac_` is a scalar (verified
directly against the installed fracridge 3.0).

We reimplement that selection rule exactly, but vectorised over fractions and chunked over
voxels: `GridSearchCV` refits the model 20 fracs x 5 folds = 100 times, each time
recomputing the SVD and the full (768 x 327k) coefficient array, which is intractable here.
`fracridge.fracridge()` computes all 20 fractions in one call, so we evaluate a fold once
per voxel-chunk and accumulate the R² sum across chunks before taking the global argmax.
This is mathematically identical to the released selection (same uniform-average R², same
folds) -- see `tests/test_ridge.py::test_shared_frac_matches_gridsearchcv`.

Frac grid: `np.linspace(1/20, 1 + 1/20, 20)` = 0.05 .. 1.05, step 0.05263.
NOT the paper's stated "0.05 to 1.00 in steps of 0.05" -- the largest fraction exceeds OLS.
SOURCE: nsd_llm_encoding_model.py:25-26.

CV folds: `cv=None` -> sklearn's 5-fold `KFold(shuffle=False)` -> CONTIGUOUS blocks of
NSD-id-sorted rows. Not shuffled, not seeded. [MI-18 -- RESOLVED]

Standardization: none. `fit_intercept=True` mean-centres X and y (and un-centres via the
intercept) but does NOT scale to unit variance. Embeddings are not L2-normalised.
[MI-19/20 -- RESOLVED]
"""

from __future__ import annotations

import numpy as np
from fracridge import fracridge
from sklearn.model_selection import KFold

from . import config as C
from . import nsd_data

FRACS = np.linspace(C.RIDGE_FRAC_MIN, C.RIDGE_FRAC_MAX, C.RIDGE_N_FRACS)


def _r2_sum(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Sum over targets of R², for each frac. y_pred: (n, n_fracs, n_targets)."""
    ss_res = ((y_true[:, None, :] - y_pred) ** 2).sum(axis=0)          # (n_fracs, n_targets)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(axis=0)              # (n_targets,)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    return np.nansum(r2, axis=1)


def select_frac(X: np.ndarray, Y: np.ndarray, chunk: int = 20000) -> tuple[float, np.ndarray]:
    """The released selection rule: ONE shared fraction maximising uniform-average R².

    Returns (best_frac, mean_r2_per_frac).
    """
    kf = KFold(n_splits=C.RIDGE_CV_FOLDS)  # shuffle=False -> contiguous folds
    total = np.zeros(len(FRACS))
    n_targets = Y.shape[1]

    for tr, va in kf.split(X):
        Xtr, Xva, Ytr, Yva = X[tr], X[va], Y[tr], Y[va]
        # fit_intercept=True => centre, fit on centred data, add the mean back
        xm, ym = Xtr.mean(0), Ytr.mean(0)
        for a in range(0, n_targets, chunk):
            b = min(a + chunk, n_targets)
            coef, _ = fracridge(Xtr - xm, Ytr[:, a:b] - ym[a:b], fracs=FRACS)
            # coef: (n_features, n_fracs, n_chunk)
            pred = np.einsum("nf,fkt->nkt", Xva - xm, coef) + ym[a:b]
            total += _r2_sum(Yva[:, a:b], pred)

    mean_r2 = total / (n_targets * C.RIDGE_CV_FOLDS)
    return float(FRACS[int(np.argmax(mean_r2))]), mean_r2


def fit_predict(
    X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, frac: float, chunk: int = 20000
) -> np.ndarray:
    """Refit at the chosen fraction on the full training set and predict the test set."""
    xm, ym = X_train.mean(0), Y_train.mean(0)
    out = np.zeros((X_test.shape[0], Y_train.shape[1]), dtype=np.float32)
    for a in range(0, Y_train.shape[1], chunk):
        b = min(a + chunk, Y_train.shape[1])
        coef, _ = fracridge(X_train - xm, Y_train[:, a:b] - ym[a:b], fracs=np.array([frac]))
        out[:, a:b] = np.einsum("nf,fkt->nkt", X_test - xm, coef)[:, 0, :] + ym[a:b]
    return out


def pairwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r between two (n_samples, n_targets) matrices. -> (n_targets,)"""
    Az = (A - A.mean(0)) / A.std(0)
    Bz = (B - B.mean(0)) / B.std(0)
    return (Az * Bz).mean(0)


def train_test_split_515(subj: str) -> tuple[np.ndarray, np.ndarray]:
    """Positional indices into the subject's 3x-seen pool: (train, test=the 515)."""
    _, keep = nsd_data.get_conditions_3rep(subj)
    c515 = nsd_data.get_conditions_515()
    is_test = np.isin(keep, c515)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def inter_participant_agreement(betas_515: dict[str, np.ndarray]) -> np.ndarray:
    """`N-IPA` (the Fig. 1d y-axis): per voxel, correlate each subject's 515 responses with
    the MEAN of the other seven, then average the 8 maps.

    SOURCE: make_noise_ceiling.py:70-77. NaN vertices are set to 0 (not dropped) there; we
    do the same so the map is comparable.
    """
    subs = list(betas_515)
    per = []
    for s in subs:
        others = np.mean([betas_515[o] for o in subs if o != s], axis=0)
        per.append(pairwise_corr(betas_515[s], others))
    return np.nanmean(per, axis=0)
