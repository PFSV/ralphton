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

Two engineering constraints forced the shape of this module
-----------------------------------------------------------
1. **The library cannot do this at scale.** `fracridge.fracridge()`'s main loop is
   `for ii in range(y.shape[-1])` -- one Python iteration per target. With 327,684 vertices
   x 5 folds x 8 subjects it did not finish a single subject in 8 hours. The algorithm is
   fully vectorisable, so `fracridge_gpu` reimplements it batched on the GPU, matching the
   library's coefficients to 1e-14 (tests/test_modules.py::test_gpu_fracridge_matches_library).
   The SVD is hoisted out of the chunk loop, since X is constant within a fold.
2. **The betas do not fit in RAM three times over.** A naive
   `Y = betas.astype(f32)[:, good]` then `Y[train]` makes three ~13 GB copies and gets
   OOM-killed. So the betas stay as an fp16 memmap and this module materialises only one
   (n_images, CHUNK) float32 block at a time. Callers pass ROW indices (train/test) and
   COLUMN indices (the non-NaN vertices) rather than pre-sliced arrays.

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
import torch
from sklearn.model_selection import KFold

from . import config as C
from . import nsd_data
from .fracridge_gpu import fracridge_from_svd

FRACS = np.linspace(C.RIDGE_FRAC_MIN, C.RIDGE_FRAC_MAX, C.RIDGE_N_FRACS)
CHUNK = 4096  # targets per GPU batch


def _pick_device() -> str:
    """Use the GPU with the most free memory.

    This box is shared: a plain `cuda` (= cuda:0) picked a card another user was holding
    36 GB on, and the ridge OOM'd. Selecting by free memory makes the pipeline independent
    of which card happens to be busy.
    """
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f"cuda:{int(np.argmax(free))}"


DEVICE = _pick_device()

# float64, NOT float32. The design matrix (mean-centred MPNet caption embeddings) has
# condition number ~3.5e19 -- its smallest singular value is 6e-19. `ols = (Uᵀy)/s` therefore
# divides by near-zero, and in float32 the whole solve returns NaN for 100% of targets. The
# reference library runs in float64 for exactly this reason. Verified: in float64 our
# predictions match `fracridge` to 9e-05 in encoding r.
DTYPE = torch.float64


def _dev(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), device=DEVICE, dtype=DTYPE)


def _block(Y, a: int, b: int) -> np.ndarray:
    """Materialise ONE contiguous float64 column-block of Y (which may be an fp16 memmap).

    Contiguous, not fancy-indexed: gathering scattered columns from a row-major memmap
    re-reads every row for each block and made this I/O-bound. NaN vertices are kept here
    and handled at the R² step instead -- they stay confined to their own target column,
    because fracridge is independent across targets.
    """
    return np.asarray(Y[:, a:b], dtype=np.float64)


def select_frac(
    X: np.ndarray, Y, rows: np.ndarray, chunk: int = CHUNK
) -> tuple[float, np.ndarray]:
    """The released selection rule: ONE shared fraction maximising uniform-average R².

    X    : (n_images, p) design matrix (full; `rows` selects the training images)
    Y    : (n_images, n_targets) targets, possibly an fp16 memmap
    rows : training image indices

    NaN targets (vertices that are NaN for this subject) are excluded from the average, as
    the release does by dropping them up front -- here they simply contribute nothing and
    are not counted in the denominator, which is equivalent.

    Returns (best_frac, mean_r2_per_frac).
    """
    kf = KFold(n_splits=C.RIDGE_CV_FOLDS)  # shuffle=False -> contiguous folds
    fracs = _dev(FRACS)
    total = torch.zeros(len(FRACS), device=DEVICE, dtype=torch.float64)
    n_valid = 0
    n_t = Y.shape[1]

    folds = list(kf.split(rows))
    svds, Xvas = [], []
    for tr, va in folds:
        Xtr = _dev(X[rows[tr]])
        Xva = _dev(X[rows[va]])
        xm = Xtr.mean(0, keepdim=True)
        svds.append(torch.linalg.svd(Xtr - xm, full_matrices=False))
        Xvas.append(Xva - xm)

    for a in range(0, n_t, chunk):
        b = min(a + chunk, n_t)
        Yblock = _block(Y, a, b)  # (n_images, chunk)
        valid = np.isfinite(Yblock).all(axis=0)
        n_valid += int(valid.sum())

        for (tr, va), svd, Xva_c in zip(folds, svds, Xvas):
            Ytr = _dev(Yblock[rows[tr]])
            Yva = _dev(Yblock[rows[va]])
            ym = Ytr.mean(0, keepdim=True)

            coef, _ = fracridge_from_svd(svd.U, svd.S, svd.Vh, Ytr - ym, fracs)  # (p,f,t)
            pred = torch.einsum("np,pfb->nfb", Xva_c, coef) + ym[:, None, :]

            ss_res = ((Yva[:, None, :] - pred) ** 2).sum(0)           # (f, t)
            ss_tot = ((Yva - Yva.mean(0, keepdim=True)) ** 2).sum(0)  # (t,)
            r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-12)
            total += torch.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0).sum(1)
            del coef, pred

    mean_r2 = (total / (n_valid * C.RIDGE_CV_FOLDS)).cpu().numpy()
    return float(FRACS[int(np.argmax(mean_r2))]), mean_r2


def fit_predict(
    X: np.ndarray, Y, rows: np.ndarray, X_test: np.ndarray, frac: float, chunk: int = CHUNK
) -> np.ndarray:
    """Refit at the chosen fraction on `rows` and predict `X_test`. -> (n_test, n_targets)"""
    Xtr = _dev(X[rows])
    Xte = _dev(X_test)
    xm = Xtr.mean(0, keepdim=True)
    svd = torch.linalg.svd(Xtr - xm, full_matrices=False)
    Xte_c = Xte - xm
    fracs = _dev(np.array([frac]))

    n_t = Y.shape[1]
    out = np.zeros((X_test.shape[0], n_t), dtype=np.float32)
    for a in range(0, n_t, chunk):
        b = min(a + chunk, n_t)
        Ytr = _dev(_block(Y, a, b)[rows])
        ym = Ytr.mean(0, keepdim=True)
        coef, _ = fracridge_from_svd(svd.U, svd.S, svd.Vh, Ytr - ym, fracs)
        pred = torch.einsum("np,pfb->nfb", Xte_c, coef)[:, 0, :] + ym
        out[:, a:b] = pred.float().cpu().numpy()
        del coef, pred
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
