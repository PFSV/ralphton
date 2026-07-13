"""A vectorised, GPU fractional-ridge solver — numerically identical to `fracridge`.

Why this exists
---------------
`fracridge.fracridge()` is correct but its main loop is `for ii in range(y.shape[-1])`,
i.e. one Python iteration PER TARGET. The encoding model has ~327,684 targets (fsaverage
vertices), five CV folds and eight subjects, which is tens of millions of iterations. In
practice the CPU version did not finish a single subject's fraction-selection in 8 hours.

Nothing about the algorithm requires that loop — it is fully vectorisable. This module
reimplements it exactly, batched over targets, on the GPU.

The algorithm (verbatim from `fracridge/fracridge.py`)
-----------------------------------------------------
1. SVD:  X = U diag(s) Vᵀ,  and  ols_coef = (Uᵀ y) / s   (coefficients in the rotated space)
2. Build a log-spaced alpha grid, prepended with 0:
       val1 = BIG_BIAS   * s[0]**2      (BIG_BIAS   = 1e4)
       val2 = SMALL_BIAS * s[-1]**2     (SMALL_BIAS = 1e-2)
       alphagrid = [0] + 10 ** arange(floor(log10(val2)), ceil(log10(val1)), 0.2)
3. For each alpha, the rotated coefficients are scaled by s²/(s²+alpha). The RELATIVE length
   of the solution vs OLS is
       newlen(alpha) = ||scaled coef|| / ||ols coef||       (== 1 at alpha = 0)
4. For each target, interpolate — in log(1+alpha) space — to find the alpha whose `newlen`
   equals the requested fraction.
5. Rebuild the coefficients at those alphas and un-rotate with V.

Equivalence is TESTED, not asserted: `tests/test_modules.py::test_gpu_fracridge_matches_library`
checks coefficients and alphas against the reference implementation to 1e-4.
"""

from __future__ import annotations

import numpy as np
import torch

# Exactly the constants the library uses (fracridge/fracridge.py).
BIG_BIAS = 1e4
SMALL_BIAS = 1e-2
BIAS_STEP = 0.2
TOL = 1e-10


def _alpha_grid(s: torch.Tensor) -> torch.Tensor:
    val1 = BIG_BIAS * float(s[0]) ** 2
    val2 = SMALL_BIAS * float(s[-1]) ** 2
    if val2 == 0:
        val2 = SMALL_BIAS
    lo = np.floor(np.log10(val2))
    hi = np.ceil(np.log10(val1))
    grid = 10.0 ** np.arange(lo, hi, BIAS_STEP)
    return torch.cat([torch.zeros(1, dtype=s.dtype, device=s.device),
                      torch.as_tensor(grid, dtype=s.dtype, device=s.device)])


def fracridge_torch(
    X: torch.Tensor, Y: torch.Tensor, fracs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """X (n, p), Y (n, b), fracs (f,) -> coef (p, f, b), alphas (f, b).

    X and Y must already be centred by the caller (the library's `fit_intercept=True` path
    centres them and adds the intercept back afterwards).
    """
    svd = torch.linalg.svd(X, full_matrices=False)
    return fracridge_from_svd(svd.U, svd.S, svd.Vh, Y, fracs)


def fracridge_from_svd(
    U: torch.Tensor, s: torch.Tensor, Vh: torch.Tensor, Y: torch.Tensor, fracs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """As `fracridge_torch`, but with the SVD of X supplied.

    The design matrix is the same for every target chunk within a CV fold, so the SVD is
    hoisted out of the chunk loop. (The library recomputes it on every call.)
    """
    ols = (U.T @ Y) / s[:, None]                          # (k, b)
    ols = torch.where((s < TOL)[:, None], torch.zeros_like(ols), ols)

    alphagrid = _alpha_grid(s)                            # (A,)
    s2 = s ** 2                                           # (k,)
    sclg_sq = (s2 / (s2 + alphagrid[:, None])) ** 2       # (A, k)

    newlen = torch.sqrt(sclg_sq @ ols ** 2)               # (A, b)
    newlen = newlen / newlen[0:1]                         # alphagrid[0] == 0 -> OLS length

    # Per-target interpolation. `newlen` decreases with alpha, so reverse it to get an
    # increasing x-axis; the y-axis (log(1+alpha)) is shared by every target.
    xp = newlen.flip(0).T.contiguous()                    # (b, A), increasing
    fp = torch.log1p(alphagrid).flip(0)                   # (A,)

    b = Y.shape[1]
    vals = fracs[None, :].expand(b, -1).contiguous()      # (b, f)
    A = xp.shape[1]
    idx = torch.searchsorted(xp, vals).clamp(1, A - 1)    # (b, f)

    x0 = xp.gather(1, idx - 1)
    x1 = xp.gather(1, idx)
    y0 = fp[idx - 1]
    y1 = fp[idx]
    # Clamping t to [0, 1] reproduces np.interp's end-clamping for out-of-range fractions.
    t = ((vals - x0) / (x1 - x0).clamp_min(1e-30)).clamp(0.0, 1.0)
    alphas = torch.expm1(y0 + t * (y1 - y0))              # (b, f)

    sc = s2[None, None, :] / (s2[None, None, :] + alphas[:, :, None])   # (b, f, k)
    coef_rot = sc * ols.T[:, None, :]                                    # (b, f, k)
    coef = coef_rot @ Vh                                                 # (b, f, p)
    return coef.permute(2, 1, 0).contiguous(), alphas.T.contiguous()


def predict(X: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """X (n, p), coef (p, f, b) -> (n, f, b)."""
    return torch.einsum("np,pfb->nfb", X, coef)
