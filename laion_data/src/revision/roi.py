"""M6 + M8 + M9 -- ROI patterns, noise ceiling, group statistics.  -> Figure 3

M6 (`N-SEARCHLIGHT` / ROI masking)
    Turn the normalized fsaverage betas into per-ROI activity patterns.
    MI-13 -- RESOLVED: there is NO voxel selection and NO reliability threshold. All
    vertices in the ROI mask are used; only vertices that are NaN for that subject are
    dropped. SOURCE: roi_utils.py:30-44.

M8 (`N-NC-RSA`, `N-NCCORR`)
    The noise ceiling is a leave-one-subject-out Pearson correlation of a subject's RDM
    against the MEAN RDM of the other 7, computed on the shared 515 images, per ROI.
    MI-15 -- RESOLVED: the correction is a PLAIN DIVISION, r_model / r_ceiling(subj).
    SOURCE: nsd_roi_analyses.py:128-162.

    H3 (hidden assumption, made concrete by the code): the numerator is a mean over the
    subject's ~100 disjoint 100-image sub-RDMs of the FULL NSD set, while the denominator
    is a single correlation on the FULL 515x515 RDM. Different image sets, different RDM
    sizes, different pair counts. Corrected values can therefore exceed 1 and are not a
    clean "fraction of explainable variance". We reproduce it as written AND report the
    matched-set variant (ceiling and model correlation both on the 515) as a sensitivity
    check.

M9 (`N-STATS`)
    Two-tailed t-test across the 8 participants; BH-FDR at 0.05.
    MI-21 -- RESOLVED: raw r, no Fisher-z (`arctanh` has zero hits in the release).
    MI-22 -- RESOLVED: the ROI FDR family is the pairwise model comparisons WITHIN one
    ROI; the one-sample tests are not FDR-corrected at all.

    NOTE (a defect we reproduce but also correct): the release compares two models with
    `scipy.stats.ttest_ind` -- an INDEPENDENT-samples test -- on the same 8 subjects
    measured under both models (nsd_roi_analyses_figure.py:198). That is an unpaired test
    on paired data, and it is exactly what a reviewer would flag. We report both the
    released (unpaired) test and the correct paired test.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import config as C
from . import nsd_data, rsa

BETAS_DIR = C.DERIV / "betas"


# ======================================================================================
# M6 -- ROI masks and patterns
# ======================================================================================
def streams_mask() -> np.ndarray:
    """The 'streams' ROI label per fsaverage vertex (lh then rh). -> (327684,) int"""
    import nibabel as nb

    lab = C.NSD_DIR / "nsddata" / "freesurfer" / "fsaverage" / "label"
    parts = [
        np.asarray(nb.load(str(lab / f"{h}.streams.mgz")).dataobj).squeeze().astype(int)
        for h in ("lh", "rh")
    ]
    m = np.concatenate(parts)
    assert m.shape == (C.N_VERTICES_FSAVERAGE,), m.shape
    return m


def load_betas(subj: str) -> np.ndarray:
    """(n_images_3x, n_vertices) float32, in the subject's sorted 3x-seen condition order."""
    return np.load(BETAS_DIR / f"{subj}_betas_z_avg_fsaverage.npy").astype(np.float32)


def roi_patterns(betas: np.ndarray, mask: np.ndarray, roi: str) -> np.ndarray:
    """Restrict to one ROI and drop NaN vertices. -> (n_images, n_roi_vertices)"""
    roi_id = next(k for k, v in C.STREAMS_LABELS.items() if v == roi)
    sel = mask == roi_id
    X = betas[:, sel]
    good = ~np.isnan(X).any(axis=0)
    return X[:, good]


# ======================================================================================
# M7 (brain side) -- per-subject, per-ROI model correlations
# ======================================================================================
def brain_rdms(X: np.ndarray, splits: np.ndarray) -> list[np.ndarray]:
    """One brain RDM per split. Computed ONCE and reused across every model -- this is the
    single biggest engineering win in the whole pipeline ([DG 8])."""
    return [rsa.rdm(X[idx]) for idx in splits]


def model_correlations(
    subj: str, model_emb: np.ndarray, brain: dict[str, list[np.ndarray]], splits: np.ndarray
) -> dict[str, float]:
    """Mean-over-splits RDM correlation between one model and each ROI, for one subject."""
    _, keep = nsd_data.get_conditions_3rep(subj)
    M = model_emb[keep]
    out = {}
    for roi, rdms in brain.items():
        rs = [rsa.corr_rdms(rdms[i], rsa.rdm(M[idx])) for i, idx in enumerate(splits)]
        out[roi] = float(np.mean(rs))
    return out


# ======================================================================================
# M8 -- noise ceiling on the shared 515
# ======================================================================================
def rdms_515(mask: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Per subject, per ROI: the full 515x515 RDM (condensed, 132,355 pairs)."""
    c515 = nsd_data.get_conditions_515()
    out: dict[str, dict[str, np.ndarray]] = {}
    for subj in C.SUBJECTS:
        betas = load_betas(subj)
        _, keep = nsd_data.get_conditions_3rep(subj)
        pos = np.searchsorted(keep, c515)  # positions of the 515 in this subject's pool
        out[subj] = {}
        for roi in C.STREAMS_MAIN_ROIS:
            X = roi_patterns(betas, mask, roi)[pos]
            out[subj][roi] = rsa.rdm(X)
        del betas
    return out


def noise_ceilings(r515: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """LOSO: correlate each subject's 515-RDM with the MEAN 515-RDM of the other 7."""
    nc: dict[str, dict[str, float]] = {roi: {} for roi in C.STREAMS_MAIN_ROIS}
    for roi in C.STREAMS_MAIN_ROIS:
        for subj in C.SUBJECTS:
            others = [r515[o][roi] for o in C.SUBJECTS if o != subj]
            nc[roi][subj] = rsa.corr_rdms(r515[subj][roi], np.mean(others, axis=0))
    return nc


# ======================================================================================
# M9 -- group statistics
# ======================================================================================
def group_stats(corrs: dict[str, dict[str, float]], models: list[str]) -> dict:
    """corrs[subj][model] -> r for ONE roi. Returns one-sample and pairwise tests.

    Reproduces the release's family structure: FDR is applied to the pairwise comparisons
    within this ROI; the one-sample tests are left uncorrected.
    """
    vals = {m: np.array([corrs[s][m] for s in C.SUBJECTS]) for m in models}

    one_sample = {m: stats.ttest_1samp(vals[m], 0, alternative="two-sided").pvalue for m in models}

    pairs, p_unpaired, p_paired = [], [], []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pairs.append((a, b))
            # as released: independent-samples test on paired data
            p_unpaired.append(stats.ttest_ind(vals[a], vals[b], alternative="two-sided").pvalue)
            # the correct test for this design
            p_paired.append(stats.ttest_rel(vals[a], vals[b], alternative="two-sided").pvalue)

    def fdr(p):
        return multipletests(p, alpha=0.05, method="fdr_bh")[1] if p else np.array([])

    return {
        "mean": {m: float(vals[m].mean()) for m in models},
        "sem": {m: float(vals[m].std(ddof=1) / np.sqrt(len(C.SUBJECTS))) for m in models},
        "one_sample_p": one_sample,
        "pairs": pairs,
        "p_unpaired_fdr": fdr(p_unpaired),  # the released test
        "p_paired_fdr": fdr(p_paired),      # the correct test
    }
