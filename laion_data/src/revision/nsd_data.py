"""M1 -- NSD brain-data ingest and beta normalization (`D-NSD`, `D-515`, `N-BETAZ`).

Purpose
-------
Produce the single brain-side artifact every downstream analysis consumes: per-subject
betas that are (1) restricted to stimuli the subject saw 3x, (2) z-scored within each
scanning session, (3) averaged over the 3 repetitions.

Inputs
------
NSD `betas_fithrf_GLMdenoise_RR` in `fsaverage` space (lh/rh .mgh, one pair per session),
and NSD `behav/responses.tsv` (trial -> 73k-image-id mapping).

Outputs
-------
- `data/derivatives/betas/{subj}_betas_z_avg_fsaverage.npy`  (n_images_3x, 327684) fp16
- `data/derivatives/betas/{subj}_conditions.npy`             (n_images_3x,) int  73k ids
- `data/derivatives/conditions_515.npy`                      (515,) int

Transform order (must be exact -- this is the whole point of the module)
-----------------------------------------------------------------------
1. z-score each session's betas across that session's single trials, per vertex
2. concatenate sessions
3. average the 3 repetitions of each stimulus

SOURCE for every step: `code/visuo_llm/src/nsd_visuo_semantics/utils/nsd_get_data_light.py`
  - `get_betas`  (lines 154-202): `zscore(all_verts, axis=-1)` per session, per vertex
  - `load_or_compute_betas_average` (126-151): concatenate, then `average_over_conditions`
  - `average_over_conditions` (109-121): `np.nanmean` over the 3 repeats
  - `get_conditions_515` (424-456): intersection over all 8 subjects of the shared-1000
    images that the subject saw 3x

Note on the /300 scale factor (MI-05)
-------------------------------------
The release divides by 300 ONLY in the `func1pt8mm` branch; the `fsaverage` branch has an
explicit comment "no need to divide by 300 in this case". Because the division is applied
*inside* a z-score over the trial axis, a constant scale cancels exactly -- so it is a
no-op for every quantity in this paper. We keep the branch faithful anyway.

Limitations
-----------
- fsaverage only. The volumetric (`func1pt8mm`) path needed for the searchlight figures
  (1b, 4c, 4d) is a separate, larger download and is not built here.
- Vertices that are NaN for a subject are NOT dropped here; downstream consumers drop
  them, matching the release code (which filters per-analysis, not once at ingest).
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

BETAS_DIR = C.DERIV / "betas"
RAW_DIR = C.NSD_DIR / "nsddata_betas" / "ppdata"


# ======================================================================================
# Conditions (which 73k-image did each trial show?)
# ======================================================================================
def get_conditions(subj: str) -> np.ndarray:
    """Return the 73k image id shown on each trial, in trial order, for one subject.

    NSD's `behav/responses.tsv` has one row per trial with a `73KID` column (1-indexed).
    We return it 0-indexed to match numpy conventions everywhere downstream.
    """
    tsv = C.NSD_DIR / "nsddata" / "ppdata" / subj / "behav" / "responses.tsv"
    df = pd.read_csv(tsv, sep="\t")
    n_sess = C.N_SESSIONS[subj]
    df = df[df["SESSION"] <= n_sess]
    return df["73KID"].to_numpy() - 1  # -> 0-indexed


def get_conditions_3rep(subj: str) -> tuple[np.ndarray, np.ndarray]:
    """(trial_conditions, unique_conditions_seen_exactly_3x) for one subject.

    SOURCE: nsd_get_data_light.py:266 -- a condition qualifies iff it appears EXACTLY 3
    times. Subjects who did not complete all 40 sessions therefore contribute fewer
    3x-seen images (this is what produces the 10,000 / 6,234 / 5,445 counts).
    """
    conditions = get_conditions(subj)
    ids, counts = np.unique(conditions, return_counts=True)
    keep = ids[counts == C.N_REPEATS]
    return conditions, np.sort(keep)


def get_conditions_1000() -> np.ndarray:
    """The shared-1000 73k image ids (0-indexed).

    These are NOT the 1,000 lowest ids -- they are a specific scattered set stored in
    `nsd_expdesign.mat` under `sharedix` (1-indexed, spanning 2951..72949).
    SOURCE: nsd_get_data_light.py `get_conditions_1000`; NSD `nsd_expdesign.mat`.
    """
    import scipy.io as sio

    mat = sio.loadmat(C.NSD_DIR / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat")
    return np.sort(mat["sharedix"].squeeze().astype(int) - 1)  # -> 0-indexed


def get_conditions_515() -> np.ndarray:
    """The 515 images that ALL 8 subjects saw 3 times (the held-out test set).

    NSD showed a shared set of 1,000 images to every subject. Three subjects did not
    complete all sessions, so not all 1,000 were seen 3x by everyone. The 515 is the
    intersection across subjects of (shared-1000 AND seen-3x).

    SOURCE: nsd_get_data_light.py:434-456.
    """
    stim_1000 = set(int(x) for x in get_conditions_1000())
    shared: set[int] | None = None
    for subj in C.SUBJECTS:
        _, three_rep = get_conditions_3rep(subj)
        s = set(int(x) for x in three_rep) & stim_1000
        shared = s if shared is None else (shared & s)
    assert shared is not None
    return np.array(sorted(shared), dtype=int)


# ======================================================================================
# Download
# ======================================================================================
def session_urls(subj: str, sess: int) -> list[tuple[str, Path]]:
    rel = f"nsddata_betas/ppdata/{subj}/{C.TARGETSPACE_ROI}/{C.BETA_VERSION}"
    out = []
    for hemi in ("lh", "rh"):
        fn = f"{hemi}.betas_session{sess:02d}.mgh"
        out.append((f"{C.NSD_S3}/{rel}/{fn}", C.NSD_DIR / rel / fn))
    return out


def _fetch(url_dest: tuple[str, Path]) -> Path:
    url, dest = url_dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(
        ["curl", "-sS", "--fail", "--retry", "5", "--retry-delay", "3", "-o", str(tmp), url],
        check=True,
    )
    tmp.rename(dest)
    return dest


def download_subject_sessions(subj: str, workers: int = 12) -> None:
    """Download every session's lh/rh betas for one subject, in parallel."""
    jobs = [u for s in range(1, C.N_SESSIONS[subj] + 1) for u in session_urls(subj, s)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_fetch, jobs))


# ======================================================================================
# N-BETAZ
# ======================================================================================
def _load_session_zscored(subj: str, sess: int) -> np.ndarray:
    """Load one session, z-score across its trials per vertex. -> (n_vertices, n_trials)."""
    import nibabel as nb

    rel = f"nsddata_betas/ppdata/{subj}/{C.TARGETSPACE_ROI}/{C.BETA_VERSION}"
    arrs = []
    for hemi in ("lh", "rh"):
        p = C.NSD_DIR / rel / f"{hemi}.betas_session{sess:02d}.mgh"
        arrs.append(nb.load(str(p)).get_fdata().squeeze())
    verts = np.vstack(arrs)  # (327684, n_trials)

    # z-score across trials (axis=-1), per vertex, WITHIN this session.
    # SOURCE: nsd_get_data_light.py:186 `zscore(all_verts, axis=cond_axis)` with cond_axis=-1
    mu = verts.mean(axis=-1, keepdims=True)
    sd = verts.std(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (verts - mu) / sd  # sd==0 -> NaN/inf, exactly as scipy.stats.zscore does
    return z.astype(np.float32)


def build_betas(subj: str, keep_raw: bool = False) -> Path:
    """Full M1 pipeline for one subject. Returns the path to the saved artifact.

    Memory: we accumulate a running sum per (vertex, condition) rather than concatenating
    all ~30,000 trials, which would need ~40 GB. Since every kept condition has exactly 3
    repeats, a sum/3 is identical to the release's `np.nanmean` over the 3 repeats.
    """
    BETAS_DIR.mkdir(parents=True, exist_ok=True)
    out = BETAS_DIR / f"{subj}_betas_z_avg_fsaverage.npy"
    cond_out = BETAS_DIR / f"{subj}_conditions.npy"
    if out.exists() and cond_out.exists():
        return out

    trial_conditions, keep = get_conditions_3rep(subj)
    n_keep = len(keep)
    # map 73k id -> column index in the output
    col_of = -np.ones(trial_conditions.max() + 1, dtype=np.int64)
    col_of[keep] = np.arange(n_keep)

    acc = np.zeros((C.N_VERTICES_FSAVERAGE, n_keep), dtype=np.float32)
    cnt = np.zeros(n_keep, dtype=np.int32)

    n_sess = C.N_SESSIONS[subj]
    trials_per_session = len(trial_conditions) // n_sess
    for sess in range(1, n_sess + 1):
        z = _load_session_zscored(subj, sess)
        lo = (sess - 1) * trials_per_session
        sess_conditions = trial_conditions[lo : lo + z.shape[1]]
        cols = col_of[sess_conditions]
        valid = cols >= 0
        np.add.at(acc.T, cols[valid], z[:, valid].T)
        np.add.at(cnt, cols[valid], 1)
        del z

    assert (cnt == C.N_REPEATS).all(), (
        f"{subj}: expected every kept condition to have exactly {C.N_REPEATS} repeats, "
        f"got counts {np.unique(cnt)}"
    )
    acc /= C.N_REPEATS

    np.save(out, acc.T.astype(np.float16))  # (n_images_3x, n_vertices)
    np.save(cond_out, keep)
    if not keep_raw:
        for sess in range(1, n_sess + 1):
            for _, p in session_urls(subj, sess):
                p.unlink(missing_ok=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="M1: NSD ingest + N-BETAZ (fsaverage)")
    ap.add_argument("--subjects", nargs="*", default=C.SUBJECTS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()

    c515 = get_conditions_515()
    (C.DERIV).mkdir(parents=True, exist_ok=True)
    np.save(C.DERIV / "conditions_515.npy", c515)
    print(f"[M1] |D-515| = {len(c515)}", flush=True)

    for subj in args.subjects:
        print(f"[M1] {subj}: downloading {C.N_SESSIONS[subj]} sessions...", flush=True)
        download_subject_sessions(subj, workers=args.workers)
        print(f"[M1] {subj}: building N-BETAZ...", flush=True)
        p = build_betas(subj, keep_raw=args.keep_raw)
        arr = np.load(p, mmap_mode="r")
        print(f"[M1] {subj}: {arr.shape} -> {p}", flush=True)


if __name__ == "__main__":
    main()
