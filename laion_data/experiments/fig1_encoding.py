"""Figures 1c / 1d — the LLM encoding model (M10).

Fit fractional ridge from the 768-d MPNet caption embedding to whole-brain fsaverage betas,
holding out the 515 shared images; score with per-vertex Pearson r on the test set.

Fig. 1c  group map of encoding accuracy            (paper colorbar -0.73 .. 0.73)
Fig. 1d  per-vertex encoding r vs inter-participant agreement, coloured by ROI group
         (paper axes 0 .. 0.8)

The paper's own stated sanity check (claim C2) is that encoding accuracy "approaches
inter-participant agreement in all ROIs" — that is the criterion we gate on, since it is the
only quantitative statement attached to this figure.

Key resolved detail (MI-17/X3): ONE shared ridge fraction for the whole model, selected by
uniform-average R² across all vertices — not one per vertex. See src/revision/ridge.py.

Memory: the betas are kept as an fp16 memmap and only 4096-vertex column blocks are ever
materialised. Slicing them into float32 up front costs ~38 GB per subject and gets
OOM-killed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.revision import config as C
from src.revision import embeddings as E
from src.revision import nsd_data, ridge, roi

BETAS_DIR = C.DERIV / "betas"


def load_betas_memmap(subj: str) -> np.ndarray:
    """(n_images, 327684) fp16, memory-mapped — NOT converted to float32."""
    return np.load(BETAS_DIR / f"{subj}_betas_z_avg_fsaverage.npy", mmap_mode="r")




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=C.SUBJECTS)
    args = ap.parse_args()

    emb = E.load("captions")
    mask = roi.streams_mask()
    c515 = nsd_data.get_conditions_515()

    enc_r: dict[str, np.ndarray] = {}
    betas_515: dict[str, np.ndarray] = {}
    fracs: dict[str, float] = {}

    for subj in args.subjects:
        betas = load_betas_memmap(subj)
        _, keep = nsd_data.get_conditions_3rep(subj)
        tr, te = ridge.train_test_split_515(subj)
        X = emb[keep]

        # leakage check: none of the 515 test images may appear in training
        assert not np.intersect1d(keep[tr], c515).size, "515 leaked into the training set"

        frac, r2_curve = ridge.select_frac(X, betas, tr)
        pred = ridge.fit_predict(X, betas, tr, X[te], frac)          # (515, n_vertices)
        true = np.asarray(betas[te], dtype=np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            r = ridge.pairwise_corr(true, pred)

        enc_r[subj] = r.astype(np.float32)
        fracs[subj] = frac
        betas_515[subj] = np.nan_to_num(true, nan=0.0)

        print(f"[Fig1] {subj}: frac={frac:.4f}  mean encoding r={np.nanmean(r):.4f}  "
              f"max={np.nanmax(r):.4f}  "
              f"(R² varies only {r2_curve.max() - r2_curve.min():.2e} across the 20 fracs)",
              flush=True)
        del betas, true, pred

    ipa = ridge.inter_participant_agreement(betas_515)
    enc_mean = np.nanmean([enc_r[s] for s in args.subjects], axis=0)

    # ---- claim C2: encoding r approaches inter-participant agreement in all ROIs ----
    rows = []
    print("\n=== Fig. 1d — encoding accuracy vs inter-participant agreement, by ROI ===")
    print(f"{'ROI':14s}{'encoding r':>12s}{'IPA':>10s}{'ratio':>9s}")
    for roi_id, roi_name in C.STREAMS_LABELS.items():
        if roi_id == 0:
            continue
        sel = (mask == roi_id) & np.isfinite(enc_mean) & np.isfinite(ipa)
        e, i = float(enc_mean[sel].mean()), float(ipa[sel].mean())
        rows.append({"roi": roi_name, "encoding_r": e, "ipa": i, "ratio": e / i if i else None})
        print(f"{roi_name:14s}{e:12.4f}{i:10.4f}{(e / i if i else float('nan')):9.2f}")

    out = C.REPORTS / "figures" / "fig1_encoding.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"fracs": fracs, "by_roi": rows}, f, indent=1)
    np.save(C.DERIV / "encoding_r_group.npy", enc_mean)
    np.save(C.DERIV / "ipa_group.npy", ipa)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
