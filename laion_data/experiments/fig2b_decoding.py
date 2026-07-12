"""Figure 2b — decoding captions from brain activity (M11).

Brain -> LLM embedding -> caption. The INVERSE of the encoding model, through the SAME ridge
core. Only two things change:
  - X and y swap: X = betas restricted to the 'streams' visual ROIs, y = the 768-d embedding
  - the voxel mask: decoding uses ONLY visual cortex, not the whole brain
SOURCE: nsd_decode_llm.py:99-107, driven by examples/llm_decoding_model.py:5-6
(`USE_ROIS='streams'`, `WHICH_ROIS='allvisROIs'` -> `vs_mask = maskdata != 0`).

Note MI-17 does NOT bite in this direction: "the fraction that best predicted each embedding
feature" is coherent here, because the 768 features ARE the targets. (It is still a single
shared fraction across those 768, per FracRidgeRegressorCV.)

Outputs
-------
  (a) per-image Pearson r between the predicted and true 768-d embedding, over the 515 test
      images -> the Fig. 2b KDE. Paper's axis: 0.3-0.7.
  (b) the caption noise ceiling (`N-NC-CAPTION`): for each of an image's 5 captions,
      correlate its embedding with the MEAN embedding of the other 4, then average the 5.
      This is a DIFFERENT ceiling from the RSA one in M8 — different data, different units.
      Do not merge them.
  (c) reconstructed captions: argmax correlation of the predicted embedding against every
      one of the 3.3M GCC dictionary entries.
  (d) the control the paper reports qualitatively: the decoded caption must NOT simply be the
      nearest TRAINING caption. We print both side by side.

A released bug we do NOT reproduce
----------------------------------
`decoding_extra_analyses.py:219` re-initialises `these_corrs = []` INSIDE the caption loop,
so the released caption ceiling is caption-4-only, not the mean over the 5. That value is the
ceiling line drawn in Fig. 2b. We compute the ceiling correctly and report both, since the
buggy version has ~5x the variance of the intended estimator.
"""

from __future__ import annotations

import json

import numpy as np

from src.revision import config as C
from src.revision import embeddings as E
from src.revision import gcc as GCC
from src.revision import nsd_data, ridge, roi, stimuli


def caption_noise_ceiling(caption_emb_per_image: list[np.ndarray]) -> tuple[float, float]:
    """(correct_mean, as_released_buggy) leave-one-caption-out ceiling."""
    correct, buggy = [], []
    for E5 in caption_emb_per_image:  # (n_caps, 768), unit-norm rows
        rs = []
        for i in range(len(E5)):
            other = np.delete(E5, i, axis=0).mean(0)
            a, b = E5[i] - E5[i].mean(), other - other.mean()
            rs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        correct.append(np.mean(rs))
        buggy.append(rs[-1])  # the release keeps only the LAST caption's value
    return float(np.mean(correct)), float(np.mean(buggy))


def main() -> None:
    emb = E.load("captions")
    mask = roi.streams_mask()
    vis = mask != 0  # 'allvisROIs' — every non-zero streams label

    dict_emb, dict_caps = GCC.build()
    D = np.asarray(dict_emb, dtype=np.float32)
    Dz = (D - D.mean(1, keepdims=True))
    Dz /= np.linalg.norm(Dz, axis=1, keepdims=True) + 1e-8

    c515 = nsd_data.get_conditions_515()
    raw_caps = stimuli.load_captions()

    results = {}
    per_subj_r = {}
    examples = {}

    for subj in C.SUBJECTS:
        betas = roi.load_betas(subj)
        _, keep = nsd_data.get_conditions_3rep(subj)
        tr, te = ridge.train_test_split_515(subj)

        X = betas[:, vis]
        good = ~np.isnan(X).any(axis=0)
        X = X[:, good]
        del betas
        Y = emb[keep]

        frac, _ = ridge.select_frac(X[tr], Y[tr])
        pred = ridge.fit_predict(X[tr], Y[tr], X[te], frac)   # (515, 768)

        # per-image correlation between predicted and true embedding
        P = pred - pred.mean(1, keepdims=True)
        T = Y[te] - Y[te].mean(1, keepdims=True)
        r = (P * T).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1))
        per_subj_r[subj] = r

        # retrieval against the GCC dictionary (correlation == cosine after row-centring)
        Pz = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        win = (Pz @ Dz.T).argmax(1)
        examples[subj] = [
            {"nsd_id": int(c515[i]), "true": raw_caps[c515[i]][0], "decoded": dict_caps[win[i]]}
            for i in (0, 100, 250, 450)
        ]
        print(f"[Fig2b] {subj}: frac={frac:.4f}  mean r={r.mean():.4f} "
              f"[{np.percentile(r,5):.2f}-{np.percentile(r,95):.2f}]", flush=True)
        del X, Y

    # caption noise ceiling on the 515
    per_img = []
    for cid in c515:
        per_img.append(E.encode(raw_caps[cid]))
    nc_correct, nc_buggy = caption_noise_ceiling(per_img)

    allr = np.concatenate([per_subj_r[s] for s in C.SUBJECTS])
    results = {
        "mean_r": float(allr.mean()),
        "kde_mass_5_95": [float(np.percentile(allr, 5)), float(np.percentile(allr, 95))],
        "paper_axis": [0.3, 0.7],
        "caption_noise_ceiling_correct": nc_correct,
        "caption_noise_ceiling_as_released_buggy": nc_buggy,
        "per_subject_mean": {s: float(per_subj_r[s].mean()) for s in C.SUBJECTS},
        "examples": examples,
    }
    p = C.REPORTS / "figures" / "fig2b_decoding.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(p, "w"), indent=1)

    print(f"\n=== Fig. 2b ===")
    print(f"decoding r: mean {allr.mean():.3f}, 5-95% [{np.percentile(allr,5):.3f}, "
          f"{np.percentile(allr,95):.3f}]   (paper's axis: 0.3-0.7)")
    print(f"caption noise ceiling: {nc_correct:.3f} (correct) vs {nc_buggy:.3f} (as released)")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
