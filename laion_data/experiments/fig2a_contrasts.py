"""Figure 2a — functional contrasts predicted from novel sentences (M12).

Feed hand-written sentences through the FROZEN encoding model and contrast the predicted
brain maps: People − Places and People − Food.

The 15 sentences are reproduced verbatim from the release
(`encoding_decoding_utils.py:191-216`), so this analysis is exactly reproducible *as run*,
even though the authors state "we did not have a precise method for selecting these
sentences".

Two properties of this figure that we preserve rather than silently "fix":

1. **NO FDR CORRECTION.** This is the only analysis in the paper without multiple-comparison
   control, and the paper says so explicitly. The release confirms it: the contrast maps are
   thresholded with `sig_mask='uncorrected'` (p < 0.05, two-tailed, across the 8 subjects) at
   `text_to_brain_prediction.py:80`, whereas the encoding-accuracy map uses `fdr_bh`.
   Reproducing this faithfully means NOT applying FDR here. We additionally report an
   FDR-corrected version as an extension, clearly labelled, because that is the obvious
   reviewer question (hidden assumption H4).

2. The 5 sentence embeddings within a category are AVERAGED into one 768-d vector, which is
   then pushed through the encoding model once (`text_to_brain_prediction.py:34-35`) — the
   contrast is a subtraction of the two predicted maps per subject.

Claim tested: C4 (the encoding model reproduces category-selective tuning). The claim is
topographic, not numeric — there is no published effect size to hit. It is the weakest claim
in the paper on evidentiary grounds.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.revision import config as C
from src.revision import embeddings as E
from src.revision import nsd_data, ridge, roi

# SOURCE: visuo_llm/src/nsd_visuo_semantics/encoding_decoding_analyses/
#         encoding_decoding_utils.py:191-216 — verbatim, character for character.
SENTENCES = {
    "people": [
        "Man with a beard smiling at the camera.",
        "Some children playing.",
        "Her face was beautiful.",
        "Woman and her daughter playing.",
        "Close up of a face of young boy.",
    ],
    "places": [
        "A view of a beautiful landscape.",
        "Houses along a street.",
        "City skyline with blue sky.",
        "Woodlands in the morning.",
        "A park with bushes and trees in the distance.",
    ],
    "food": [
        "A plate of food with vegetables.",
        "A hamburger with fries.",
        "A bowl of fruit.",
        "A plate of spaghetti.",
        "A bowl of soup.",
    ],
}
# SOURCE: examples/llm_encoding_model.py:35
CONTRASTS = [("people", "places"), ("food", "people")]


def main() -> None:
    emb = E.load("captions")
    mask = roi.streams_mask()

    # One averaged 768-d vector per category.
    cat_emb = {k: E.encode(v).mean(axis=0, keepdims=True) for k, v in SENTENCES.items()}

    preds: dict[str, dict[str, np.ndarray]] = {k: {} for k in SENTENCES}
    for subj in C.SUBJECTS:
        # fp16 memmap: never materialise the full float32 beta matrix (see ridge.py)
        betas = np.load(
            C.DERIV / "betas" / f"{subj}_betas_z_avg_fsaverage.npy", mmap_mode="r"
        )
        _, keep = nsd_data.get_conditions_3rep(subj)
        tr, _ = ridge.train_test_split_515(subj)
        X = emb[keep]

        frac, _ = ridge.select_frac(X, betas, tr)
        for cat, e in cat_emb.items():
            preds[cat][subj] = ridge.fit_predict(X, betas, tr, e, frac)[0]
        print(f"[Fig2a] {subj}: frac={frac:.4f}", flush=True)
        del betas

    out = {}
    for a, b in CONTRASTS:
        diff = np.stack([preds[a][s] - preds[b][s] for s in C.SUBJECTS])  # (8, n_vertices)
        t, p = stats.ttest_1samp(diff, 0, axis=0, nan_policy="omit")

        valid = np.isfinite(p)
        sig_uncorr = np.zeros_like(valid)
        sig_uncorr[valid] = p[valid] < 0.05                      # AS PUBLISHED (no FDR)
        sig_fdr = np.zeros_like(valid)
        sig_fdr[valid] = multipletests(p[valid], alpha=0.05, method="fdr_bh")[0]  # extension

        name = f"{a}_minus_{b}"
        np.save(C.DERIV / f"contrast_{name}.npy", np.nanmean(diff, axis=0))
        out[name] = {
            "n_sig_uncorrected_as_published": int(sig_uncorr.sum()),
            "n_sig_fdr_extension": int(sig_fdr.sum()),
            "n_valid_vertices": int(valid.sum()),
        }
        # Which ROIs does the positive lobe fall in?
        m = np.nanmean(diff, axis=0)
        by_roi = {}
        for rid, rname in C.STREAMS_LABELS.items():
            if rid == 0:
                continue
            sel = (mask == rid) & valid
            by_roi[rname] = float(np.nanmean(m[sel]))
        out[name]["mean_contrast_by_roi"] = by_roi
        print(f"\n[{name}] significant vertices: {sig_uncorr.sum()} uncorrected (as published) / "
              f"{sig_fdr.sum()} with FDR (extension)")
        for r, v in by_roi.items():
            print(f"    {r:14s} {v:+.4f}")

    p = C.REPORTS / "figures" / "fig2a_contrasts.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
