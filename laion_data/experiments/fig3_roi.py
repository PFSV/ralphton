"""Figure 3 — ROI-wise RSA: which text representation best matches higher visual cortex?

Pipeline: M6 (ROI patterns) -> M7 (100-image RSA) -> M8 (noise ceiling) -> M9 (stats).

Design invariant (the whole point of the figure): the brain side and the encoder are held
FIXED, and only the string fed to the encoder changes. Any difference between bars is
therefore attributable to linguistic content.

What this reproduces
--------------------
Fig. 3a  category words vs multi-hot vs word vectors
Fig. 3b  full captions vs nouns-only vs verbs-only
Fig. 3c  full captions vs averaged single-word embeddings

Claims tested: C6, C7, C8, C10.

Noise-ceiling correction (MI-15): plain division by that subject's LOSO ceiling, computed
on the shared 515. Applied to Fig. 3 ONLY -- Figs. 1b/4c/4d are on a RAW r scale and must
never be cross-compared with these numbers ("the consistency trap").

Two variants of the ceiling are reported:
  as_released  : numerator on the ~100 disjoint 100-image sub-RDMs of the FULL pool;
                 denominator on the full 515x515 RDM. Different image sets and RDM sizes.
  matched_515  : both computed on the 515. This is hidden assumption H3 made testable.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.revision import config as C
from src.revision import embeddings as E
from src.revision import nsd_data, roi, rsa

# The models on the x-axis of Fig. 3, in the paper's grouping.
MODELS = {
    "mpnet_captions": ("captions", "LLM (full captions) — the reference bar"),
    "mpnet_categories": ("categories", "LLM (category words)"),
    "mpnet_nouns": ("nouns", "LLM (nouns only)"),
    "mpnet_verbs": ("verbs", "LLM (verbs only)"),
    "mpnet_allwords": ("allwords", "LLM (single words, averaged)"),
}
# Non-LLM controls (M5). Added only if their artifacts exist.
CONTROLS = {
    "multihot": "multihot_73k_model",
    "fasttext_categories": "fasttext_categories",
    "fasttext_allwords": "fasttext_allwords",
    "glove_categories": "glove_categories",
    "glove_allwords": "glove_allwords",
}


def load_model_embeddings() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, (variant, _) in MODELS.items():
        try:
            out[name] = E.load(variant)
        except FileNotFoundError:
            print(f"  [skip] {name}: not built")
    for name, fn in CONTROLS.items():
        p = C.DERIV / "embeddings" / f"{fn}_73k.npy" if not fn.endswith("model") else C.DERIV / "embeddings" / f"{fn}.npy"
        if p.exists():
            out[name] = np.load(p)
        else:
            print(f"  [skip] {name}: not built (M5 pending)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.RSA_SPLIT_SEED)
    ap.add_argument("--out", default=str(C.REPORTS / "figures" / "fig3_roi.json"))
    args = ap.parse_args()

    mask = roi.streams_mask()
    models = load_model_embeddings()
    print(f"[Fig3] models: {list(models)}")

    # ---- M7: per-subject, per-ROI model correlations (brain RDMs cached per split) ----
    corrs: dict[str, dict[str, dict[str, float]]] = {}   # [subj][model][roi]
    r515: dict[str, dict[str, np.ndarray]] = {}
    corrs_515: dict[str, dict[str, dict[str, float]]] = {}  # matched-set variant

    c515 = nsd_data.get_conditions_515()
    for subj in C.SUBJECTS:
        betas = roi.load_betas(subj)
        _, keep = nsd_data.get_conditions_3rep(subj)
        splits = rsa.make_splits(subj, seed=args.seed)
        pos515 = np.searchsorted(keep, c515)

        brain = {}
        r515[subj] = {}
        for roi_name in C.STREAMS_MAIN_ROIS:
            X = roi.roi_patterns(betas, mask, roi_name)
            brain[roi_name] = roi.brain_rdms(X, splits)      # reused across ALL models
            r515[subj][roi_name] = rsa.rdm(X[pos515])        # for the ceiling
        del betas

        corrs[subj] = {}
        corrs_515[subj] = {}
        for m, emb in models.items():
            corrs[subj][m] = roi.model_correlations(subj, emb, brain, splits)
            M515 = emb[keep][pos515]
            mr = rsa.rdm(M515)
            corrs_515[subj][m] = {
                r: rsa.corr_rdms(r515[subj][r], mr) for r in C.STREAMS_MAIN_ROIS
            }
        print(f"[Fig3] {subj}: {len(models)} models x {len(C.STREAMS_MAIN_ROIS)} ROIs done", flush=True)

    # ---- M8: noise ceiling + correction ----
    nc = roi.noise_ceilings(r515)

    results = {"seed": args.seed, "noise_ceilings": nc, "rois": {}}
    for roi_name in C.STREAMS_MAIN_ROIS:
        corrected = {
            s: {m: corrs[s][m][roi_name] / nc[roi_name][s] for m in models} for s in C.SUBJECTS
        }
        corrected_515 = {
            s: {m: corrs_515[s][m][roi_name] / nc[roi_name][s] for m in models} for s in C.SUBJECTS
        }
        st = roi.group_stats(corrected, list(models))
        st_515 = roi.group_stats(corrected_515, list(models))
        results["rois"][roi_name] = {
            "raw": {s: {m: corrs[s][m][roi_name] for m in models} for s in C.SUBJECTS},
            "as_released": st,
            "matched_515": st_515,
        }

    out = C.REPORTS / "figures" / "fig3_roi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    # ---- report ----
    print("\n=== Fig. 3 — noise-ceiling-corrected RSA (mean over 8 participants) ===")
    hdr = f"{'model':22s}" + "".join(f"{r:>12s}" for r in C.STREAMS_MAIN_ROIS)
    print(hdr)
    for m in models:
        row = f"{m:22s}"
        for roi_name in C.STREAMS_MAIN_ROIS:
            row += f"{results['rois'][roi_name]['as_released']['mean'][m]:12.3f}"
        print(row)
    print("\nnoise ceilings (LOSO on the 515):")
    for roi_name in C.STREAMS_MAIN_ROIS:
        v = np.mean([nc[roi_name][s] for s in C.SUBJECTS])
        print(f"  {roi_name:12s} {v:.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
