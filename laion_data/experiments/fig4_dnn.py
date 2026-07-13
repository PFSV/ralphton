"""Figure 4 — the LLM-trained RCNN (M15 -> M16, M17).

Runs off the AUTHORS' OWN released weights (10 MPNet-arm seeds + 10 category-arm seeds).
No training required — see reports/modules/M15_dnn_activations.md.

Fig. 4b  frozen-readout probe: can category labels be read out of an LLM-trained net, and
         vice versa? The claim (C15) is the ASYMMETRY, not the bar heights (which appear
         nowhere in the paper — MI-34 — so C15 is only checkable qualitatively).
Fig. 4c  RCNN activations vs the LLM embeddings they were trained on (C14)
Fig. 4d  LLM-trained vs category-trained RCNN, matched arch/data/dims/seeds (C16)

Seed handling — a trap the plan flagged and we enforce:
    compute an RDM per seed, then average the CORRELATIONS with the brain across seeds.
    Do NOT average the RDMs first. That is a different (and wrong) estimator.

Crop mode — the headline methods finding:
    the release applies an UNSEEDED random crop at test time, while the paper says "the
    largest square crop". We run BOTH and report the gap, which bounds how much of Fig. 4
    is attributable to that bug.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.revision import config as C
from src.revision import dnn
from src.revision import embeddings as E
from src.revision import nsd_data, roi, rsa

SEEDS = list(range(1, 11))
ARMS = {"mpnet": "LLM-trained (ours)", "multihot": "category-trained (control)"}


def stimuli_73k() -> np.ndarray:
    """The 73,000 NSD stimuli as uint8 (73000, 256, 256, 3) — memory-mapped."""
    import h5py

    p = C.NSD_DIR / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    f = h5py.File(p, "r")
    return f["imgBrick"]  # (73000, 425, 425, 3) uint8 in NSD; sliced lazily


def activations(arm: str, seed: int, crop: str, imgs) -> np.ndarray:
    out = C.DERIV / "dnn_act" / f"{arm}_seed{seed}_{crop}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return np.load(out)
    model = f"blt_vNet_half_channels_{arm}_Dec23_seed{seed}"
    a = np.zeros((73000, C.RCNN_PREREADOUT_DIM), dtype=np.float32)
    for lo in range(0, 73000, 2000):
        hi = min(lo + 2000, 73000)
        # offset=lo keys the crop RNG on the GLOBAL image index, so every model and every
        # seed sees the same crop of a given image (required for the matched Fig. 4d contrast)
        a[lo:hi] = dnn.extract(model, np.asarray(imgs[lo:hi]), crop=crop, offset=lo)
    np.save(out, a)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="*", default=["center", "random"])
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()

    imgs = stimuli_73k()
    mask = roi.streams_mask()
    llm_emb = E.load("captions")

    results: dict = {}
    for crop in args.crops:
        results[crop] = {}
        # brain RDMs computed ONCE per (subject, split, ROI) and reused for every model
        brain, splits = {}, {}
        for subj in C.SUBJECTS:
            b = roi.load_betas(subj)
            splits[subj] = rsa.make_splits(subj)
            brain[subj] = {
                r: roi.brain_rdms(roi.roi_patterns(b, mask, r), splits[subj])
                for r in C.STREAMS_MAIN_ROIS
            }
            del b

        for arm in ARMS:
            # per-seed correlations, averaged across seeds AT THE CORRELATION LEVEL
            per_seed: list[dict[str, dict[str, float]]] = []
            for seed in args.seeds:
                act = activations(arm, seed, crop, imgs)
                assert act.shape[1] == 512
                per_seed.append(
                    {s: roi.model_correlations(s, act, brain[s], splits[s]) for s in C.SUBJECTS}
                )
            results[crop][arm] = {
                s: {
                    r: float(np.mean([ps[s][r] for ps in per_seed]))
                    for r in C.STREAMS_MAIN_ROIS
                }
                for s in C.SUBJECTS
            }

        # the LLM embedding itself (Fig. 4c reference): does the RCNN beat its own target?
        results[crop]["llm_embedding"] = {
            s: roi.model_correlations(s, llm_emb, brain[s], splits[s]) for s in C.SUBJECTS
        }

    # ---- report ----
    print("\n=== Fig. 4c/4d — RSA, mean over 8 participants (uncorrected r) ===")
    for crop in args.crops:
        print(f"\ncrop = {crop}" + ("   [the paper's stated preprocessing]" if crop == "center"
                                    else "   [what the released code actually does, unseeded]"))
        hdr = f"{'model':30s}" + "".join(f"{r:>11s}" for r in C.STREAMS_MAIN_ROIS)
        print(hdr)
        for key, label in [("mpnet", ARMS["mpnet"]), ("multihot", ARMS["multihot"]),
                           ("llm_embedding", "MPNet embedding (target)")]:
            row = f"{label:30s}"
            for r in C.STREAMS_MAIN_ROIS:
                v = np.mean([results[crop][key][s][r] for s in C.SUBJECTS])
                row += f"{v:11.4f}"
            print(row)

    p = C.REPORTS / "figures" / "fig4_dnn.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
