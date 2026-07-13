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
import time

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


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def activations(arm: str, seed: int, crop: str, imgs) -> np.ndarray:
    out = C.DERIV / "dnn_act" / f"{arm}_seed{seed}_{crop}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        say(f"  {out.name}: cached")
        return np.load(out)
    model = f"blt_vNet_half_channels_{arm}_Dec23_seed{seed}"
    t0 = time.time()
    extractor = dnn.load_extractor(model)   # build the graph ONCE, not once per chunk
    say(f"  {model} [{crop}]: graph built in {time.time() - t0:.0f}s, extracting 73,000 images")
    a = np.zeros((73000, C.RCNN_PREREADOUT_DIM), dtype=np.float32)
    for lo in range(0, 73000, 2000):
        hi = min(lo + 2000, 73000)
        # offset=lo keys the crop RNG on the GLOBAL image index, so every model and every
        # seed sees the same crop of a given image (required for the matched Fig. 4d contrast)
        a[lo:hi] = dnn.extract(model, np.asarray(imgs[lo:hi]), crop=crop, offset=lo,
                               extractor=extractor)
        el = time.time() - t0
        say(f"    {hi:6d}/73000  {el / 60:5.1f} min elapsed, "
            f"~{el / hi * (73000 - hi) / 60:5.1f} min left")
    np.save(out, a)
    say(f"  {out.name}: written in {(time.time() - t0) / 60:.1f} min")
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="*", default=["center", "random"])
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()

    say(f"crops={args.crops}  seeds={args.seeds}")
    imgs = stimuli_73k()
    mask = roi.streams_mask()
    llm_emb = E.load("captions")

    results: dict = {}
    for crop in args.crops:
        results[crop] = {}
        # brain RDMs computed ONCE per (subject, split, ROI) and reused for every model
        say(f"[{crop}] brain RDMs: 8 subjects x {len(C.STREAMS_MAIN_ROIS)} ROIs")
        brain, splits, r515 = {}, {}, {}
        c515 = nsd_data.get_conditions_515()
        for subj in C.SUBJECTS:
            t0 = time.time()
            b = roi.load_betas(subj)
            _, keep = nsd_data.get_conditions_3rep(subj)
            pos515 = np.searchsorted(keep, c515)
            splits[subj] = rsa.make_splits(subj)
            brain[subj], r515[subj] = {}, {}
            for r in C.STREAMS_MAIN_ROIS:
                X = roi.roi_patterns(b, mask, r)
                brain[subj][r] = roi.brain_rdms(X, splits[subj])
                r515[subj][r] = rsa.rdm(X[pos515])          # for the noise ceiling
            del b
            say(f"  {subj}: {len(splits[subj])} splits, {time.time() - t0:.0f}s")

        for arm in ARMS:
            say(f"[{crop}] arm={arm} ({ARMS[arm]})")
            # per-seed correlations, averaged across seeds AT THE CORRELATION LEVEL
            per_seed: list[dict[str, dict[str, float]]] = []
            for seed in args.seeds:
                act = activations(arm, seed, crop, imgs)
                assert act.shape[1] == 512
                t0 = time.time()
                per_seed.append(
                    {s: roi.model_correlations(s, act, brain[s], splits[s]) for s in C.SUBJECTS}
                )
                mu = np.mean([per_seed[-1][s][r] for s in C.SUBJECTS
                              for r in C.STREAMS_MAIN_ROIS])
                say(f"  seed{seed}: RSA done in {time.time() - t0:.0f}s, mean r over "
                    f"8 subj x {len(C.STREAMS_MAIN_ROIS)} ROIs = {mu:.4f}")
            results[crop][arm] = {
                s: {
                    r: float(np.mean([ps[s][r] for ps in per_seed]))
                    for r in C.STREAMS_MAIN_ROIS
                }
                for s in C.SUBJECTS
            }

        # the LLM embedding itself: does the RCNN beat the very target it was trained on? (C14)
        say(f"[{crop}] MPNet caption embedding (the RCNN's training target)")
        results[crop]["llm_embedding"] = {
            s: roi.model_correlations(s, llm_emb, brain[s], splits[s]) for s in C.SUBJECTS
        }

        # Fig. 4e is noise-ceiling-corrected (caption: "Noise-ceiling-corrected correlations";
        # release: roi_analyses.py DO_NOISE_CEILING=True, nsd_roi_analyses.py:162 divides).
        # Plain division, ROI analyses only -- same convention as our Fig. 3.
        say(f"[{crop}] noise ceilings (LOSO on the shared 515)")
        nc = roi.noise_ceilings(r515)
        results[crop]["noise_ceilings"] = nc
        keys = ["mpnet", "multihot", "llm_embedding"]
        results[crop]["corrected"] = {
            r: {s: {k: results[crop][k][s][r] / nc[r][s] for k in keys} for s in C.SUBJECTS}
            for r in C.STREAMS_MAIN_ROIS
        }
        results[crop]["stats"] = {
            r: roi.group_stats(results[crop]["corrected"][r], keys)
            for r in C.STREAMS_MAIN_ROIS
        }

    # ---- report ----
    # NOT Fig. 4c/4d. Those two panels are volumetric SEARCHLIGHT contrast maps (radius 6,
    # func1pt8mm; nsd_searchlight_main_tf.py:24-26), which we do not compute -- the volumetric
    # betas were never downloaded. What follows is the streams-ROI, fsaverage analysis, which is
    # the authors' OWN adjudication of the same two contrasts (examples/roi_analyses.py,
    # PAPER_FIG=='fig4') and is what Fig. 4e plots. It tests C14 and C16 directionally; it
    # cannot test their spatial extent ("a wide network of higher visual areas").
    print("\n=== Fig. 4e — streams-ROI RSA, noise-ceiling-corrected, mean over 8 participants ===")
    for crop in args.crops:
        print(f"\ncrop = {crop}" + ("   [the paper's stated preprocessing]" if crop == "center"
                                    else "   [what the released code actually does, unseeded]"))
        hdr = f"{'model':30s}" + "".join(f"{r:>11s}" for r in C.STREAMS_MAIN_ROIS)
        print(hdr)
        for key, label in [("mpnet", ARMS["mpnet"]), ("multihot", ARMS["multihot"]),
                           ("llm_embedding", "MPNet embedding (target)")]:
            row = f"{label:30s}"
            for r in C.STREAMS_MAIN_ROIS:
                v = np.mean([results[crop]["corrected"][r][s][key] for s in C.SUBJECTS])
                row += f"{v:11.4f}"
            print(row)

        print("\n  contrasts (paired t over the 8 participants, BH-FDR; the release uses an "
              "unpaired\n  ttest_ind on this within-subject design -- we report both):")
        for r in C.STREAMS_MAIN_ROIS:
            st = results[crop]["stats"][r]
            c14 = _pair(st, "mpnet", "llm_embedding")
            c16 = _pair(st, "mpnet", "multihot")
            print(f"    {r:9s} C14 RCNN>MPNet: d={c14[0]:+.4f} p_paired={c14[1]:.2e} "
                  f"p_unpaired={c14[2]:.2e}   "
                  f"C16 LLM>categ: d={c16[0]:+.4f} p_paired={c16[1]:.2e} p_unpaired={c16[2]:.2e}")

    p = C.REPORTS / "figures" / "fig4_dnn.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(p_safe(results), open(p, "w"), indent=1)
    print(f"\nwrote {p}")


def _pair(st: dict, a: str, b: str) -> tuple[float, float, float]:
    """(mean difference a-b, paired p, unpaired p) for one ROI out of roi.group_stats.

    group_stats stores each comparison once, under whichever order the model list gave it, so
    look the pair up in both orders. The p-values are order-free (two-sided); only the
    difference carries a sign, and we always report it as a-b.
    """
    i = st["pairs"].index((a, b)) if (a, b) in st["pairs"] else st["pairs"].index((b, a))
    d = st["mean"][a] - st["mean"][b]
    return (d, float(st["p_paired_fdr"][i]), float(st["p_unpaired_fdr"][i]))


def p_safe(o):
    """numpy/tuple -> json"""
    if isinstance(o, dict):
        return {str(k): p_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [p_safe(v) for v in o]
    if hasattr(o, "tolist"):
        return o.tolist()
    return o


if __name__ == "__main__":
    main()
