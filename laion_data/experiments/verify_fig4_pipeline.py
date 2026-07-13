"""Three checks the Fig. 4 numbers rest on, none of which the driver itself performs.

Motivated by an adversarial audit of the Fig. 4 code path (reports/modules/M15_M17_fig4.md §3).

V1  Are the released weights ACTUALLY in the graph?
    The driver's only guard is `assert act.shape[1] == 512`, which a randomly-initialised net
    passes just as happily. If `load_weights` ever silently no-ops, every Fig. 4 number becomes
    a measurement of Keras's initialiser. We compare the in-graph tensors against the raw HDF5
    checkpoint, and confirm an unloaded net differs.

V2  How much does the resize path matter?
    The release never feeds the net the 425x425 NSD stimuli. It reads
    `ms_coco_embeddings_square256.h5` (get_dnn_activities.py:11) -- images already resampled to
    256 -- and only then crops and resizes to 128. That file is a private cluster path and was
    NOT published, so we cannot consume it. We resize 425 -> 128 in one step; they effectively
    did 425 -> 256 -> 128. This measures the gap instead of arguing about it: same net, same
    images, two resize paths, compared at the level of the thing we actually report (an RDM).

V3  Would a broken checkpoint be caught?
    Brain RSA of a randomly-initialised net, same architecture, same images. This is the floor.
    If the trained net does not clear it by a wide margin, nothing else in Fig. 4 means anything.

Run:  python3 -u -m experiments.verify_fig4_pipeline
"""

from __future__ import annotations

import h5py
import numpy as np

from src.revision import config as C
from src.revision import dnn, roi, rsa
from src.revision.rsa import corr_rdms, rdm

N_IMG = 2000          # enough for a stable 100-image-split RDM comparison, cheap on GPU
MODEL = "blt_vNet_half_channels_mpnet_Dec23_seed1"


def stimuli(n: int) -> np.ndarray:
    p = C.NSD_DIR / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    with h5py.File(p, "r") as f:
        return np.asarray(f["imgBrick"][:n])


def v1_weights_are_loaded() -> None:
    print("\n=== V1  released weights are in the graph ===")
    hp = dnn.load_hparams(MODEL)
    trained = dnn.build_model(MODEL, hp)                     # load_weights called inside

    ckpt = dnn.CKPT_DIR / MODEL / "training_checkpoints" / "ckpt_ep200.h5"
    # pull the first conv kernel straight out of the HDF5, bypassing Keras entirely
    with h5py.File(ckpt, "r") as f:
        def first_kernel(g):
            for k in g:
                item = g[k]
                if isinstance(item, h5py.Dataset):
                    if item.ndim == 4:                        # (kh, kw, cin, cout)
                        return item.name, np.asarray(item)
                else:
                    got = first_kernel(item)
                    if got:
                        return got
            return None

        name, disk = first_kernel(f)
    print(f"  checkpoint tensor : {name}  {disk.shape}")

    hit = None
    for layer in trained.layers:
        for w in layer.get_weights():
            if w.shape == disk.shape and np.array_equal(w, disk):
                hit = layer.name
                break
        if hit:
            break
    assert hit is not None, "NO in-graph tensor equals the checkpoint kernel -- weights NOT loaded"
    print(f"  found bit-identical in layer: {hit}")

    # and a net built WITHOUT loading must not accidentally match
    fresh = _fresh_net(hp)
    same = any(
        np.array_equal(w, disk)
        for layer in fresh.layers
        for w in layer.get_weights()
        if w.shape == disk.shape
    )
    assert not same, "a freshly-initialised net matches the checkpoint -- the test is vacuous"
    print("  PASS: trained graph == checkpoint, fresh graph != checkpoint")


def _fresh_net(hp: dict):
    """Same architecture, random init -- build_model() minus the load_weights call."""
    setup = dnn._load_module(
        "setup_model",
        dnn.CODE_DIR / "blt_vNet_half_channels_mpnet_Dec23_seed1"
        / "_code_used_for_training" / "models" / "setup_model.py",
    )
    shape = [None, hp["image_size"], hp["image_size"], 3]
    return setup.get_model_function(hp["model_name"])(shape, C.MPNET_DIM, hp)


def _rdm_corr(a: np.ndarray, b: np.ndarray, n_split: int = 20) -> float:
    """Mean correlation between the RDMs of two feature sets over disjoint 100-image splits."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(a.shape[0])[: n_split * C.RSA_SAMPLE_SIZE]
    rs = []
    for s in np.split(idx, n_split):
        rs.append(corr_rdms(rdm(a[s]), rdm(b[s])))
    return float(np.mean(rs))


def v2_resize_path(imgs: np.ndarray) -> None:
    print("\n=== V2  425 -> 128 (ours) vs 425 -> 256 -> 128 (the release's square256 dataset) ===")
    import tensorflow as tf

    extractor = dnn.load_extractor(MODEL)
    act_direct = dnn.extract(MODEL, imgs, crop="center", offset=0, extractor=extractor)

    # the release's offline stage: an antialiased resample to 256, stored as uint8
    two_step = tf.image.resize(tf.cast(imgs, tf.float32), [256, 256], antialias=True)
    two_step = np.clip(two_step.numpy(), 0, 255).astype(np.uint8)
    act_256 = dnn.extract(MODEL, two_step, crop="center", offset=0, extractor=extractor)

    r_rdm = _rdm_corr(act_direct, act_256)
    # per-image feature agreement, for reference
    a = act_direct - act_direct.mean(0)
    b = act_256 - act_256.mean(0)
    cos = float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))
    print(f"  RDM correlation between the two paths : {r_rdm:.4f}")
    print(f"  mean per-image feature cosine          : {cos:.4f}")
    print("  (r_RDM ~ 1.0 => the missing 425->256 stage is immaterial to anything we report)")


def v3_untrained_floor(imgs: np.ndarray) -> None:
    print("\n=== V3  brain RSA floor: a randomly-initialised net of the same architecture ===")
    subj = C.SUBJECTS[0]
    betas = roi.load_betas(subj)
    mask = roi.streams_mask()
    splits = rsa.make_splits(subj)
    brain = {r: roi.brain_rdms(roi.roi_patterns(betas, mask, r), splits)
             for r in C.STREAMS_MAIN_ROIS}
    del betas

    # the trained net, on ALL 73k images (cached by the Fig. 4 run)
    trained_act = np.load(C.DERIV / "dnn_act" / f"mpnet_seed1_center.npy")

    hp = dnn.load_hparams(MODEL)
    fresh = dnn.prereadout_model(_fresh_net(hp), hp)
    rand_act = np.zeros((73000, C.RCNN_PREREADOUT_DIM), dtype=np.float32)
    for lo in range(0, 73000, 2000):
        hi = min(lo + 2000, 73000)
        rand_act[lo:hi] = dnn.extract(MODEL, _all_stim(lo, hi), crop="center", offset=lo,
                                      extractor=(fresh, hp))

    t = roi.model_correlations(subj, trained_act, brain, splits)
    u = roi.model_correlations(subj, rand_act, brain, splits)
    print(f"  {'ROI':10s}{'trained':>10s}{'random init':>14s}{'ratio':>8s}")
    for r in C.STREAMS_MAIN_ROIS:
        print(f"  {r:10s}{t[r]:10.4f}{u[r]:14.4f}{t[r] / u[r]:8.2f}x")
    print("  (a random net is NOT at zero -- architecture alone carries image structure.")
    print("   This is the floor any Fig. 4 number must clear, and the driver never checked it.)")


def _all_stim(lo: int, hi: int) -> np.ndarray:
    p = C.NSD_DIR / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    with h5py.File(p, "r") as f:
        return np.asarray(f["imgBrick"][lo:hi])


def main() -> None:
    v1_weights_are_loaded()
    imgs = stimuli(N_IMG)
    v2_resize_path(imgs)
    v3_untrained_floor(imgs)


if __name__ == "__main__":
    main()
