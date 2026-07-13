"""M15 -- Activation extraction from the released RCNN checkpoints (`N-ACT`, ours).

Purpose
-------
Extract the representations that re-enter the RSA family (Figs. 4c/4d) and feed the
frozen-readout probe (Fig. 4b), from the authors' OWN trained weights.

Why we use the released weights rather than retraining
------------------------------------------------------
The reproduction plan budgeted 10-40 GPU-days for M14 because the paper's architecture was
believed unrecoverable (MI-32/MI-33). It is recoverable: every checkpoint ships its own
`_code_used_for_training/` directory, and all 20 RCNN seeds + 2 ResNet50s are published.
Reproducing Figs. 4b-4e therefore needs no training at all. Retraining is still worth doing
-- but as an EXTENSION (e.g. hidden assumption H5: cosine-on-sigmoid vs BCE), not as a
prerequisite.

X6 -- the tensor to tap
-----------------------
`layernorm_layer_9_time_5`, spatially average-pooled -> 512 features. Layer 9 has
1024 // 2 = 512 channels (`blt_vNet.py:55` with `divide_n_channels=2`). NOT the 768-d
readout. `include_readout=False` is the default in the release and is never overridden.
We assert `shape[1] == 512`.

Seed handling (a trap the plan flagged)
---------------------------------------
Compute an RDM per seed and average the CORRELATIONS with the brain across seeds -- do NOT
average the RDMs first. SOURCE: nsd_prepare_modelrdms.py / the release's `average_seeds`
path. Averaging RDMs first is a different (and wrong) estimator. Enforced in the Fig. 4
driver, not here.

A reproducibility defect we must NOT inherit
--------------------------------------------
The release's input pipeline (`tf_dataset_helper_functions.py:32-49`) applies a RANDOM
square crop of RANDOM size on EVERY split, including test, with no seed. So the authors'
own activations -- and therefore their Figs. 4c-4e RDMs -- are not reproducible run to run.
The paper describes the preprocessing as "the largest square crop", which is what a
deterministic reimplementation would do.

We implement BOTH:
  crop="center"  -- deterministic, and what the paper actually describes ("largest square
                    crop", i.e. the full 256x256 stimulus, resized to 128)
  crop="random"  -- the released behaviour, seeded so at least OUR run is reproducible
and report the difference as a first-class result, because it bounds how much of Fig. 4 is
attributable to an unseeded preprocessing bug.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np

from . import config as C

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def _configure_gpu() -> None:
    """Pin TF to the emptiest GPU and stop it pre-allocating the whole card.

    This box is shared. TF defaults to GPU 0 and grabs all of its memory, so when another
    user was holding 37 GB there, cuDNN failed with "No algorithm worked!" -- a
    RESOURCE_EXHAUSTED that looks like a model bug but is pure allocation. Select by free
    memory (as ridge.py does) and enable memory growth.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(min(range(len(out)), key=lambda i: int(out[i]))))
    except Exception:
        pass


_configure_gpu()

CKPT_DIR = C.DATA / "checkpoints"
CODE_DIR = C.CODE / "checkpoint_code"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_hparams(model: str) -> dict:
    with open(CKPT_DIR / model / "hparams.pickle", "rb") as f:
        return pickle.load(f)


def build_model(model: str, hparams: dict):
    """Rebuild the network from the code that shipped WITH the weights, then load them.

    The architecture is not in the analysis repo; it lives in
    `_code_used_for_training/models/setup_model.py` inside each checkpoint. We vendored one
    copy (the arms are byte-identical apart from hparams) under code/checkpoint_code/.
    """
    arm = "mpnet" if "mpnet" in model else "multihot"
    code = CODE_DIR / f"blt_vNet_half_channels_{arm}_Dec23_seed1" / "_code_used_for_training" / "models"
    setup = _load_module("setup_model", code / "setup_model.py")

    n_classes = C.MPNET_DIM if arm == "mpnet" else C.N_COCO_CATEGORIES
    input_shape = [None, hparams["image_size"], hparams["image_size"], 3]
    net = setup.get_model_function(hparams["model_name"])(input_shape, n_classes, hparams)
    net.load_weights(str(CKPT_DIR / model / "training_checkpoints" / "ckpt_ep200.h5"))
    return net


def _enable_memory_growth() -> None:
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass  # already initialised


def prereadout_model(net, hparams: dict):
    """A model whose output is the 512-d spatially-averaged pre-readout tensor."""
    import tensorflow as tf

    name = f"LayerNorm_Layer_{C.RCNN_N_LAYERS - 1}_Time_{hparams['n_recurrent_steps'] - 1}"
    layer = net.get_layer(name)
    pooled = tf.keras.layers.GlobalAvgPool2D()(layer.output)
    return tf.keras.Model(inputs=net.inputs, outputs=pooled)


def preprocess(
    images: np.ndarray, size: int, crop: str = "center", offset: int = 0, seed: int = 0
) -> np.ndarray:
    """uint8 (n, H, W, 3) -> float32 (n, size, size, 3) in [-1, 1].

    crop="center": the largest square crop (the stimuli are already square) -- deterministic,
                   and what the paper describes.
    crop="random": the released behaviour -- a random square crop of random SIZE at a random
                   LOCATION, the side at least sqrt(1/3) of the image
                   (tf_dataset_helper_functions.py:38-45).

    `offset` is the index of `images[0]` in the full stimulus set. It is required: both the
    crop size and the crop location are keyed on the GLOBAL image index, so that
      (a) the whole run is reproducible, and
      (b) every model and every seed sees the SAME crop of a given image.
    (b) matters more than it looks. The Fig. 4d comparison (C16) is supposed to be matched
    between the LLM-trained and category-trained arms; if each arm drew its own crops, the
    contrast would be confounded by preprocessing rather than by the training objective.

    A bug we shipped and then fixed: an earlier version seeded only the crop SIZE (via
    `tf.random.Generator.from_seed`) and left the crop LOCATION to `tf.image.random_crop`,
    which draws from TensorFlow's UNSEEDED global RNG. Two identical calls returned different
    tensors. Our claim to have made the released behaviour reproducible was therefore false.
    We now use `tf.image.stateless_random_crop`, whose seed is explicit.
    """
    import tensorflow as tf

    x = tf.convert_to_tensor(images)
    if crop == "random":
        h = float(images.shape[1])
        lo = int(np.ceil(np.sqrt(h * h * 0.33)))
        out = []
        for i in range(images.shape[0]):
            gid = offset + i                      # global image index -> stable per image
            s2 = tf.constant([seed, gid], dtype=tf.int32)
            side = int(
                tf.random.stateless_uniform([], seed=s2, minval=lo, maxval=int(h),
                                            dtype=tf.int32)
            )
            crop_i = tf.image.stateless_random_crop(x[i], [side, side, 3], seed=s2)
            out.append(tf.image.resize(tf.cast(crop_i, tf.float32), [size, size],
                                       antialias=True))
        x = tf.stack(out)
    else:
        x = tf.image.resize(tf.cast(x, tf.float32), [size, size], antialias=True)
    x = x / 255.0                      # Rescaling(1/255)
    x = x * 2.0 - 1.0                  # -> [-1, 1]   (hparams['image_normalization'])
    return x.numpy().astype(np.float32)


def load_extractor(model: str):
    """Build the network ONCE and return (activation_model, hparams).

    Rebuilding a 53M-parameter graph and reloading its weights for every image chunk cost
    ~25 min per model (37 rebuilds over 73,000 images). Build once, reuse across chunks.
    """
    _enable_memory_growth()
    hp = load_hparams(model)
    net = build_model(model, hp)
    return prereadout_model(net, hp), hp


def extract(
    model: str, images: np.ndarray, crop: str = "center", batch: int = 32, offset: int = 0,
    extractor=None,
) -> np.ndarray:
    """-> (n_images, 512) float32 pre-readout activations.

    `offset` is the index of images[0] in the full stimulus set; it keys the crop RNG so the
    result is reproducible and identical across models and seeds.
    `extractor` is an optional (act_model, hparams) pair from `load_extractor`, so callers
    looping over chunks do not rebuild the network each time.
    """
    if extractor is None:
        act, hp = load_extractor(model)
    else:
        act, hp = extractor

    out = np.zeros((len(images), C.RCNN_PREREADOUT_DIM), dtype=np.float32)
    for a in range(0, len(images), batch):
        b = min(a + batch, len(images))
        x = preprocess(images[a:b], hp["image_size"], crop=crop, offset=offset + a)
        out[a:b] = act.predict(x, verbose=0)
    assert out.shape[1] == C.RCNN_PREREADOUT_DIM == 512, (
        f"expected the 512-d pre-readout, got {out.shape[1]} -- X6: you are probably "
        f"tapping the {C.MPNET_DIM}-d readout, which breaks Figs. 4c-4e"
    )
    return out
