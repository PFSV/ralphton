# M15 — Activation extraction from the released RCNN checkpoints

**Status: IMPLEMENTED and VERIFIED.** The released weights load, run, and behave correctly.
**Date:** 2026-07-13

---

## 1. Why this module no longer needs 10–40 GPU-days

The reproduction plan classified M14 (RCNN training) as the project's dominant compute cost
and a hard blocker, because the paper never states the number of recurrent timesteps
(MI-32) or the per-layer channel counts (MI-33) — "the network cannot be instantiated".

It can. All 22 trained checkpoints are published on the NSD S3 bucket, and each one ships
`_code_used_for_training/` containing the model definition, the training loop, and
`hparams.txt`. **Reproducing Figs. 4b–4e requires no training at all.**

## 2. Verification against ground truth

The `hparams.pickle` read back out of the checkpoint at load time:

| | |
|---|---|
| `model_name` | `blt_vNet_half_channels` |
| `n_recurrent_steps` | **6** — MI-32 resolved |
| `image_size` | 128 |
| `norm_type` | LN |
| `embedding_loss` | **cosine** (both arms) |
| `model_output_activation` | `no_model_output_activation` (mpnet) / `sigmoid` (multihot) |
| `batch_size` | **256** — the paper says 96 |
| `learning_rate` / `optim_epsilon` | 0.05 / **0.1** — MI-31 confirmed as written |
| `clip_norm` | **500** — mentioned nowhere in the paper |
| `n_epochs` / `n_warmup_epochs` | 200 / 10 |

Two structural facts read straight off the **built model graph**, not inferred:

- **`len(net.outputs) == 6`** — the network emits a readout at *every* timestep, and
  `task.py:91-95` attaches the cosine loss to all of them. This is **deep supervision**, and
  the paper never mentions it. It materially changes what the network optimises.
- **pre-readout output shape `(None, 512)`** — X6 confirmed empirically. Layer 9 has
  `1024 // 2 = 512` channels; `GlobalAvgPool2D` gives the 512-d vector the RSA consumes.
  Tapping the 768-d readout instead breaks Figs. 4c–4e, so `extract()` asserts
  `shape[1] == 512`.

Total parameters: 52,950,368.

## 3. Functional check

`blt_mpnet` ships six sanity images. Feeding them through the released MPNet-arm checkpoint
and comparing the predicted 768-d output against the MPNet embedding of "a photo of a {X}":

| image | cosine to its own caption | best match |
|---|---|---|
| airplane | **0.785** | airplane ✅ |
| boat | **0.747** | boat ✅ |
| bottle | **0.535** | bottle ✅ |
| car | **0.626** | car ✅ |
| cat | **0.757** | cat ✅ |
| dog | **0.759** | dog ✅ |

**Top-1 retrieval: 6/6.** The network genuinely maps images into the LLM's sentence-embedding
space. This validates the whole M15 path — checkpoint loading, architecture reconstruction,
preprocessing, and the pre-readout tap — without needing the 40 GB NSD stimulus file.

## 4. The preprocessing defect we refuse to inherit

The release's input pipeline (`tf_dataset_helper_functions.py:32-49`) takes a **random square
crop of random size** (uniform over sides from `ceil(sqrt(0.33·area))` to the full side) and
applies it on **every split, including test**, with **no seed**:

```python
crop_sizes = tf.cast(tf.random.uniform(shape=[hparams['batch_size']],
                                       minval=min_crop_size, maxval=max_crop_size), tf.int16)
images = tf.map_fn(fn=lambda inp: tf.image.resize(
    tf.image.random_crop(inp[0], [inp[1], inp[1], 3]), [target_image_size]*2, antialias=True), ...)
```

`preprocess_batch_imgs` is on the `.map()` for every split unconditionally
(`make_tf_dataset.py:186`); only the *augmentation* (flip/brightness/saturation/contrast) is
gated on `dataset == 'train'`. There is no `force_no_crop` flag.

**Consequences.** The paper describes the preprocessing as taking "the largest square crop".
The code does not do that. And because the crop is unseeded and applied at test time, the
authors' own NSD activations — and therefore every RDM behind Figs. 4c–4e — differ from run
to run. This is a first-order reproducibility defect in the figure that carries the paper's
headline claim.

`preprocess()` therefore implements both modes:
- `crop="center"` — deterministic, and what the paper actually describes;
- `crop="random"` — the released behaviour, seeded so that at least *our* run is reproducible.

Reporting Figs. 4c–4e under both bounds how much of the result is attributable to the bug.
That comparison is, on its own, a publishable methods finding.

## 5. Limitations

- Needs the NSD stimulus file (`nsd_stimuli.hdf5`, 39.6 GB) to extract the 73k activations;
  deferred until the beta build frees disk.
- The **ecoset** RCNN (doubled channels) and the 13 competitor models (M17) are separate
  downloads.
- Seed handling for Figs. 4c/4d: compute an RDM per seed and average the **correlations**
  across seeds — never average the RDMs first. Enforced in the Fig. 4 driver.
