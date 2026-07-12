# M3 + M4 — Text encoder and text-variant representations

**Status: COMPLETE and VERIFIED against the paper's only exactly-published text-side number.**
**Date:** 2026-07-13

---

## 1. What was built

| Artifact | Shape | Construction |
|---|---|---|
| `mpnet_captions_73k.npy` | (73000, 768) | encode each caption independently, mean across the image's captions |
| `mpnet_scrambled_73k.npy` | (73000, 768) | shuffle each caption's NLTK tokens, re-join with the release's `custom_join`, then as above |
| `mpnet_categories_73k.npy` | (73000, 768) | the image's category words joined into **one** string, **one** forward pass |
| `mpnet_nouns_73k.npy` | (73000, 768) | NN/NNS from all captions, **each word encoded separately**, then averaged |
| `mpnet_verbs_73k.npy` | (73000, 768) | VB* likewise |
| `mpnet_allwords_73k.npy` | (73000, 768) | every token of every caption likewise |

Encoder: `sentence-transformers/all-mpnet-base-v2`, HF revision pinned (the release pins
nothing — MI-30).

## 2. Two conventions that the reproduction plan got backwards

The plan inferred (marked `[DERIVED]`) that nouns and verbs were **concatenated** into one
string. They are not. The release loops over words and calls the encoder **once per word**,
then averages (`get_nsd_noun_embeddings_simple.py:87-107`). Category words take the
*opposite* convention: they **are** concatenated into a single string and encoded in one
pass (`get_nsd_sentence_embeddings_categories_simple.py:71-79`).

This is directly measurable in the embedding norms, and it is why we report them:

| variant | ‖e‖ mean | why |
|---|---|---|
| categories | **1.0000** | one forward pass → the encoder's `Normalize` module → unit norm |
| captions | 0.8388 | mean of ~5 unit vectors → not unit norm |
| scrambled | 0.8207 | ditto |
| nouns | 0.6320 | mean of many *single-word* unit vectors, which point in more varied directions |

The norm is a fingerprint of the aggregation convention, so it makes the two conventions
falsifiable rather than a matter of interpretation. **This also settles MI-11**: single-pass
outputs *are* L2-normalised (via the checkpoint's own `modules.json`), but the per-image
**mean** is not, and the release never re-normalises it. Anything that assumes unit-norm
inputs to the ridge or to the RCNN's cosine loss is wrong.

## 3. Claim C11 — the one exactly-published number

> *"MPNet is largely insensitive to word order: mean r = 0.91, s.d. = 0.03."*

This is the only exact number in the paper that validates the text pipeline **with no brain
data**, which makes it the cheapest possible gate on the whole encoder configuration. Result:

| Reading | mean r | s.d. |
|---|---|---|
| **(A)** RDM-level: per participant, mean over that participant's 100-image splits — *the plan's reading* | **0.9121** ✅ | **0.0012** ❌ |
| **(B)** RDM-level: s.d. across all 632 splits pooled | 0.9119 | 0.0095 ❌ |
| **(C)** Per-image: correlation of each image's original vs scrambled 768-d embedding, s.d. across the 73,000 images | 0.8941 ❌ | **0.0365** ✅ |
| | *paper: 0.91* | *paper: 0.03* |

**The mean reproduces exactly under reading (A)** — 0.9121 vs 0.91 — and is stable across
sampler seeds (0.9121 / 0.9121 / 0.9117 for seeds 0/1/2), so the claim itself holds and the
encoder is correctly configured.

**No single reading reproduces both halves of "0.91 ± 0.03".** Reading (A) matches the mean
but its s.d. is 25× too small; reading (C) matches the s.d. but its mean is 0.017 too low.
The likeliest explanation is that the paper's mean and s.d. are computed over **different
units of analysis** — a mean across participants (or splits) reported alongside a spread
across images. Between-participant variance *must* be tiny here, because all 8 subjects'
model RDMs are built from the same 73k embedding matrix and differ only in which images
each subject saw; a between-participant s.d. of 0.03 is not achievable under any
implementation of this pipeline.

**Verdict on C11: REPRODUCED (mean).** The dispersion statistic as published could not be
reproduced under any reading we could construct, and we believe it does not describe the
across-participant spread it is presented as describing. This does not weaken the claim —
insensitivity to word order is if anything *more* robust than the paper's error bar
suggests.

## 4. Known deviations from the original, and why they are unavoidable

- **The scrambler is unseeded in the release** (`random.shuffle`, and there is no
  `random.seed` anywhere in the repository). The authors' exact scrambled strings are
  therefore unrecoverable. We seed ours (`SCRAMBLE_SEED = 0`) and show the result is
  seed-insensitive.
- **The release's scrambled-caption producer is dead code** — nothing in the repo calls
  `scramble_word_order`, yet the downstream figure code consumes a
  `..._mean_embeddings_scrambled.pkl` that no released script produces. Same for the
  adjective/adverb/preposition variants. We reimplement them from the helpers.
- **`verb_adjustments`** (the 232-entry misspelling map, `word_lists.py:126-362`) exists to
  patch words missing from *fastText*, but the release applies it even when the encoder is
  MPNet, where it is unnecessary. It matters only for M5 (fastText/GloVe), so it is
  deferred to that module.

## 5. Tests

`tests/test_modules.py` — 16 passed. The text-side ones:
- `test_caption_embeddings_shape` — (73000, 768)
- `test_single_pass_embeddings_are_unit_norm_but_means_are_not` — pins MI-11
- `test_c11_scrambled_caption_correlation` — gates on mean r = 0.91 ± 0.015
