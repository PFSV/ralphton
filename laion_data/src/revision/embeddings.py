"""M3 + M4 -- Text encoder core and text-variant representations.

M3 (`N-MPNET`, `N-EMB-CAPTION`)
------------------------------
The single text encoder serving every text-side use in the paper. Canonical construction:
encode each of an image's captions independently -> (n_caps, 768) -> mean across captions.
SOURCE: get_nsd_sentence_embeddings_simple.py:58-67 (`np.mean(img_embeddings, axis=0)`);
`get_embeddings` -> `embedding_model.encode(sentences)` with all defaults
(embedding_models_zoo.py:76).

MI-11 (MPNet pooling / normalization) -- RESOLVED
    The release passes no `normalize_embeddings` kwarg, so it defaults to False. BUT
    `all-mpnet-base-v2`'s own `modules.json` ships a `Normalize` module, so the encoder's
    outputs are unit-norm regardless. The distinction is therefore moot for the *encoder
    output*; it is NOT moot for the per-image mean, which is a mean of unit vectors and is
    NOT itself unit-norm. We reproduce that faithfully (no re-normalisation after the mean).

MI-30 (checkpoint drift) -- MITIGATED: we pin the HF revision in config.MPNET_REVISION.
    The release pins nothing.

M4 (text variants)
------------------
Each variant changes ONLY the string that is fed to the frozen encoder, so any difference
in brain alignment is attributable to linguistic content.

  captions   : mean over the image's captions                      [reference]
  scrambled  : word order shuffled within each caption, then mean  [claim C11]
  nouns      : NN/NNS from all captions, each word encoded SEPARATELY, then averaged
  verbs      : VB* from all captions, likewise
  allwords   : every token of every caption, likewise
  categories : the image's category words joined into ONE string, ONE forward pass

Two facts here contradict the reproduction plan's [DERIVED] guess and are worth flagging:
  - nouns/verbs are AVERAGED per-word, NOT concatenated into one string.
    SOURCE: get_nsd_noun_embeddings_simple.py:87-107 (loop over words, `get_word_embedding`
    per word, then `np.mean`). The per-word route means duplicates are frequency-weighted.
  - category words ARE concatenated into one string (the opposite convention).
    SOURCE: get_nsd_sentence_embeddings_categories_simple.py:71-79 (`" ".join(...)`).

POS tagging: NLTK `pos_tag(word_tokenize(...))`, Penn Treebank. Nouns are NN/NNS only --
proper nouns (NNP/NNPS) are EXCLUDED. SOURCE: nsd_embeddings_utils.py:120.

Fallbacks when an image yields no word of the requested type: "something" (nouns,
categories), "is" (verbs). SOURCE: as cited in config.py.

Determinism: the release's scrambler is unseeded (`random.shuffle`, no seed anywhere in
the repo), so the paper's exact scrambled strings are unrecoverable. We seed it
(config.SCRAMBLE_SEED) so our own result is reproducible, and we verify the published
statistic (r = 0.91 +/- 0.03) is seed-insensitive.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from . import config as C
from . import stimuli

EMB_DIR = C.DERIV / "embeddings"


# ======================================================================================
# Encoder
# ======================================================================================
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(C.MPNET_MODEL, revision=C.MPNET_REVISION, device="cuda")
    return _MODEL


def encode(sentences: list[str], batch_size: int = 512) -> np.ndarray:
    """Encode strings with the frozen encoder. Returns (n, 768) float32."""
    model = get_model()
    return model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def _encode_grouped(groups: list[list[str]], batch_size: int = 512) -> np.ndarray:
    """Encode a list of string-groups, returning the per-group MEAN embedding.

    All strings are flattened into one batch (so the GPU sees a full batch regardless of
    group size), then means are taken per group. Empty groups must be pre-filled by the
    caller; an empty group here is an error.
    """
    lens = np.array([len(g) for g in groups])
    if (lens == 0).any():
        raise ValueError("empty group: caller must apply the fallback word first")
    flat = [s for g in groups for s in g]
    emb = encode(flat, batch_size=batch_size)
    out = np.zeros((len(groups), emb.shape[1]), dtype=np.float32)
    ends = np.cumsum(lens)
    starts = ends - lens
    for i, (a, b) in enumerate(zip(starts, ends)):
        out[i] = emb[a:b].mean(axis=0)
    return out


# ======================================================================================
# Text variants
# ======================================================================================
def _pos_words(captions: list[str], kind: str) -> list[str]:
    import nltk

    tags = set(C.POS_TAGS[kind])
    words: list[str] = []
    for s in captions:
        for w, t in nltk.pos_tag(nltk.word_tokenize(s)):
            if t in tags:
                words.append(w)
    return words


def _scramble(s: str, rng: random.Random) -> str:
    """Shuffle the token order of one caption.

    SOURCE: nsd_embeddings_utils.py:146-150 -- tokenize with NLTK, `shuffle` in place,
    re-join. Punctuation tokens are shuffled along with words, exactly as in the release.
    """
    import nltk

    toks = nltk.word_tokenize(s)
    rng.shuffle(toks)
    return " ".join(toks)


def build_variant(name: str, force: bool = False) -> Path:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    out = EMB_DIR / f"mpnet_{name}_73k.npy"
    if out.exists() and not force:
        return out

    captions = stimuli.load_captions()

    if name == "captions":
        groups = captions

    elif name == "scrambled":
        rng = random.Random(C.SCRAMBLE_SEED)
        groups = [[_scramble(s, rng) for s in caps] for caps in captions]

    elif name in ("nouns", "verbs"):
        kind = "noun" if name == "nouns" else "verb"
        fallback = C.FALLBACK_NOUN if name == "nouns" else C.FALLBACK_VERB
        groups = []
        for caps in captions:
            w = _pos_words(caps, kind)
            groups.append(w if w else [fallback])

    elif name == "allwords":
        import nltk

        groups = []
        for caps in captions:
            w = [t for s in caps for t in nltk.word_tokenize(s)]
            groups.append(w if w else [C.FALLBACK_NOUN])

    elif name == "categories":
        names = stimuli.coco_categories_91()
        mh = stimuli.load_multihot()
        # ONE string per image, ONE forward pass (unlike nouns/verbs).
        strings = []
        for row in mh:
            words = stimuli.words_from_multihot(row, names)
            strings.append(" ".join(words) if words else C.FALLBACK_CATEGORY)
        emb = encode(strings)
        np.save(out, emb)
        return out

    else:
        raise ValueError(f"unknown variant: {name}")

    emb = _encode_grouped(groups)
    np.save(out, emb)
    return out


def load(name: str) -> np.ndarray:
    return np.load(EMB_DIR / f"mpnet_{name}_73k.npy")


VARIANTS = ["captions", "scrambled", "categories", "nouns", "verbs", "allwords"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="M3/M4: MPNet embeddings and text variants")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import nltk

    for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    for v in args.variants:
        p = build_variant(v, force=args.force)
        e = np.load(p)
        norms = np.linalg.norm(e, axis=1)
        print(f"[M3/M4] {v:11s} {e.shape}  |e| mean={norms.mean():.4f} sd={norms.std():.4f}  -> {p.name}",
              flush=True)
