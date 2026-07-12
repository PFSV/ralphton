"""M5 -- Non-LLM semantic controls (`N-EMB-WORDVEC`, `N-MULTIHOT`).

Purpose
-------
The control models of Fig. 3a/3c: fastText, GloVe, and the raw multi-hot category vector.
These are the bars the LLM embedding must beat for claims C6 and C10.

Inputs
------
- fastText `crawl-300d-2M.vec` (300-d, Common Crawl 2M words)
- GloVe    `glove.840B.300d.txt` (300-d)
  SOURCE for both files: examples/get_embeddings.py:15-16
- COCO category labels and captions from M2

Outputs
-------
Four (73000, 300) arrays -- {fastText, GloVe} x {category words, all caption words} --
plus a machine-readable OOV decision log. Word vectors are combined by AVERAGING
("word embeddings can be combined additively ... so we average the embeddings across
words").

MI-12 -- the OOV story, corrected
---------------------------------
The reproduction plan concluded that Fig. 3a/3c is "not exactly reproducible, by
construction", because the paper says the authors "either corrected the misspelling, found
a similar word in the fasttext corpus, or removed them" with no published list.

**That list does exist.** `word_lists.py:126-362` contains `verb_adjustments`, a 232-entry
map from misspelled caption tokens to either a correction ("staanding" -> "standing") or
one of two sentinel values, `_____no_embedding_____` / `_____not_verb_/_unknown_____`,
meaning "drop this token". We reproduce it verbatim by parsing it out of the release.

Three caveats that remain, and that we log rather than paper over:
1. `noun_adjustments` is **empty** -- nouns get no spelling correction at all.
2. The map was built from what was missing in *fastText*, but the release applies it
   unconditionally, including to GloVe (different vocabulary) and even to MPNet (where no
   word is ever OOV).
3. Words still OOV after the map are **silently skipped** (`KeyError` -> `pass`). If every
   word of an image is OOV, the fallback is the vector for "something" (nouns/categories)
   or "is" (verbs). The `allwords` route has **no fallback** in the release and would
   produce a NaN; we log any such image instead of emitting NaN.

Category-word specials (`word_lists.py:3-17`): three COCO categories have no single
fastText token and are built as the MEAN of two words --
  baseball-bat -> mean(baseball, bat); baseball-glove -> mean(baseball, glove);
  tennis-racket -> mean(tennis, racket).
SOURCE: get_nsd_category_embeddings_simple.py:78-89.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import config as C
from . import stimuli

WV_DIR = C.DATA / "wordvec"
OUT_DIR = C.DERIV / "embeddings"
WORD_LISTS = C.CODE / "visuo_llm" / "src" / "nsd_visuo_semantics" / "get_embeddings" / "word_lists.py"

DROP_SENTINELS = ("_____no_embedding_____", "_____not_verb_/_unknown_____")

# Categories with no single-token vector; built as the mean of their parts.
# SOURCE: get_nsd_category_embeddings_simple.py:78-89
CATEGORY_COMPOUNDS = {
    "baseball-bat": ["baseball", "bat"],
    "baseball-glove": ["baseball", "glove"],
    "tennis-racket": ["tennis", "racket"],
}

VECTOR_FILES = {
    "fasttext": WV_DIR / "crawl-300d-2M.vec",
    "glove": WV_DIR / "glove.840B.300d.txt",
}


def verb_adjustments() -> dict[str, str]:
    """Parse the release's 232-entry misspelling map verbatim. [MI-12]"""
    src = WORD_LISTS.read_text()
    body = src.split("verb_adjustments = {", 1)[1].split("\n}", 1)[0]
    return dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', body))


def load_vectors(kind: str, vocab: set[str]) -> dict[str, np.ndarray]:
    """Stream the vector file, keeping only the words we need (the full file is 2M x 300)."""
    path = VECTOR_FILES[kind]
    if not path.exists():
        raise FileNotFoundError(f"{path} not downloaded yet")
    out: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if kind == "fasttext" and i == 0:
                continue  # fastText .vec has a "n_words dim" header line
            sp = line.rstrip().split(" ")
            if len(sp) < 10:
                continue
            w = sp[0]
            if w in vocab:
                out[w] = np.asarray(sp[1:], dtype=np.float32)
    return out


def _apply_adjustment(w: str, adj: dict[str, str]) -> str | None:
    """Map a token through the release's misspelling list. None => drop it."""
    if w in adj:
        r = adj[w]
        return None if r in DROP_SENTINELS else r
    return w


def build(kind: str, route: str, force: bool = False) -> Path:
    """route: 'categories' (Fig 3a branch) or 'allwords' (Fig 3c branch)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{kind}_{route}_73k.npy"
    log_path = OUT_DIR / f"{kind}_{route}_oov_log.json"
    if out.exists() and not force:
        return out

    adj = verb_adjustments()

    # ---- assemble the token list per image ----
    if route == "categories":
        names = stimuli.coco_categories_91()
        mh = stimuli.load_multihot()
        per_image = [stimuli.words_from_multihot(row, names) or [C.FALLBACK_CATEGORY] for row in mh]
    elif route == "allwords":
        import nltk

        captions = stimuli.load_captions()
        per_image = []
        for caps in captions:
            toks = [t for s in caps for t in nltk.word_tokenize(s)]
            mapped = [_apply_adjustment(t, adj) for t in toks]
            per_image.append([t for t in mapped if t is not None])
    else:
        raise ValueError(route)

    # ---- vocabulary we must look up ----
    vocab: set[str] = {C.FALLBACK_CATEGORY, C.FALLBACK_NOUN, C.FALLBACK_VERB}
    for words in per_image:
        vocab.update(words)
    for parts in CATEGORY_COMPOUNDS.values():
        vocab.update(parts)

    vecs = load_vectors(kind, vocab)
    dim = len(next(iter(vecs.values())))

    def lookup(w: str) -> np.ndarray | None:
        if w in vecs:
            return vecs[w]
        if w in CATEGORY_COMPOUNDS:  # mean(baseball, bat), etc.
            parts = [vecs[p] for p in CATEGORY_COMPOUNDS[w] if p in vecs]
            if parts:
                return np.mean(parts, axis=0)
        return None

    # ---- embed ----
    emb = np.zeros((73000, dim), dtype=np.float32)
    oov: dict[str, int] = {}
    n_fallback = 0
    for i, words in enumerate(per_image):
        got = []
        for w in words:
            v = lookup(w)
            if v is None:
                oov[w] = oov.get(w, 0) + 1
            else:
                got.append(v)
        if not got:
            n_fallback += 1
            fb = lookup(C.FALLBACK_CATEGORY)
            assert fb is not None, "fallback word 'something' is itself OOV"
            got = [fb]
        emb[i] = np.mean(got, axis=0)

    np.save(out, emb)
    with open(log_path, "w") as f:
        json.dump(
            {
                "kind": kind,
                "route": route,
                "n_images_using_fallback": n_fallback,
                "n_distinct_oov_tokens": len(oov),
                "n_oov_occurrences": int(sum(oov.values())),
                "oov_tokens": dict(sorted(oov.items(), key=lambda kv: -kv[1])),
            },
            f,
            indent=1,
        )
    return out


def build_multihot() -> Path:
    """`N-MULTIHOT` -- the raw binary category vector, used directly as a model RDM."""
    out = OUT_DIR / "multihot_73k_model.npy"
    np.save(out, stimuli.load_multihot().astype(np.float32))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="M5: fastText / GloVe / multi-hot controls")
    ap.add_argument("--kinds", nargs="*", default=["fasttext", "glove"])
    ap.add_argument("--routes", nargs="*", default=["categories", "allwords"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    p = build_multihot()
    print(f"[M5] multihot   {np.load(p).shape} -> {p.name}", flush=True)

    for kind in args.kinds:
        for route in args.routes:
            p = build(kind, route, force=args.force)
            e = np.load(p)
            log = json.loads((OUT_DIR / f"{kind}_{route}_oov_log.json").read_text())
            print(
                f"[M5] {kind:8s} {route:10s} {e.shape}  "
                f"OOV: {log['n_distinct_oov_tokens']} distinct / "
                f"{log['n_oov_occurrences']} occurrences, "
                f"{log['n_images_using_fallback']} images fell back",
                flush=True,
            )
