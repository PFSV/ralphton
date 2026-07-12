"""M11 (part) — embed the Google Conceptual Captions dictionary for caption retrieval.

The decoder predicts a 768-d MPNet vector from brain activity; the predicted vector is then
matched against every caption in a large external dictionary, and the argmax is reported as
the "reconstructed caption". The dictionary is GCC (train + validation), embedded with the
same frozen MPNet encoder used everywhere else.

SOURCE: encoding_decoding_utils.py:138-140 (`conceptual_captions_{train,val}.tsv` +
`conceptual_captions_mpnet_{train,val}.npy`).

Retrieval metric: the release's main decoding script uses scipy `cdist(..., metric=METRIC)`
with METRIC="correlation" (examples/llm_decoding_model.py:7), i.e. 1 - Pearson r after
row-centring. Note `decoding_extra_analyses.py:146` hard-codes 'cosine' instead — the two
retrieval sites in the release disagree. We use "correlation", the one that drives Fig. 2b.
"""
from __future__ import annotations
import json
import numpy as np
from . import config as C
from . import embeddings as E

GCC_DIR = C.DATA / "gcc"


def build(batch: int = 1024) -> tuple[np.ndarray, list[str]]:
    out = C.DERIV / "gcc_mpnet.npy"
    caps = json.load(open(GCC_DIR / "gcc_captions.json"))
    if out.exists():
        return np.load(out, mmap_mode="r"), caps
    emb = np.zeros((len(caps), C.MPNET_DIM), dtype=np.float16)
    for a in range(0, len(caps), 100_000):
        b = min(a + 100_000, len(caps))
        emb[a:b] = E.encode(caps[a:b], batch_size=batch).astype(np.float16)
        print(f"[GCC] {b}/{len(caps)}", flush=True)
    np.save(out, emb)
    return emb, caps


if __name__ == "__main__":
    e, c = build()
    print(f"[GCC] dictionary: {e.shape} for {len(c):,} captions")
