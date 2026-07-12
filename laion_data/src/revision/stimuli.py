"""M2 -- Stimulus and annotation ingest (`D-COCO-CAP`, `D-COCO-CAT`).

Purpose
-------
Assemble the non-brain input artifacts: the 5 COCO captions and the 91-d multi-hot
category vector for each of the 73,000 NSD images, via the NSD->COCO id mapping.

Inputs
------
- `data/nsd/nsddata/experiments/nsd/nsd_stim_info_merged.csv`  (nsdId -> cocoId, cocoSplit)
- `data/coco/annotations/{captions,instances}_{train,val}2017.json`

Outputs
-------
- `data/derivatives/captions_73k.json`    list[list[str]], len 73000
- `data/derivatives/multihot_73k.npy`     (73000, 91) uint8

Resolved ambiguities
--------------------
COCO release year (not stated in the paper, and the release repo delegates it to an
unpinned `nsd_access` dependency): NSD's own `nsd_stim_info_merged.csv` carries a
`cocoSplit` column whose values are `train2017` / `val2017`. So the 2017 release is not a
guess -- it is recorded in the NSD metadata. [resolves the COCO-year UNKNOWN]

MI-07 -- K, the multi-hot width. The release uses a 91-entry category list
(`word_lists.py:18-110`), corroborated by `blt_mpnet/example.py:46`
(`OUTPUT_DIM = 768 if 'mpnet' ... else 91`). Verified here: that list is ordered by COCO
category id, i.e. `coco_categories_91[i]` is the name of COCO category `i+1` (checked
against the deprecated ids: index 9 = traffic-light = id 10, index 11 = signpost = the
deleted "street sign" = id 12, ...). We therefore build a length-91 vector indexed by
COCO category id (ids run 1..90; index 0 is never set). This makes the release's
`get_words_from_multihot` -> `coco_categories_91[lin - 1]` decode correctly, which is the
consistency check that pins the convention. [MI-07 -- RESOLVED: K = 91]

Note: only 80 of the 91 categories are actually annotated in COCO; the 11 deprecated ones
(street sign, hat, shoe, eye glasses, plate, mirror, window, desk, door, blender,
hair brush) can never be set. We keep the width at 91 to match the trained networks'
readout dimension.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# The 91 category names exactly as the release spells them (with the fastText-motivated
# renames: fire hydrant -> fireplug, street sign -> signpost, parking meter ->
# self-parking, potted plant -> houseplant, hair drier -> hairdryer, ...).
# SOURCE: code/visuo_llm/src/nsd_visuo_semantics/get_embeddings/word_lists.py:18-110
_WORD_LISTS = C.CODE / "visuo_llm" / "src" / "nsd_visuo_semantics" / "get_embeddings" / "word_lists.py"


def coco_categories_91() -> list[str]:
    """Parse the release's verbatim 91-name category list (index i <-> COCO id i+1)."""
    src = _WORD_LISTS.read_text()
    body = src.split("coco_categories_91 = [", 1)[1].split("]", 1)[0]
    names = re.findall(r'"([^"]+)"', body)
    if len(names) != C.N_COCO_CATEGORIES:
        raise ValueError(f"expected {C.N_COCO_CATEGORIES} categories, parsed {len(names)}")
    return names


def nsd_to_coco() -> pd.DataFrame:
    csv = C.NSD_DIR / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
    df = pd.read_csv(csv)[["nsdId", "cocoId", "cocoSplit"]]
    if len(df) != 73000:
        raise ValueError(f"expected 73000 NSD stimuli, got {len(df)}")
    return df.sort_values("nsdId").reset_index(drop=True)


def _load_coco_json(kind: str) -> dict:
    """Merge the train2017 and val2017 halves of a COCO annotation file."""
    merged: dict = {}
    for split in ("train2017", "val2017"):
        p = C.COCO_DIR / "annotations" / f"{kind}_{split}.json"
        with open(p) as f:
            d = json.load(f)
        for ann in d["annotations"]:
            merged.setdefault(ann["image_id"], []).append(ann)
    return merged


def build(force: bool = False) -> tuple[Path, Path]:
    C.DERIV.mkdir(parents=True, exist_ok=True)
    cap_path = C.DERIV / "captions_73k.json"
    cat_path = C.DERIV / "multihot_73k.npy"
    if cap_path.exists() and cat_path.exists() and not force:
        return cap_path, cat_path

    stim = nsd_to_coco()
    coco_ids = stim["cocoId"].to_numpy()

    # --- captions ---
    cap_ann = _load_coco_json("captions")
    captions: list[list[str]] = []
    for cid in coco_ids:
        anns = cap_ann.get(int(cid), [])
        # COCO orders annotations by id; keep that order for determinism.
        anns = sorted(anns, key=lambda a: a["id"])
        captions.append([a["caption"].strip() for a in anns])

    n_caps = np.array([len(c) for c in captions])
    if (n_caps == 0).any():
        raise ValueError(f"{(n_caps == 0).sum()} NSD images have no COCO caption")

    # --- categories (multi-hot, length 91, indexed by COCO category id) ---
    inst_ann = _load_coco_json("instances")
    multihot = np.zeros((73000, C.N_COCO_CATEGORIES), dtype=np.uint8)
    for i, cid in enumerate(coco_ids):
        for a in inst_ann.get(int(cid), []):
            multihot[i, a["category_id"]] = 1  # ids are 1..90; index 0 stays 0

    with open(cap_path, "w") as f:
        json.dump(captions, f)
    np.save(cat_path, multihot)
    return cap_path, cat_path


def load_captions() -> list[list[str]]:
    with open(C.DERIV / "captions_73k.json") as f:
        return json.load(f)


def load_multihot() -> np.ndarray:
    return np.load(C.DERIV / "multihot_73k.npy")


def words_from_multihot(row: np.ndarray, names: list[str]) -> list[str]:
    """Decode a multi-hot row to category words.

    Mirrors the release's `get_words_from_multihot`, which indexes
    `coco_categories_91[lin - 1]` for each set bit `lin` -- correct precisely because the
    vector is indexed by COCO category id.
    """
    return [names[i - 1] for i in np.flatnonzero(row)]


if __name__ == "__main__":
    cap, cat = build()
    caps = load_captions()
    mh = load_multihot()
    n = np.array([len(c) for c in caps])
    print(f"[M2] captions: {len(caps)} images, {n.min()}-{n.max()} caps each "
          f"({(n == 5).mean() * 100:.1f}% have exactly 5)")
    print(f"[M2] multihot: {mh.shape}, {mh.sum(1).mean():.2f} categories/image, "
          f"{(mh.sum(1) == 0).sum()} images with no category")
    print(f"[M2] active category columns: {(mh.sum(0) > 0).sum()} of {C.N_COCO_CATEGORIES}")
