"""Figure 4b — the frozen-readout probe (M16).

Tests the asymmetric-subsumption claim (C15): can category labels be read out of an
LLM-trained network, and can LLM embeddings be read out of a category-trained one?

The claim IS the asymmetry, not the bar heights — which appear nowhere in the paper or its
caption (MI-34), so C15 can be reproduced qualitatively but not checked numerically.

Design (SOURCE: transfer_readout.py, and the paper's Methods)
------------------------------------------------------------
- Inputs are the FROZEN 512-d pre-readout activations over the whole NSD set. Collecting
  activations this way is equivalent to freezing the weights, but avoids recomputing them
  each epoch (`get_dnn_activities` E7).
- Split: first 71,000 NSD images train, last 2,000 test — a CONTIGUOUS positional split in
  NSD-id order (MI-29).
- A single Linear layer. Readout activation: none for the MPNet target, Sigmoid for the
  multi-hot target.
- Loss: cosine distance for BOTH targets — including the multi-hot one, on a sigmoid output.
  NOT BCE. This is unusual and the paper never tests the alternative (hidden assumption H5);
  we reproduce it as written. SOURCE: transfer_readout.py:108-111,
  `criterion = nn.CosineEmbeddingLoss()`.
- Hyperparameters verbatim from transfer_readout.py:25-29 — Adam, lr = 5e-2,
  eps = 1e-1 (10^7x the default; stated, not a typo), 200 epochs, batch 96, cosine-annealed.
- Metric: test cosine similarity, averaged over the 10 seeds; error bars = s.d. across seeds.

Noise floor (`N-PROBE-FLOOR`): the mean target vector over the training images, and its mean
cosine similarity to the test targets. A readout that learns nothing scores this.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from src.revision import config as C
from src.revision import embeddings as E
from src.revision import stimuli

N_TEST = 2000            # last 2,000 NSD images (MI-29: contiguous, in NSD-id order)
LR = 5e-2                # transfer_readout.py:27
ADAM_EPS = 1e-1          # transfer_readout.py:26 -- 10^7x the Adam default, as written
EPOCHS = 200             # transfer_readout.py:28
BATCH = 96               # transfer_readout.py:29
ACT_DIR = C.DERIV / "dnn_act"


def targets(kind: str) -> np.ndarray:
    if kind == "mpnet":
        return E.load("captions")                                  # (73000, 768)
    return stimuli.load_multihot().astype(np.float32)              # (73000, 91)


def train_probe(X: np.ndarray, Y: np.ndarray, kind: str, seed: int, device: str) -> float:
    """-> test cosine similarity."""
    torch.manual_seed(seed)
    Xtr = torch.as_tensor(X[:-N_TEST], device=device)
    Ytr = torch.as_tensor(Y[:-N_TEST], device=device)
    Xte = torch.as_tensor(X[-N_TEST:], device=device)
    Yte = torch.as_tensor(Y[-N_TEST:], device=device)

    layers: list[nn.Module] = [nn.Linear(X.shape[1], Y.shape[1])]
    if kind == "multihot":
        layers.append(nn.Sigmoid())     # transfer_readout.py:101 -- sigmoid, then COSINE loss
    model = nn.Sequential(*layers).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, eps=ADAM_EPS)
    n_steps = int(np.ceil(len(Xtr) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * n_steps, eta_min=0)
    crit = nn.CosineEmbeddingLoss()
    ones = torch.ones(BATCH, device=device)

    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xtr), device=device)
        for a in range(0, len(Xtr) - BATCH + 1, BATCH):
            idx = perm[a : a + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xtr[idx]), Ytr[idx], ones)
            loss.backward()
            opt.step()
            sched.step()

    with torch.no_grad():
        pred = model(Xte)
        cos = nn.functional.cosine_similarity(pred, Yte, dim=1).mean()
    return float(cos)


def noise_floor(Y: np.ndarray) -> float:
    """The mean training target, scored against the test targets. A readout that learns
    nothing achieves this. SOURCE: N-PROBE-FLOOR."""
    mu = Y[:-N_TEST].mean(0, keepdims=True)
    a = mu / (np.linalg.norm(mu) + 1e-8)
    b = Y[-N_TEST:] / (np.linalg.norm(Y[-N_TEST:], axis=1, keepdims=True) + 1e-8)
    return float((b @ a.T).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--crop", default="center")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Y = {k: targets(k) for k in ("mpnet", "multihot")}
    floors = {k: noise_floor(v) for k, v in Y.items()}

    results: dict[str, dict[str, list[float]]] = {}
    for source in ("mpnet", "multihot"):                # which net the activations come from
        for target in ("mpnet", "multihot"):            # what we try to read out of it
            key = f"read_{target}_from_{source}"
            scores = []
            for seed in args.seeds:
                p = ACT_DIR / f"{source}_seed{seed}_{args.crop}.npy"
                if not p.exists():
                    print(f"  [skip] {p.name} not extracted yet")
                    continue
                X = np.load(p)
                assert X.shape[1] == C.RCNN_PREREADOUT_DIM == 512
                scores.append(train_probe(X, Y[target], target, seed, device))
                print(f"  {key} seed{seed}: {scores[-1]:.4f}", flush=True)
            if scores:
                results[key] = {"scores": scores,
                                "mean": float(np.mean(scores)),
                                "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0}

    print("\n=== Fig. 4b — frozen-readout probe (test cosine similarity) ===")
    print(f"{'':28s}{'from LLM-trained':>20s}{'from category-trained':>24s}")
    for target, label in [("multihot", "read out CATEGORY"), ("mpnet", "read out LLM")]:
        row = f"{label:28s}"
        for source in ("mpnet", "multihot"):
            k = f"read_{target}_from_{source}"
            row += f"{results[k]['mean']:>20.4f}" if k in results else f"{'--':>20s}"
        print(row + f"   (floor {floors[target]:.3f})")

    print("\nC15 predicts an ASYMMETRY: category reads out of the LLM-trained net "
          "(matching the category-trained ceiling); LLM does NOT read out of the "
          "category-trained net.")

    out = C.REPORTS / "figures" / "fig4b_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "noise_floors": floors}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
