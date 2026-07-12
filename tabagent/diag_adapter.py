"""Is the adapter inert, and if so, why?

pretrained, base and random all landed on the same test AUC to four decimals. Two very
different explanations:

  (a) the adapter never moves. LoRA's B is zero-initialised, so at lr=1e-5 over 300 steps it
      may simply stay at zero and the model is literally unchanged.
  (b) the adapter moves, but adapting to a different prior genuinely changes nothing.

These need completely different fixes, so measure rather than guess. For each setting we
report ||B|| (did the adapter leave its zero init?) and the mean absolute change in predicted
probability on a real task (did the model's behaviour change?) alongside DEV AUC.

Crucially this is run on a prior that is DELIBERATELY DIFFERENT from TabICL's own -- small
binary categorical-heavy tables, i.e. what real data actually looks like. Adapting to the
base prior is expected to do nothing: the model was already trained on it. That expectation
is not evidence of a bug, and testing on it would tell us nothing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import priortrain as pt
import tabarena
from stage1 import evaluate

SEED = 0

# What real tables look like, per TabArena: ~22 features, binary, ~40% categorical, small.
REALISTIC = dict(pt.BASE_CFG, max_features=40, min_features=4, max_classes=2,
                 max_seq_len=800, min_seq_len=300, cat_prob=0.5, replay_small=1,
                 num_causes_max=20, num_layers_max=4)

GRID = [
    dict(lr=1e-5, steps=300,  alpha=32),   # what the grid ran
    dict(lr=1e-5, steps=1500, alpha=32),
    dict(lr=3e-5, steps=1500, alpha=32),
    dict(lr=1e-4, steps=1500, alpha=32),
    dict(lr=3e-5, steps=1500, alpha=128),  # bigger adapter scale
    dict(lr=1e-4, steps=300,  alpha=32),   # the destructive setting, for reference
]


def adapter_norm(model) -> float:
    """How far has LoRA's zero-initialised B travelled? 0.0 == the adapter is the identity."""
    return float(sum(p.detach().float().norm() ** 2
                     for n, p in model.named_parameters()
                     if p.requires_grad and n.endswith(".B")) ** 0.5)


def pred_drift(state, task, seed, alpha) -> float:
    """Mean |change in predicted probability| on a real task. This is what actually matters:
    an adapter with a big norm that does not move predictions is still inert."""
    p0 = pt.LoRATabICL(None, seed=seed).fit(
        task.X_ctx.to_numpy(), task.y_ctx).predict_proba(task.X_val.to_numpy())
    p1 = pt.LoRATabICL(state, seed=seed, alpha=alpha).fit(
        task.X_ctx.to_numpy(), task.y_ctx).predict_proba(task.X_val.to_numpy())
    k = min(p0.shape[1], p1.shape[1])
    return float(np.abs(p0[:, :k] - p1[:, :k]).mean())


def main():
    dev, _ = tabarena.load_split(SEED)
    probe = dev[0]
    base = evaluate(None, dev, SEED)
    bd = float(np.mean(list(base.values())))
    print(f"released checkpoint: DEV {bd:.4f}\n", flush=True)
    print(f"prior under test = REALISTIC (40 feat max, binary, 50% categorical, small)\n")
    print(f"{'lr':>7s} {'steps':>6s} {'alpha':>6s} | {'||B||':>8s} {'|dp|':>8s} "
          f"{'DEV':>8s} {'dDEV':>8s} {'loss':>7s} {'s':>5s}", flush=True)

    rows = []
    for g in GRID:
        t0 = time.time()
        clf = pt.load_backbone(seed=SEED)
        clf, st = pt.train_lora(clf, REALISTIC, steps=g["steps"], batch_size=4,
                                lr=g["lr"], seed=SEED, log_every=0, adapter="lora",
                                r=16, alpha=g["alpha"])
        nb = adapter_norm(clf.model_)
        state = pt.lora_state(clf.model_)
        drift = pred_drift(state, probe, SEED, g["alpha"])
        sc = evaluate(state, dev, SEED)
        dm = float(np.mean(list(sc.values())))
        print(f"{g['lr']:>7.0e} {g['steps']:>6d} {g['alpha']:>6d} | {nb:>8.3f} {drift:>8.5f} "
              f"{dm:>8.4f} {dm-bd:>+8.4f} {st.final_loss:>7.3f} {time.time()-t0:>5.0f}",
              flush=True)
        rows.append(dict(**g, B_norm=nb, pred_drift=drift, dev=dm, d_dev=dm - bd,
                         loss=st.final_loss))
        del clf, state
        torch.cuda.empty_cache()

    Path("diag_adapter.json").write_text(json.dumps(dict(base_dev=bd, rows=rows), indent=1))
    inert = [r for r in rows if r["pred_drift"] < 1e-3]
    print(f"\n{len(inert)}/{len(rows)} settings are INERT (predictions move < 0.001).")
    live = [r for r in rows if r["pred_drift"] >= 1e-3]
    if live:
        best = max(live, key=lambda r: r["d_dev"])
        print(f"best setting that actually moves the model: lr={best['lr']:.0e} "
              f"steps={best['steps']} alpha={best['alpha']}  dDEV {best['d_dev']:+.4f} "
              f"(drift {best['pred_drift']:.4f})")
    else:
        print("NOTHING moves the model. The adapter surface (FFN of icl_predictor) may be "
              "too narrow -- widen it before drawing any conclusion about the agent.")


if __name__ == "__main__":
    main()
