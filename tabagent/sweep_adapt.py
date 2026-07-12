"""Can adapting the released checkpoint to its OWN prior ever beat leaving it alone?

Every arm so far lost to the untouched checkpoint on held-out tasks. Before blaming the
prior, blame the optimiser: 200 steps at lr 1e-4 may simply be damaging weights that were
already trained on this exact prior for far longer.

So sweep the adaptation setup itself -- LoRA vs full fine-tuning, learning rate, steps --
holding the prior fixed at TabICL's own. If nothing here recovers the baseline, the problem
is adaptation, not the agent.

SELECTION IS ON DEV. Test scores are printed for the record but must not choose anything.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import priortrain as pt
from stage1 import DEV_TASKS, TEST_TASKS, _load, evaluate

SEED = 0
OUT = Path(__file__).parent / "sweep_adapt.jsonl"

GRID = [
    dict(adapter="lora", lr=1e-4, steps=200),   # what the grid has been running
    dict(adapter="lora", lr=1e-5, steps=200),
    dict(adapter="lora", lr=3e-5, steps=600),
    dict(adapter="lora", lr=1e-5, steps=600),
    dict(adapter="full", lr=1e-5, steps=200),
    dict(adapter="full", lr=1e-6, steps=200),
    dict(adapter="full", lr=1e-6, steps=600),
    dict(adapter="full", lr=3e-6, steps=600),
]

dev, test = _load(DEV_TASKS, SEED), _load(TEST_TASKS, SEED)

base_dev = evaluate(None, dev, SEED)
base_test = evaluate(None, test, SEED)
bd, bt = np.mean(list(base_dev.values())), np.mean(list(base_test.values()))
print(f"{'released checkpoint (no adaptation)':<44s} DEV {bd:.4f}   TEST {bt:.4f}\n", flush=True)
print(f"{'setting':<44s} {'DEV':>8s} {'dDEV':>8s} {'TEST':>8s} {'dTEST':>8s} {'loss':>7s} {'s':>5s}",
      flush=True)

rows = []
for g in GRID:
    t0 = time.time()
    clf = pt.load_backbone(seed=SEED)
    clf, st = pt.train_lora(clf, pt.BASE_CFG, steps=g["steps"], batch_size=4, lr=g["lr"],
                            seed=SEED, log_every=0, adapter=g["adapter"])
    state = pt.lora_state(clf.model_)
    d = evaluate(state, dev, SEED)
    t = evaluate(state, test, SEED)
    dm, tm = float(np.mean(list(d.values()))), float(np.mean(list(t.values())))
    tag = f"{g['adapter']:<5s} lr={g['lr']:<7.0e} steps={g['steps']:<4d}"
    print(f"{tag:<44s} {dm:>8.4f} {dm-bd:>+8.4f} {tm:>8.4f} {tm-bt:>+8.4f} "
          f"{st.final_loss:>7.3f} {time.time()-t0:>5.0f}", flush=True)
    rows.append(dict(**g, dev_mean=dm, test_mean=tm, d_dev=dm - bd, d_test=tm - bt,
                     loss=st.final_loss, dev=d, test=t))
    del clf, state

with OUT.open("w") as f:
    f.write(json.dumps(dict(base_dev=bd, base_test=bt, rows=rows)) + "\n")

best = max(rows, key=lambda r: r["dev_mean"])          # DEV picks. Not TEST.
print(f"\nDEV-selected setting: {best['adapter']} lr={best['lr']:.0e} steps={best['steps']}"
      f"  (DEV {best['dev_mean']:.4f}, dDEV {best['d_dev']:+.4f})")
print(f"  its TEST, for the record: {best['test_mean']:.4f} ({best['d_test']:+.4f})")
