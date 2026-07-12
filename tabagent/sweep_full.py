"""Full fine-tuning vs LoRA, clean GPU. Selection on DEV only; TEST printed for the record."""
import json, time
from pathlib import Path
import numpy as np
import priortrain as pt
from stage1 import DEV_TASKS, TEST_TASKS, _load, evaluate

SEED = 0
GRID = [
    dict(adapter="full", lr=1e-5, steps=200),
    dict(adapter="full", lr=1e-6, steps=200),
    dict(adapter="full", lr=1e-6, steps=600),
    dict(adapter="full", lr=3e-6, steps=600),
    dict(adapter="lora", lr=1e-5, steps=600),   # best LoRA, as the anchor
]
dev, test = _load(DEV_TASKS, SEED), _load(TEST_TASKS, SEED)
bd = float(np.mean(list(evaluate(None, dev, SEED).values())))
bt = float(np.mean(list(evaluate(None, test, SEED).values())))
print(f"{'released checkpoint':<40s} DEV {bd:.4f}   TEST {bt:.4f}\n", flush=True)
print(f"{'setting':<40s} {'DEV':>8s} {'dDEV':>8s} {'TEST':>8s} {'dTEST':>8s} {'train%':>7s}", flush=True)
rows = []
for g in GRID:
    clf = pt.load_backbone(seed=SEED)
    clf, st = pt.train_lora(clf, pt.BASE_CFG, steps=g["steps"], batch_size=4, lr=g["lr"],
                            seed=SEED, log_every=0, adapter=g["adapter"])
    state = pt.lora_state(clf.model_)
    dm = float(np.mean(list(evaluate(state, dev, SEED, g["adapter"]).values())))
    tm = float(np.mean(list(evaluate(state, test, SEED, g["adapter"]).values())))
    tag = f"{g['adapter']:<5s} lr={g['lr']:<7.0e} steps={g['steps']:<4d}"
    print(f"{tag:<40s} {dm:>8.4f} {dm-bd:>+8.4f} {tm:>8.4f} {tm-bt:>+8.4f} "
          f"{100*st.trainable/st.total:>6.1f}%", flush=True)
    rows.append(dict(**g, dev=dm, test=tm, d_dev=dm-bd, d_test=tm-bt,
                     trainable_pct=100*st.trainable/st.total))
    del clf, state
Path("sweep_full.json").write_text(json.dumps(dict(base_dev=bd, base_test=bt, rows=rows), indent=1))
best = max(rows, key=lambda r: r["dev"])
print(f"\nDEV picks: {best['adapter']} lr={best['lr']:.0e} steps={best['steps']} "
      f"(dDEV {best['d_dev']:+.4f})  [its TEST {best['d_test']:+.4f}]")
