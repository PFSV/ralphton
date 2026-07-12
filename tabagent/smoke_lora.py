import time

import numpy as np
from sklearn.metrics import roc_auc_score

import data
import priortrain as pt

print("device:", pt.DEV, flush=True)

prior = pt.make_prior(pt.BASE_CFG, batch_size=4, n_jobs=8)
t0 = time.time()
X, y, d, seq_len, train_size = next(iter(prior))
print(f"prior batch: X{tuple(X.shape)} y{tuple(y.shape)} ({time.time()-t0:.1f}s)", flush=True)

TASKS = [(31, "credit-g", True), (23, "cmc", True), (1489, "phoneme", False)]
tasks = [data.load(d_, n, s, seed=0) for d_, n, s in TASKS]


def ev(state, label):
    out = []
    for t in tasks:
        m = pt.LoRATabICL(state, seed=0).fit(t.X_ctx.to_numpy(), t.y_ctx)
        p = m.predict_proba(t.X_test.to_numpy())
        if t.n_classes == 2:
            a = roc_auc_score(t.y_test, p[:, 1])
        else:
            a = roc_auc_score(t.y_test, p, multi_class="ovr", average="macro")
        out.append(a)
    print(f"{label:<22s} " + "  ".join(f"{t.name}={a:.4f}" for t, a in zip(tasks, out))
          + f"  | mean={np.mean(out):.4f}", flush=True)
    return np.array(out)


base = ev(None, "pretrained (no LoRA)")

clf = pt.load_backbone(seed=0)
clf, st = pt.train_lora(clf, pt.BASE_CFG, steps=100, batch_size=4, seed=0, log_every=25)
state = pt.lora_state(clf.model_)
print(f"trainable {st.trainable/1e3:.0f}K / {st.total/1e6:.1f}M "
      f"({100*st.trainable/st.total:.2f}%) | loss {st.final_loss:.4f} | {st.secs}s "
      f"| adapter keys {len(state)}", flush=True)

after = ev(state, "LoRA on base prior")
print(f"\ndelta per task: {np.round(after - base, 4)}  | mean delta {np.mean(after-base):+.4f}")
