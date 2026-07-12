import numpy as np
import torch
import torch.nn as nn
from tabicl import TabICLClassifier

c = TabICLClassifier(device="cuda", random_state=0)
c.fit(np.random.randn(80, 6), np.random.randint(0, 2, 80))

cands = [a for a in dir(c) if not a.startswith("__")]
mod = None
for a in cands:
    try:
        v = getattr(c, a)
    except Exception:
        continue
    if isinstance(v, nn.Module):
        print("nn.Module attr:", a, "->", type(v).__name__)
        mod = v
if mod is None:
    print("no nn.Module found; attrs:", cands)
    raise SystemExit

n_all = sum(p.numel() for p in mod.parameters())
print(f"\ntotal params: {n_all/1e6:.1f}M")
lins = [(n, m.in_features, m.out_features) for n, m in mod.named_modules() if isinstance(m, nn.Linear)]
print(f"nn.Linear count: {len(lins)}")
from collections import Counter
pref = Counter(n.split(".")[0] for n, _, _ in lins)
print("top-level blocks holding Linears:", dict(pref))
for n, i, o in lins[:12]:
    print(f"  {n:<60s} {i}->{o}")
print("  ...")
for n, i, o in lins[-6:]:
    print(f"  {n:<60s} {i}->{o}")
