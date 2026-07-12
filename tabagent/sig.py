"""Paired significance on what we actually have. One difference per (task, seed)."""
import json, itertools
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

rows = [json.loads(l) for l in open("stage1_lr1e5.jsonl") if l.strip()]
by = defaultdict(dict)
for r in rows:
    by[r["arm"]][r["seed"]] = r["test"]

arms = [a for a in ("pretrained", "base", "random", "agent") if a in by]
seeds = sorted(set.intersection(*[set(by[a]) for a in arms]))
tasks = sorted(by[arms[0]][seeds[0]])
print(f"arms {arms} | seeds {seeds} | {len(tasks)} test tasks -> n = {len(tasks)*len(seeds)} pairs\n")

def paired(a, b):
    return np.array([by[a][s][t] - by[b][s][t] for s in seeds for t in tasks])

print(f"{'comparison':<22s} {'mean d':>9s} {'95% CI':>20s} {'wilcoxon p':>11s} {'t-test p':>9s} {'wins':>8s}")
print("-" * 86)
rng = np.random.default_rng(0)
for a, b in itertools.combinations(arms, 2):
    if b == "pretrained":
        a, b = b, a
    d = paired(a, b) if a != "pretrained" else -paired(b, a)
    d = paired(a, b)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    tp = ttest_rel(np.zeros_like(d), -d).pvalue
    print(f"{a} - {b:<12s} {d.mean():>+9.5f} [{lo:>+8.5f},{hi:>+8.5f}] {wp:>11.3f} {tp:>9.3f} "
          f"{(d>0).sum():>4d}/{len(d)}")

print("\nper-arm TEST mean over seeds:")
for a in arms:
    m = [np.nanmean(list(by[a][s].values())) for s in seeds]
    print(f"  {a:<12s} {np.mean(m):.4f} ± {np.std(m):.4f}   (seeds: {np.round(m,4).tolist()})")
