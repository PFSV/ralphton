"""Measure the gap the agent claims to close.

The AUC table shows that revising the prior helps. It does not show *why*. This does: it
samples datasets from a prior, samples real OpenML tables, and measures both on the same
axes. If TabICL's released prior is mismatched to real tabular data, it should be visible
here — before any agent, any LoRA, any AUC.

Runs on CPU; the GPU is busy with the grid.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import data
import priortrain as pt

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent


def describe(X: np.ndarray, y: np.ndarray) -> dict:
    """Axes on which a synthetic table and a real table are commensurable."""
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    _, counts = np.unique(y, return_counts=True)
    k = len(counts)
    # a column is "categorical-like" if it takes few distinct values
    card = np.array([len(np.unique(X[:, j])) for j in range(m)])
    with np.errstate(all="ignore"):
        C = np.corrcoef(X.T) if m > 1 else np.array([[1.0]])
        C = np.nan_to_num(C)
        offdiag = C[~np.eye(m, dtype=bool)] if m > 1 else np.array([0.0])
        kurt = np.nan_to_num(stats.kurtosis(X, axis=0, fisher=True))
    return dict(
        n_rows=n,
        n_features=m,
        n_classes=k,
        minority=float(counts.min() / counts.sum()),      # class imbalance
        cat_frac=float((card <= 10).mean()),              # low-cardinality columns
        corr=float(np.abs(offdiag).mean()),               # feature redundancy
        kurtosis=float(np.median(kurt)),                  # tail heaviness
    )


AXES = ["n_rows", "n_features", "n_classes", "minority", "cat_frac", "corr", "kurtosis"]


def sample_prior(cfg: dict, n_datasets: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    bs = 4
    prior = pt.make_prior(cfg, batch_size=bs, n_jobs=8)
    it = iter(prior)
    while len(rows) < n_datasets:
        X, y, d, seq_len, train_size = next(it)
        Xn, yn = X.numpy(), y.numpy()
        for b in range(Xn.shape[0]):
            xb = Xn[b]
            # columns are zero-padded up to max_features; keep the real ones
            keep = ~np.all(np.abs(xb) < 1e-12, axis=0)
            rows.append(describe(xb[:, keep], yn[b]))
            if len(rows) >= n_datasets:
                break
    return pd.DataFrame(rows)


def sample_real() -> pd.DataFrame:
    rows = []
    for did, name, sem, _ in data.REGISTRY:
        t = data.load(did, name, sem, seed=0)
        X = pd.concat([t.X_ctx, t.X_pool, t.X_val, t.X_test], ignore_index=True)
        y = np.concatenate([t.y_ctx, t.y_pool, t.y_val, t.y_test])
        r = describe(X.to_numpy(), y)
        r["name"] = name
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    n_ds = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    real = sample_real()
    print(f"real tables (n={len(real)}):")
    print(real[["name"] + AXES].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    cfgs = {"TabICL default prior": dict(pt.BASE_CFG)}
    f = HERE / "stage1.jsonl"
    if f.exists():
        best = [json.loads(l) for l in f.read_text().splitlines()
                if l.strip() and json.loads(l).get("arm") == "agent"]
        if best:
            top = max(best, key=lambda r: r["best_dev_mean"])
            cfgs["agent-revised prior"] = top["best_cfg"]
            print(f"\nusing agent config from seed {top['seed']} "
                  f"(dev {top['best_dev_mean']:.4f})")

    out = {}
    for label, cfg in cfgs.items():
        df = sample_prior(cfg, n_ds)
        out[label] = df
        print(f"\n{label}  (n={len(df)} synthetic datasets)")
        print(df[AXES].describe().loc[["mean", "50%"]].to_string(
            float_format=lambda v: f"{v:.2f}"))

    # the gap, per axis: standardised distance between prior median and real median
    print("\n" + "=" * 78)
    print("MISMATCH vs real tables (|prior median - real median| / real IQR; lower = closer)")
    print("=" * 78)
    hdr = f"{'axis':<12s}{'real med':>10s}" + "".join(f"{k[:18]:>22s}" for k in out)
    print(hdr)
    tot = {k: 0.0 for k in out}
    for ax in AXES:
        rmed = real[ax].median()
        iqr = max(real[ax].quantile(0.75) - real[ax].quantile(0.25), 1e-6)
        line = f"{ax:<12s}{rmed:>10.2f}"
        for k, df in out.items():
            gap = abs(df[ax].median() - rmed) / iqr
            tot[k] += gap
            line += f"{df[ax].median():>12.2f} ({gap:>4.1f})"
        print(line)
    print("-" * 78)
    print(f"{'TOTAL':<12s}{'':>10s}" + "".join(f"{tot[k]:>22.1f}" for k in out))

    res = {"real": real.drop(columns=["name"]).median().to_dict(),
           "priors": {k: df.median().to_dict() for k, df in out.items()},
           "mismatch": tot}
    (HERE / "prior_audit.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {HERE/'prior_audit.json'}")


if __name__ == "__main__":
    main()
