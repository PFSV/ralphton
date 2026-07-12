"""Is the prior's distribution actually the same as real tabular data? Test it properly.

The first audit compared a handful of medians and called the gap "4.0". That is not a test
of distributional fit, and if the downstream deltas turn out small it is exactly the kind of
weak instrument that would let us shrug and call the result null.

So: characterise every dataset -- prior-sampled and real -- by the meta-features a tabular
foundation model actually consumes (marginal shape, dependence structure, and the *form* of
the decision boundary), then run a proper two-sample test between the two populations.

    C2ST (classifier two-sample test): train a classifier to tell "this dataset came from the
    prior" from "this dataset is real". AUC 0.5 => the prior is indistinguishable from
    reality. AUC 1.0 => it is a different world. Significance by permutation.

The payoff is not the p-value. It is the feature importance: whatever the classifier used to
tell them apart IS the list of things wrong with the prior, ranked. That list is what the
agent gets to act on.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent


# ────────────────────────────────────────── meta-features a TFM actually cares about

def meta(X: np.ndarray, y: np.ndarray) -> dict:
    X = np.asarray(X, float)
    n, m = X.shape
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()

    card = np.array([len(np.unique(X[:, j])) for j in range(m)])
    Z = StandardScaler().fit_transform(X)

    with np.errstate(all="ignore"):
        skew = np.nan_to_num(stats.skew(X, axis=0))
        kurt = np.nan_to_num(stats.kurtosis(X, axis=0))
        C = np.nan_to_num(np.corrcoef(Z.T)) if m > 1 else np.eye(1)
        off = C[~np.eye(m, dtype=bool)] if m > 1 else np.array([0.0])
        ev = np.linalg.eigvalsh(C + 1e-9 * np.eye(m))
        ev = np.clip(ev, 1e-9, None)
        # participation ratio: how many columns are "really" independent
        eff_rank = float((ev.sum() ** 2) / (ev**2).sum()) / m

    # what SHAPE is the decision boundary? A TFM is learning that shape, not the numbers.
    lin, tree = 0.5, 0.5
    try:
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        pl = cross_val_predict(LogisticRegression(max_iter=400), Z, y, cv=cv,
                               method="predict_proba")
        pt_ = cross_val_predict(DecisionTreeClassifier(max_depth=4, random_state=0), Z, y,
                                cv=cv, method="predict_proba")
        if len(p) == 2:
            lin, tree = roc_auc_score(y, pl[:, 1]), roc_auc_score(y, pt_[:, 1])
        else:
            lin = roc_auc_score(y, pl, multi_class="ovr", average="macro")
            tree = roc_auc_score(y, pt_, multi_class="ovr", average="macro")
    except Exception:
        pass

    try:
        mi = float(mutual_info_classif(X, y, random_state=0).mean())
    except Exception:
        mi = 0.0

    return {
        # shape of the task
        "n_rows": float(n),
        "n_features": float(m),
        "rows_per_feature": float(n / max(m, 1)),
        "n_classes": float(len(p)),
        "minority": float(p.min()),
        "class_entropy": float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12)),
        # marginals
        "abs_skew": float(np.abs(skew).mean()),
        "abs_kurtosis": float(np.abs(kurt).mean()),
        "frac_heavy_tailed": float((np.abs(kurt) > 3).mean()),
        "frac_outliers": float((np.abs(Z) > 3).mean()),
        "frac_categorical": float((card <= 10).mean()),
        "frac_binary_cols": float((card <= 2).mean()),
        "median_cardinality": float(np.median(card) / max(n, 1)),
        # dependence
        "mean_abs_corr": float(np.abs(off).mean()),
        "max_abs_corr": float(np.abs(off).max()) if m > 1 else 0.0,
        "effective_rank": eff_rank,
        # signal: how much, and of what shape
        "mutual_info": mi,
        "auc_linear": float(lin),
        "auc_tree": float(tree),
        "nonlinearity": float(tree - lin),     # >0 => boundary needs a tree, not a plane
        "difficulty": float(1.0 - max(lin, tree)),
    }


KEYS = list(meta(np.random.randn(60, 4), np.random.randint(0, 2, 60)).keys())


# ─────────────────────────────────────────────────────────── the two-sample test

def c2st(prior: pd.DataFrame, real: pd.DataFrame, n_perm: int = 200, seed: int = 0):
    """Can a classifier tell prior-sampled datasets from real ones? If yes, what gave it away?"""
    Xf = pd.concat([prior[KEYS], real[KEYS]], ignore_index=True).to_numpy()
    yf = np.r_[np.ones(len(prior)), np.zeros(len(real))]
    Xf = np.nan_to_num(StandardScaler().fit_transform(Xf))

    def auc_of(labels):
        clf = RandomForestClassifier(400, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        k = min(5, int(min(np.bincount(labels.astype(int)))))
        if k < 2:
            return 0.5
        pr = cross_val_predict(clf, Xf, labels, method="predict_proba",
                               cv=StratifiedKFold(k, shuffle=True, random_state=seed))[:, 1]
        return float(roc_auc_score(labels, pr))

    observed = auc_of(yf)
    rng = np.random.default_rng(seed)
    null = np.array([auc_of(rng.permutation(yf)) for _ in range(n_perm)])
    p = float((null >= observed).mean())

    imp = RandomForestClassifier(400, min_samples_leaf=2, random_state=seed,
                                 n_jobs=-1).fit(Xf, yf).feature_importances_
    return observed, p, dict(sorted(zip(KEYS, imp), key=lambda kv: -kv[1]))


def per_axis(prior: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Per meta-feature: how far apart are the two populations, and which way?"""
    rows = []
    for k in KEYS:
        a, b = prior[k].to_numpy(), real[k].to_numpy()
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        ks, ksp = stats.ks_2samp(a, b)
        pooled = np.sqrt((a.var() + b.var()) / 2) or 1e-9
        rows.append(dict(axis=k, prior=np.median(a), real=np.median(b),
                         cohens_d=(np.median(a) - np.median(b)) / pooled,
                         ks=ks, p=ksp))
    d = pd.DataFrame(rows)
    d["sig"] = d.p < 0.05
    return d.sort_values("ks", ascending=False).reset_index(drop=True)


# ───────────────────────────────────────────────────────────────────── drivers

def sample_prior_meta(cfg: dict, n: int = 40, seed: int = 0) -> pd.DataFrame:
    import priortrain as pt
    rows, it = [], iter(pt.make_prior(cfg, batch_size=4))
    while len(rows) < n:
        X, y, *_ = next(it)
        Xn, yn = X.cpu().numpy(), y.cpu().numpy()
        for b in range(Xn.shape[0]):
            xb = Xn[b]
            keep = ~np.all(np.abs(xb) < 1e-12, axis=0)
            if keep.sum() >= 2 and len(np.unique(yn[b])) >= 2:
                rows.append(meta(xb[:, keep], yn[b]))
            if len(rows) >= n:
                break
    return pd.DataFrame(rows)


def real_meta(seed: int = 0) -> pd.DataFrame:
    """Real TabArena tables, described exactly as the prior samples are.

    Cached: this runs cross-validated logistic + tree fits on all 36 tables, which costs
    minutes of CPU, and it is a pure function of the seed. Every agent round was paying for
    it again."""
    import tabarena
    f = HERE / "cache" / f"real_meta_{seed}.parquet"
    if f.exists():
        return pd.read_parquet(f)

    dev, test = tabarena.load_split(seed)
    rows = []
    for t in dev + test:
        X = pd.concat([t.X_ctx, t.X_pool, t.X_val, t.X_test], ignore_index=True).to_numpy()
        y = np.concatenate([t.y_ctx, t.y_pool, t.y_val, t.y_test])
        r = meta(X, y)
        r["name"] = t.name
        rows.append(r)
    df = pd.DataFrame(rows)
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f)
    return df


def report(cfg: dict, label: str, real: pd.DataFrame, n: int = 40, seed: int = 0) -> dict:
    prior = sample_prior_meta(cfg, n=n, seed=seed)
    auc, p, imp = c2st(prior, real, seed=seed)
    axes = per_axis(prior, real)

    print(f"\n{'='*82}\n{label}\n{'='*82}")
    print(f"C2ST: a classifier separates prior from real with AUC = {auc:.3f} "
          f"(permutation p = {p:.3f})")
    verdict = ("indistinguishable from real data" if auc < 0.6 else
               "clearly a different distribution" if auc > 0.85 else
               "distinguishable, but overlapping")
    print(f"  -> the prior is {verdict}.\n")

    print("what gives it away (the ranked list of things wrong with the prior):")
    for k, v in list(imp.items())[:8]:
        row = axes[axes.axis == k]
        if row.empty:
            continue
        r = row.iloc[0]
        arrow = "TOO HIGH" if r.cohens_d > 0 else "TOO LOW "
        star = "*" if r.sig else " "
        print(f"  {v:5.3f}  {k:<20s} prior {r.prior:9.3f} vs real {r.real:9.3f}   "
              f"{arrow} (d={r.cohens_d:+.2f}, KS={r.ks:.2f}){star}")
    print("  (* = KS two-sample test significant at 0.05)")
    return dict(label=label, c2st_auc=auc, c2st_p=p, importance=imp,
                axes=axes.to_dict("records"))


if __name__ == "__main__":
    import priortrain as pt

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    real = real_meta(seed)
    print(f"real TabArena tables: {len(real)}")

    out = [report(pt.BASE_CFG, "TabICL's released prior  vs  real TabArena tables", real,
                  seed=seed)]

    # if the agent has already produced a revised prior, test that too
    f = HERE / f"pipeline_{seed}.json"
    if f.exists():
        st = json.loads(f.read_text())
        h = st.get("history", [])
        if h:
            best = max(h, key=lambda x: x["dev_mean"])
            out.append(report(best["cfg"],
                              f"agent-revised prior (round {best['round']})  vs  real tables",
                              real, seed=seed))
            print(f"\nC2ST AUC: released {out[0]['c2st_auc']:.3f} -> "
                  f"agent-revised {out[1]['c2st_auc']:.3f}  "
                  f"({'closer to real' if out[1]['c2st_auc'] < out[0]['c2st_auc'] else 'no closer'})")

    (HERE / f"analysis2_{seed}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote analysis2_{seed}.json")
