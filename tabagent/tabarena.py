"""Evaluate on TabArena, not on tables we picked ourselves.

TabArena (Erickson et al., 2025) curated 51 datasets from the 1053 used across the tabular
literature, and flags which of them TabICL can actually run: 36 classification tasks. Using
their suite instead of our own selection removes the obvious objection -- that we chose the
tables that made the method look good -- and gives the prior audit an authoritative target
distribution rather than one we assembled.

DEV/TEST is split by dataset size so that neither half is systematically easier, and the
split is fixed here, in code, before any result exists.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from data import N_CTX, N_POOL, N_VAL, N_TEST, Task

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
_CANDIDATES = [
    HERE / "curated_tabarena_dataset_metadata.csv",              # vendored copy
    HERE / "tabarena-src" / "packages" / "tabarena" / "src" / "tabarena" / "benchmark"
         / "task" / "metadata" / "data" / "curated_tabarena_dataset_metadata.csv",
]
META = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])


# A few TabArena tables cannot be materialised here: OpenML serves a malformed ARFF for
# customer_satisfaction_in_airline (bad @DATA row) and its parquet mirror 404s. Excluding
# them is fine, but it MUST be deterministic -- every arm has to be scored on exactly the
# same tasks -- so the usable list is computed once by probe_suite() and pinned to disk.
USABLE = HERE / "tabarena_usable.json"


def suite(usable_only: bool = True) -> pd.DataFrame:
    if not META.exists():
        raise FileNotFoundError(
            f"TabArena metadata not found. Looked in: {[str(p) for p in _CANDIDATES]}"
        )
    d = pd.read_csv(META)
    c = d[d.is_classification & d.can_run_tabicl].copy()
    if usable_only and USABLE.exists():
        import json
        ok = set(json.loads(USABLE.read_text())["usable_task_ids"])
        c = c[c.task_id.isin(ok)]
    return c.sort_values("num_instances").reset_index(drop=True)


def split_dev_test():
    """Interleave by size so DEV and TEST cover the same size range. DEV gets every 3rd."""
    c = suite()
    dev = c.iloc[::3]
    test = c.drop(dev.index)
    return dev.reset_index(drop=True), test.reset_index(drop=True)


def probe_suite():
    """Load every candidate once, sequentially, and pin the ones that work."""
    import json
    c = suite(usable_only=False)
    ok, bad = [], []
    for r in c.itertuples():
        try:
            load_task(r, seed=0)
            ok.append(int(r.task_id))
        except Exception as e:
            bad.append((r.dataset_name, type(e).__name__, str(e)[:70]))
            print(f"  DROP {r.dataset_name:<45s} {type(e).__name__}: {str(e)[:60]}", flush=True)
    USABLE.write_text(json.dumps({"usable_task_ids": ok,
                                  "dropped": [b[0] for b in bad]}, indent=1))
    print(f"\nusable {len(ok)}/{len(c)} TabArena tasks; dropped {[b[0] for b in bad]}")
    return ok, bad


def load_task(row, seed: int = 0) -> Task:
    """OpenML *task* id -> the same ctx/pool/val/test protocol as data.load()."""
    t = openml.tasks.get_task(int(row.task_id), download_splits=False)
    ds = t.get_dataset()
    X, y, cat, _ = ds.get_data(target=t.target_name)
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes.astype(float)
    X = X.astype(float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    classes = sorted(y.astype(str).unique())
    y = y.astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()

    need = N_CTX + N_POOL + N_VAL + N_TEST
    if len(X) > need:
        X, _, y, _ = train_test_split(X, y, train_size=need, random_state=seed, stratify=y)
        X = X.reset_index(drop=True)

    def cut(Xa, ya, n, s):
        n = int(np.clip(n, 1, len(Xa) - 1))
        a, b, ya_, yb_ = train_test_split(Xa, ya, train_size=n, random_state=s, stratify=ya)
        return a.reset_index(drop=True), ya_, b.reset_index(drop=True), yb_

    X_ctx, y_ctx, R, yR = cut(X, y, min(N_CTX, len(X) // 3), seed)
    rest = len(R)
    X_pool, y_pool, R, yR = cut(R, yR, min(N_POOL, int(rest * 0.40)), seed + 1)
    X_val, y_val, X_test, y_test = cut(R, yR, min(N_VAL, int(len(R) * 0.33)), seed + 2)

    return Task(int(row.task_id), str(row.dataset_name), True,
                X_ctx, y_ctx, X_pool, y_pool, X_val, y_val, X_test, y_test,
                n_classes=len(classes))


CACHE = HERE / "cache" / "tasks"


def load_split(seed: int = 0):
    """Build the 36 TabArena tasks once, then hand every process the pickle.

    Downloading and ARFF-parsing 36 tables takes minutes; doing it independently inside each
    of the twelve arm-runs was most of the wall clock. The split is a pure function of
    (suite, seed), so it caches perfectly.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"tabarena_seed{seed}.pkl"
    if f.exists():
        import pickle
        with f.open("rb") as fh:
            return pickle.load(fh)

    dev_rows, test_rows = split_dev_test()
    dev = [load_task(r, seed) for r in dev_rows.itertuples()]
    test = [load_task(r, seed) for r in test_rows.itertuples()]

    import pickle
    tmp = f.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump((dev, test), fh)
    tmp.replace(f)                       # atomic: concurrent readers never see a partial file
    return dev, test


def real_stats() -> dict:
    """The target distribution, straight from TabArena's own metadata — no downloads,
    no sampling, nothing we chose."""
    c = suite()
    return {
        "n_features": float(c.num_features.median()),
        "n_rows": float(c.num_instances.median()),
        "n_classes": float(c.num_classes.median()),
        "cat_frac": float(c.percentage_cat_features.median() / 100.0),
    }


if __name__ == "__main__":
    dev_rows, test_rows = split_dev_test()
    print(f"TabArena classification suite runnable by TabICL: {len(suite())} tasks")
    print(f"  DEV  {len(dev_rows):2d}: {', '.join(dev_rows.dataset_name)}")
    print(f"  TEST {len(test_rows):2d}: {', '.join(test_rows.dataset_name)}")
    print("\ntarget distribution (TabArena metadata):")
    for k, v in real_stats().items():
        print(f"  {k:<12s} {v:.2f}")
