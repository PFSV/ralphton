"""Datasets + the acquisition protocol.

Every task is split into four disjoint parts:

    D_ctx   the in-context set the frozen TFM currently sees   (small: the low-data regime)
    D_pool  a reserve of labelled rows the agent may BUY, one credit per row
    D_val   the agent's only feedback signal (accept/reject an action)
    D_test  held out; touched exactly once, to score the final context

`semantic=True` datasets carry meaningful column names; `semantic=False` ones ship with
opaque V1..Vn headers. The split is the natural experiment: world knowledge can only
help where there is world to know about.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

N_CTX, N_POOL, N_VAL, N_TEST = 200, 800, 300, 700

# did, name, semantic?, note
REGISTRY = [
    (31,    "credit-g",   True,  "checking_status, credit_history, purpose, ..."),
    (23,    "cmc",        True,  "Wifes_age, Wifes_education, ..."),
    (40701, "churn",      True,  "state, account_length, international_plan, ..."),
    (44,    "spambase",   True,  "word_freq_make, word_freq_address, ..."),
    (1049,  "pc4",        True,  "LOC_BLANK, BRANCH_COUNT, CALL_PAIRS, ..."),
    (40994, "climate",    True,  "vconst_corr, vconst_2, ... (simulator params)"),
    (1494,  "qsar-biodeg", False, "V1..V41"),
    (1487,  "ozone",      False, "V1..V72"),
    (1489,  "phoneme",    False, "V1..V5"),
    (1462,  "banknote",   False, "V1..V4"),
    (1461,  "bank-marketing-anon", False, "V1..V16"),
    # adult (1590) is deliberately excluded from the main table: Bordt et al. (2024)
    # show GPT-class models have memorised it verbatim. It is used only as a
    # contamination probe in memcheck.py.
]


@dataclass
class Task:
    did: int
    name: str
    semantic: bool
    X_ctx: pd.DataFrame
    y_ctx: np.ndarray
    X_pool: pd.DataFrame
    y_pool: np.ndarray          # hidden from the agent until a row is bought
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    n_classes: int
    bought: list[int] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return list(self.X_ctx.columns)

    def schema_card(self, n_rows: int = 5) -> str:
        """What the agent is allowed to see about the task."""
        head = self.X_ctx.head(n_rows).to_string(index=False, max_colwidth=18)
        dtypes = "\n".join(
            f"  - {c}: {t}" for c, t in zip(self.X_ctx.columns, self.X_ctx.dtypes.astype(str))
        )
        return (
            f"dataset: {self.name}\n"
            f"target: {self.n_classes}-class classification\n"
            f"rows currently in context: {len(self.X_ctx)}\n"
            f"columns:\n{dtypes}\n\n"
            f"first {n_rows} rows:\n{head}"
        )


def anonymize(t: Task) -> Task:
    """Strip semantics: rename every column to V1..Vn. Same numbers, no world knowledge.

    This is the control that decides whether the agent's gain comes from the LLM's
    knowledge or merely from having a search loop.
    """
    ren = {c: f"V{i+1}" for i, c in enumerate(t.X_ctx.columns)}
    out = Task(**{**t.__dict__, "name": t.name + "-anon", "semantic": False, "bought": []})
    for attr in ("X_ctx", "X_pool", "X_val", "X_test"):
        setattr(out, attr, getattr(t, attr).rename(columns=ren))
    return out


def load(did: int, name: str, semantic: bool, seed: int = 0) -> Task:
    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    X, y, _, _ = d.get_data(target=d.default_target_attribute)
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    # categoricals -> integer codes; keep column NAMES (that is the semantic signal)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes.astype(float)
    X = X.astype(float).fillna(X.median(numeric_only=True))
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

    # Context is fixed at N_CTX (the low-data regime we care about). Whatever remains is
    # divided pool/val/test in fixed proportion, so small datasets shrink the reserve
    # rather than starving the test set — test AUC is the number the paper reports.
    X_ctx, y_ctx, R, yR = cut(X, y, min(N_CTX, len(X) // 3), seed)
    rest = len(R)
    n_pool = min(N_POOL, int(rest * 0.40))
    n_val = min(N_VAL, int(rest * 0.20))
    X_pool, y_pool, R, yR = cut(R, yR, n_pool, seed + 1)
    X_val, y_val, X_test, y_test = cut(R, yR, n_val, seed + 2)

    return Task(did, name, semantic, X_ctx, y_ctx, X_pool, y_pool,
                X_val, y_val, X_test, y_test, n_classes=len(classes))


def load_all(seed: int = 0) -> list[Task]:
    return [load(d, n, s, seed) for d, n, s, _ in REGISTRY]
