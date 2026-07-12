"""The frozen tabular foundation model, and the cost model the agent spends against.

Nothing here is trained. TabICLv2 is downloaded once and used purely as an in-context
learner: fit() loads a context, predict() is one forward pass. That is what makes the
agent's search affordable — evaluating a candidate context costs a forward pass, not a
training run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_CALLS = {"n": 0, "secs": 0.0}


def tfm_calls() -> dict:
    return dict(_MODEL_CALLS)


def score(X_ctx: pd.DataFrame, y_ctx: np.ndarray,
          X_eval: pd.DataFrame, y_eval: np.ndarray, seed: int = 0) -> float:
    """AUC of the frozen TFM when shown (X_ctx, y_ctx) as context. One forward pass."""
    if len(np.unique(y_ctx)) < 2:
        return 0.5
    t0 = time.time()
    clf = TabICLClassifier(device=DEVICE, random_state=seed)
    clf.fit(X_ctx.to_numpy(), y_ctx)
    p = clf.predict_proba(X_eval.to_numpy())
    _MODEL_CALLS["n"] += 1
    _MODEL_CALLS["secs"] += time.time() - t0

    present = np.unique(y_ctx)
    if len(present) == 2:
        return float(roc_auc_score(y_eval, p[:, 1]))
    # multiclass: restrict to classes the context actually contains
    P = np.zeros((len(X_eval), int(y_eval.max()) + 1))
    P[:, present.astype(int)] = p
    P = P / P.sum(1, keepdims=True).clip(1e-9)
    return float(roc_auc_score(y_eval, P, multi_class="ovr", average="macro",
                               labels=list(range(P.shape[1]))))


def proba(X_ctx: pd.DataFrame, y_ctx: np.ndarray, X_eval: pd.DataFrame, seed: int = 0) -> np.ndarray:
    clf = TabICLClassifier(device=DEVICE, random_state=seed)
    clf.fit(X_ctx.to_numpy(), y_ctx)
    _MODEL_CALLS["n"] += 1
    return clf.predict_proba(X_eval.to_numpy())


# ---------------------------------------------------------------- cost model

@dataclass(frozen=True)
class Prices:
    """Credits. One credit == one labelled row bought from the reserve.

    Everything else is priced relative to that, which is the whole point: the agent
    must decide whether a unit of budget is better spent on real data or on the
    cheaper substitutes (a derived feature, a synthetic row, a better context).
    """
    llm_call: float = 1.0        # an LLM proposal (feature code, synthesiser, plan)
    curate: float = 0.5          # compute-only reshaping of the context
    synth_setup: float = 2.0     # LLM designs a conditional sampler
    synth_per_row: float = 0.01  # simulated rows are ~100x cheaper than real ones
    acquire_per_row: float = 1.0 # a real labelled row

PRICES = Prices()


class Budget:
    def __init__(self, total: float):
        self.total = float(total)
        self.spent = 0.0
        self.ledger: list[tuple[str, float]] = []

    def can(self, c: float) -> bool:
        return self.spent + c <= self.total + 1e-9

    def charge(self, what: str, c: float) -> None:
        self.spent += c
        self.ledger.append((what, round(c, 3)))

    @property
    def left(self) -> float:
        return max(0.0, self.total - self.spent)
