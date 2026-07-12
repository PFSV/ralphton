"""The LLM stops tuning knobs and starts writing the generator.

The C2ST said the released prior is perfectly separable from real tables (AUC 1.000), and
named the reasons: real columns are skewed (2.20 vs 0.18), heavy-tailed (47% of columns vs
0%), and discrete (normalised cardinality 0.008 vs 1.000). None of those three are reachable
from the 19 configuration knobs -- `sampling` offers normal/mixed/uniform and nothing else.
The agent was being asked to fix things it had no lever for.

So give it a lever. The SCM still draws the causal structure -- that is the part of TabPFN's
prior worth keeping. On top of it, the LLM writes a *realisation* function that rewrites the
marginals: exponentiate a column into a log-normal income, discretise one into a category,
cube one into a heavy tail. Structure from the program, shape from the model.

It writes that function from the diagnosis alone -- the C2ST numbers -- and never sees a real
row. Nothing memorised can leak into the prior.

    audit ─► C2ST diagnosis ─► LLM writes realize() ─► prior' ─► re-audit ─► LoRA ─► TabArena
                    ▲                                     │
                    └─────────────────────────────────────┘
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

import llm

HERE = Path(__file__).parent

# Reject what is actually dangerous — filesystem, network, arbitrary execution — and nothing
# else. An earlier filter also banned the substrings `import` and `__`, which threw away three
# of five rounds because the model had written `import numpy as np` at the top (harmless, and
# np is already in scope). Over-strict screening does not make the loop safer; it makes it not
# a loop.
DANGEROUS = re.compile(
    r"\b(exec|eval|compile|open|input|globals|locals|getattr|setattr|delattr"
    r"|subprocess|socket|requests|urllib|shutil|pathlib|pickle)\b"
    r"|\b(os|sys|io)\.|__[a-z]+__|while\s+True"
)
IMPORT_LINE = re.compile(r"^\s*(import|from)\s+.*$", re.M)


def _sanitize(code: str) -> tuple[str, str]:
    """Strip import lines (numpy is already provided), then screen what remains."""
    if not code:
        return "", "empty"
    code = IMPORT_LINE.sub("", code)
    hit = DANGEROUS.search(code)
    if hit:
        return "", f"dangerous construct: {hit.group(0)!r}"
    return code, "ok"


# ────────────────────────────────────────────────────────── the LLM writes the generator

def write_realizer(diagnosis: str, previous: str | None = None,
                   feedback: str | None = None) -> tuple[str, str]:
    """Ask for a function that reshapes SCM output into something that looks real."""
    retry = ""
    if previous:
        retry = f"""
YOUR PREVIOUS ATTEMPT
{previous}

WHAT HAPPENED
{feedback}

Fix it. Keep what worked; change what did not."""

    prompt = f"""A tabular foundation model is pre-trained on synthetic datasets drawn from a structural
causal model (SCM). The SCM gets the *causal structure* right, but the numbers it emits do
not look like real tabular data at all. Here is the measured evidence -- a classifier
separates synthetic from real tables perfectly, and this is what gave it away:

{diagnosis}

Write a function that post-processes each synthetic dataset so its COLUMNS look like real
tabular columns, WITHOUT destroying the causal relationships the SCM built.

def realize(X, y, rng):
    # X: float32 numpy array (n_rows, n_cols) straight from the SCM
    # y: int   numpy array (n_rows,) -- the labels. DO NOT CHANGE y.
    # rng: numpy Generator
    # return: a float array of the SAME shape as X
    ...

What real tabular columns actually are, and the SCM emits none of them:
  - heavily right-skewed and heavy-tailed (income, price, counts, durations)
  - discrete / low-cardinality (categories, codes, ratings, binary flags, small integers)
  - bounded or censored (ages, percentages, counts >= 0)
  - occasionally near-duplicated or redundant with another column

Rules:
  - Apply DIFFERENT treatments to DIFFERENT columns, chosen by rng -- real tables are a mix
    of continuous, skewed and categorical columns, not one transform applied uniformly.
  - Transformations must be MONOTONE per column (or a coarsening of a monotone map), so the
    SCM's causal ordering survives. Rank-preserving maps are safe: exp, x**3, log1p on a
    shifted column, quantile-binning. A random permutation of values is NOT safe -- it
    destroys the signal.
  - Every output must be finite. No NaN, no inf. Guard every log and division.
  - numpy is available as np. No imports, no I/O, no loops over rows.
  - Do not change the number of rows or columns, and do not touch y.
{retry}

JSON: {{"code": "def realize(X, y, rng):\\n    ...", "why": "<one sentence>"}}"""

    r = llm.ask_json(prompt, {})
    if not isinstance(r, dict):
        return "", "llm returned nothing"
    return r.get("code", ""), str(r.get("why", ""))[:200]


# ─────────────────────────────────────────────────────────────────── run it safely

def compile_realizer(code: str):
    """Return a callable, or None. A realiser that crashes or emits garbage is rejected here,
    not three hours later in a training run."""
    code, why = _sanitize(code)
    if not code:
        return None, why
    g = {"np": np}
    try:
        exec(code, g)  # noqa: S102 - LLM-authored, screened above, no imports or IO
        fn = g.get("realize")
        if not callable(fn):
            return None, "no realize() defined"
    except Exception as e:
        return None, f"compile failed: {type(e).__name__}: {e}"

    # smoke it on a fake batch before trusting it with the real one
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 7)).astype(np.float32)
    y = rng.integers(0, 2, 64)
    try:
        out = np.asarray(fn(X.copy(), y.copy(), rng), dtype=np.float64)
    except Exception as e:
        return None, f"raised on a test batch: {type(e).__name__}: {e}"
    if out.shape != X.shape:
        return None, f"changed shape {X.shape} -> {out.shape}"
    if not np.isfinite(out).all():
        return None, "produced NaN or inf"
    if np.allclose(out, X):
        return None, "is the identity -- it changes nothing"
    return fn, "ok"


class RealisticPrior:
    """A PriorDataset, with the LLM's realiser applied to every batch it yields."""

    def __init__(self, prior, realize):
        self.prior, self.realize = prior, realize

    def __iter__(self):
        for batch in self.prior:
            X, y, *rest = batch
            dev, dt = X.device, X.dtype
            Xn = X.detach().cpu().numpy().astype(np.float64)
            yn = y.detach().cpu().numpy()
            rng = np.random.default_rng(int(abs(float(Xn.sum())) * 1e3) % (2**31))
            out = np.empty_like(Xn)
            for b in range(Xn.shape[0]):
                try:
                    r = np.asarray(self.realize(Xn[b].copy(), yn[b].copy(), rng), dtype=np.float64)
                    out[b] = r if (r.shape == Xn[b].shape and np.isfinite(r).all()) else Xn[b]
                except Exception:
                    out[b] = Xn[b]          # a bad row falls back to the raw SCM, never crashes
            # re-standardise: the model's embedder expects roughly unit scale
            mu = out.mean(1, keepdims=True)
            sd = out.std(1, keepdims=True)
            sd[sd < 1e-6] = 1.0
            out = np.clip((out - mu) / sd, -100, 100)
            yield (torch.as_tensor(out, dtype=dt, device=dev), y, *rest)


def make_realistic_prior(cfg: dict, batch_size: int, realize):
    import priortrain as pt
    return RealisticPrior(pt.make_prior(cfg, batch_size=batch_size), realize)
