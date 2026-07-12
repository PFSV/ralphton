"""The cost-aware context agent.

The foundation model is frozen, so the only thing the agent can change is what the model
is shown. Each iteration it (1) diagnoses the current context against validation errors,
(2) chooses one priced action, (3) executes it, (4) keeps it only if validation improves,
and (5) writes the outcome to a scratchpad that conditions the next choice.

The interesting decision is (2): a real labelled row costs 1.0 credit, a derived feature
costs 1.0 total, and a synthetic row costs 0.01. Under a fixed budget the agent has to
decide when world knowledge is a good enough substitute for going out and buying data.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import llm
from tfm import PRICES, Budget, proba, score

ACTIONS = ("add_feature", "drop_feature", "curate_context", "synthesize", "acquire")


@dataclass
class Step:
    it: int
    action: str
    detail: str
    cost: float
    val_before: float
    val_after: float
    accepted: bool


@dataclass
class Run:
    task: str
    arm: str
    semantic: bool
    budget: float
    val0: float
    test0: float
    val: float = 0.0
    test: float = 0.0
    spent: float = 0.0
    rows_bought: int = 0
    rows_synth: int = 0
    feats_added: int = 0
    llm_calls: int = 0
    steps: list[Step] = field(default_factory=list)
    ledger: list = field(default_factory=list)


def _exec_col(code: str, frames: dict[str, pd.DataFrame], new: str) -> bool:
    """Run LLM-authored feature code against every split. All-or-nothing."""
    out = {}
    for k, df in frames.items():
        g = {"pd": pd, "np": np, "df": df.copy()}
        try:
            exec(code, g)  # noqa: S102 - LLM-authored feature code, sandboxed by review below
            d = g["df"]
            if new not in d.columns:
                return False
            v = pd.to_numeric(d[new], errors="coerce")
            if not np.isfinite(v.to_numpy(dtype=float)).all():
                return False
            out[k] = d.assign(**{new: v.astype(float)})
        except Exception:
            return False
    for k, df in out.items():
        frames[k] = df
    return True


# The same over-strict filter that silently ate three of five rounds in the prior loop was
# still live here: it banned the SUBSTRING "import", and the model writes `import numpy as np`
# out of habit even when told numpy is in scope. Every feature it proposed was thrown away
# before it ran, which reads in the logs as "the LLM had no good ideas" — it never got to have
# one. Strip the import lines (pd/np are already provided) and screen only what is actually
# dangerous.
DANGEROUS = re.compile(
    r"\b(exec|eval|compile|open|input|globals|locals|getattr|setattr|delattr"
    r"|subprocess|socket|requests|urllib|shutil|pathlib|pickle)\b"
    r"|\b(os|sys|io)\.|__[a-z]+__|while\s+True"
)
IMPORT_LINE = re.compile(r"^\s*(import|from)\s+.*$", re.M)


def _sanitize(code: str) -> tuple[str, str]:
    if not code:
        return "", "empty"
    code = IMPORT_LINE.sub("", code)
    hit = DANGEROUS.search(code)
    if hit:
        return "", f"dangerous construct: {hit.group(0)!r}"
    return code, "ok"


# ------------------------------------------------------------------ diagnosis

def _diagnose(t, X_ctx, y_ctx, X_val, seed) -> str:
    """Where is the current context failing? Report class balance + hardest val slices.

    X_val must be the CURRENT validation frame (same columns as X_ctx), not the pristine
    one — the agent may have added features since.
    """
    p = proba(X_ctx, y_ctx, X_val, seed)
    pred = p.argmax(1)
    conf = p.max(1)
    wrong = pred != t.y_val
    bal = pd.Series(y_ctx).value_counts().sort_index().to_dict()
    lines = [
        f"context size: {len(X_ctx)} rows, class counts {bal}",
        f"validation error rate: {wrong.mean():.3f} ({wrong.sum()}/{len(wrong)})",
        f"mean confidence on errors: {conf[wrong].mean():.3f}" if wrong.any() else "no errors",
    ]
    if wrong.any() and (~wrong).any():
        err_mean = X_val[wrong].mean(numeric_only=True)
        ok_mean = X_val[~wrong].mean(numeric_only=True)
        sd = X_val.std(numeric_only=True).replace(0, 1)
        drift = ((err_mean - ok_mean) / sd).abs().sort_values(ascending=False).head(5)
        lines.append("columns most shifted on misclassified rows (|z|): "
                     + ", ".join(f"{c}={v:.2f}" for c, v in drift.items()))
    return "\n".join(lines)


# ------------------------------------------------------------------ the loop

def run(t, arm: str, budget_total: float, seed: int = 0,
        allow: tuple[str, ...] = ACTIONS, max_iters: int = 12) -> Run:
    llm.reset_calls()
    B = Budget(budget_total)
    frames = {"ctx": t.X_ctx.copy(), "pool": t.X_pool.copy(),
              "val": t.X_val.copy(), "test": t.X_test.copy()}
    y_ctx = t.y_ctx.copy()
    bought: list[int] = []
    synth_X: list[pd.DataFrame] = []
    synth_y: list[np.ndarray] = []

    def ctx_now():
        X = frames["ctx"]
        y = y_ctx
        if bought:
            X = pd.concat([X, frames["pool"].iloc[bought]], ignore_index=True)
            y = np.concatenate([y, t.y_pool[bought]])
        if synth_X:
            X = pd.concat([X] + synth_X, ignore_index=True)
            y = np.concatenate([y] + synth_y)
        return X, y

    X0, y0 = ctx_now()
    val0 = score(X0, y0, frames["val"], t.y_val, seed)
    test0 = score(X0, y0, frames["test"], t.y_test, seed)
    R = Run(t.name, arm, t.semantic, budget_total, val0, test0)
    cur_val = val0
    scratch: list[str] = []

    for it in range(max_iters):
        if B.left < PRICES.curate:
            break
        Xc, yc = ctx_now()
        diag = _diagnose(t, Xc, yc, frames["val"], seed)

        affordable = [a for a in allow if _min_cost(a) <= B.left]
        if not affordable:
            break

        choice = _choose(t, frames["ctx"], diag, scratch, B, affordable)
        act = choice.get("action")
        if act not in affordable:
            act = affordable[0]

        val_before = cur_val
        ok, detail, cost = False, "", 0.0

        if act == "add_feature":
            cost = PRICES.llm_call
            B.charge("llm:add_feature", cost)
            spec = _propose_feature(t, frames["ctx"], diag, scratch)
            name = spec.get("name", "")
            code, why = _sanitize(spec.get("code", ""))
            if not name:
                detail = "no feature name"
            elif not code:
                detail = f"rejected: {why}"          # never silent — a filtered-out proposal
            elif name in frames["ctx"].columns:      # looks identical to a stupid agent
                detail = f"{name} (already exists)"
            else:
                trial = {k: v.copy() for k, v in frames.items()}
                if _exec_col(code, trial, name):
                    Xn, yn = _rebuild(trial, y_ctx, t, bought, synth_X, synth_y, name)
                    v = score(Xn, yn, trial["val"], t.y_val, seed)
                    if v > val_before + 1e-4:
                        frames, ok, cur_val = trial, True, v
                        synth_X[:] = [s.assign(**{name: 0.0}) if name not in s else s for s in synth_X]
                        R.feats_added += 1
                    detail = f"{name} := {code.splitlines()[-1][:60]}"
                else:
                    detail = f"{name} (exec failed)"

        elif act == "drop_feature":
            cost = PRICES.curate
            B.charge("drop_feature", cost)
            col = choice.get("column")
            if col in frames["ctx"].columns and frames["ctx"].shape[1] > 2:
                trial = {k: v.drop(columns=[col]) for k, v in frames.items()}
                sX = [s.drop(columns=[col], errors="ignore") for s in synth_X]
                Xn, yn = _rebuild(trial, y_ctx, t, bought, sX, synth_y)
                v = score(Xn, yn, trial["val"], t.y_val, seed)
                if v > val_before + 1e-4:
                    frames, synth_X[:], ok, cur_val = trial, sX, True, v
                detail = f"drop {col}"
            else:
                detail = "no such column"

        elif act == "curate_context":
            cost = PRICES.curate
            B.charge("curate_context", cost)
            Xc2, yc2 = ctx_now()
            keep = _curate(Xc2, yc2, t, frames, seed)
            if keep is not None and keep.sum() >= 50:
                v = score(Xc2[keep], yc2[keep], frames["val"], t.y_val, seed)
                if v > val_before + 1e-4:
                    # materialise the curated context as the new base
                    frames["ctx"], y_ctx = Xc2[keep].reset_index(drop=True), yc2[keep]
                    bought.clear(); synth_X.clear(); synth_y.clear()
                    ok, cur_val = True, v
                detail = f"kept {int(keep.sum())}/{len(keep)} rows"
            else:
                detail = "curation degenerate"

        elif act == "synthesize":
            k = int(np.clip(choice.get("k", 200), 20, 2000))
            cost = PRICES.synth_setup + PRICES.synth_per_row * k
            if not B.can(cost):
                k = max(20, int((B.left - PRICES.synth_setup) / PRICES.synth_per_row))
                cost = PRICES.synth_setup + PRICES.synth_per_row * k
            B.charge(f"synthesize:{k}", cost)
            Xs, ys = _synthesize(t, frames["ctx"], y_ctx, diag, k, seed)
            if Xs is not None:
                trial_sX, trial_sy = synth_X + [Xs], synth_y + [ys]
                Xn, yn = _rebuild(frames, y_ctx, t, bought, trial_sX, trial_sy)
                v = score(Xn, yn, frames["val"], t.y_val, seed)
                if v > val_before + 1e-4:
                    synth_X[:], synth_y[:] = trial_sX, trial_sy
                    ok, cur_val = True, v
                    R.rows_synth += len(Xs)
                detail = f"{len(Xs)} synthetic rows"
            else:
                detail = "synthesiser failed"

        elif act == "acquire":
            k = int(np.clip(choice.get("k", 50), 10, 400))
            k = min(k, int(B.left / PRICES.acquire_per_row), len(t.X_pool) - len(bought))
            if k <= 0:
                break
            cost = PRICES.acquire_per_row * k
            B.charge(f"acquire:{k}", cost)
            new = _acquire(t, frames, y_ctx, bought, synth_X, synth_y, k, seed,
                           strategy=choice.get("strategy", "uncertainty"))
            bought.extend(new)
            Xn, yn = ctx_now()
            cur_val = score(Xn, yn, frames["val"], t.y_val, seed)
            ok = cur_val > val_before
            R.rows_bought += len(new)
            detail = f"bought {len(new)} rows ({choice.get('strategy','uncertainty')})"

        R.steps.append(Step(it, act, detail, cost, val_before, cur_val, ok))
        scratch.append(f"[{it}] {act}: {detail} -> val {val_before:.4f}->{cur_val:.4f} "
                       f"{'KEPT' if ok else 'reverted'}; budget left {B.left:.1f}")
        scratch[:] = scratch[-8:]

    Xf, yf = ctx_now()
    R.val = cur_val
    R.test = score(Xf, yf, frames["test"], t.y_test, seed)
    R.spent, R.ledger, R.llm_calls = round(B.spent, 2), B.ledger, llm.n_calls()
    return R


def _min_cost(a: str) -> float:
    return {"add_feature": PRICES.llm_call, "drop_feature": PRICES.curate,
            "curate_context": PRICES.curate,
            "synthesize": PRICES.synth_setup + PRICES.synth_per_row * 20,
            "acquire": PRICES.acquire_per_row * 10}[a]


def _rebuild(frames, y_ctx, t, bought, synth_X, synth_y, newcol: str | None = None):
    X, y = frames["ctx"], y_ctx
    if bought:
        X = pd.concat([X, frames["pool"].iloc[bought]], ignore_index=True)
        y = np.concatenate([y, t.y_pool[bought]])
    if synth_X:
        sX = [s.assign(**{newcol: 0.0}) if newcol and newcol not in s.columns else s
              for s in synth_X]
        X = pd.concat([X] + [s[X.columns] for s in sX], ignore_index=True)
        y = np.concatenate([y] + list(synth_y))
    return X, y


# ------------------------------------------------------------------ LLM heads

def _choose(t, X, diag, scratch, B, affordable) -> dict:
    prompt = f"""You are the planner of a cost-aware data agent. A tabular foundation model is
FROZEN — you cannot train it. You can only change WHAT IT IS SHOWN (its in-context set).

{t.schema_card()}

Current diagnosis:
{diag}

What you have already tried (most recent last):
{chr(10).join(scratch) if scratch else "  (nothing yet)"}

BUDGET: {B.left:.1f} of {B.total:.1f} credits left.
PRICES — one credit buys ONE REAL LABELLED ROW.
  add_feature      1.0    LLM writes a new column from existing ones (world knowledge; free of new data)
  drop_feature     0.5    remove a column that is hurting
  curate_context   0.5    keep only the most useful rows already in context
  synthesize       2.0 + 0.01/row   LLM designs a sampler; simulated rows are ~100x cheaper than real
  acquire          1.0/row          buy real labelled rows from the reserve pool

Affordable right now: {affordable}

Pick the ONE action with the best expected validation gain per credit. Buying real data always
works but is expensive; a good derived feature is nearly free but only exists if the columns
mean something. Spend accordingly.

JSON: {{"action": <one of {affordable}>, "k": <rows, if synthesize/acquire>,
"strategy": "uncertainty"|"random", "column": <name, if drop_feature>, "why": "<12 words>"}}"""
    r = llm.ask_json(prompt, {"action": affordable[0]})
    return r if isinstance(r, dict) else {"action": affordable[0]}


def _propose_feature(t, X, diag, scratch) -> dict:
    prompt = f"""Write ONE new feature for this table. It must encode real-world knowledge about what
these columns MEAN — a ratio, an interaction, a threshold, a domain rule — not a random transform.

{t.schema_card()}

Diagnosis of current failures:
{diag}

Already tried:
{chr(10).join(scratch) if scratch else "  (nothing)"}

Rules: pandas only, `df` is in scope, no imports, no I/O. Assign exactly one new numeric column.
Must be finite for every row (guard divisions). Existing columns must not be modified.

JSON: {{"name": "<new_column>", "code": "df['<new_column>'] = ...", "why": "<12 words>"}}"""
    r = llm.ask_json(prompt, {})
    return r if isinstance(r, dict) else {}


def _synthesize(t, X_ctx, y_ctx, diag, k: int, seed: int):
    """LLM designs a conditional sampler; we run it. Falls back to a class-conditional
    Gaussian copula fit on the context if the LLM's code does not run."""
    prompt = f"""Design a SAMPLER that generates plausible new labelled rows for this table, to fill the
gaps the diagnosis reports. Use what the column names MEAN (ranges, signs, correlations,
plausible joint structure) — this is a simulator, not a resampler.

{t.schema_card()}

Diagnosis:
{diag}

Write a function `sample(n, rng)` returning `(rows, labels)`:
  rows   -> list of dicts with EXACTLY these keys: {list(X_ctx.columns)}
  labels -> list of ints in 0..{t.n_classes - 1}
`rng` is a numpy Generator. numpy is available as np. No imports, no I/O.

JSON: {{"code": "def sample(n, rng):\\n    ...", "why": "<12 words>"}}"""
    spec = llm.ask_json(prompt, {})
    code, _ = _sanitize(spec.get("code", "") if isinstance(spec, dict) else "")
    rng = np.random.default_rng(seed)
    if code:
        g = {"np": np}
        try:
            exec(code, g)  # noqa: S102
            rows, labels = g["sample"](k, rng)
            Xs = pd.DataFrame(rows)[list(X_ctx.columns)].astype(float)
            ys = np.asarray(labels, dtype=int)
            if len(Xs) == len(ys) and np.isfinite(Xs.to_numpy()).all() and ys.max() < t.n_classes:
                return Xs.reset_index(drop=True), ys
        except Exception:
            pass
    # fallback: class-conditional Gaussian on the current context
    try:
        Xs, ys = [], []
        for c in np.unique(y_ctx):
            m = X_ctx[y_ctx == c]
            if len(m) < 3:
                continue
            n_c = max(1, int(k * (y_ctx == c).mean()))
            cov = np.cov(m.to_numpy().T) + np.eye(m.shape[1]) * 1e-3
            s = rng.multivariate_normal(m.mean().to_numpy(), cov, n_c)
            Xs.append(pd.DataFrame(s, columns=X_ctx.columns))
            ys.append(np.full(n_c, c))
        if Xs:
            return pd.concat(Xs, ignore_index=True), np.concatenate(ys)
    except Exception:
        pass
    return None, None


def _curate(X, y, t, frames, seed):
    """Drop the context rows the model is most confidently wrong about (label-noise-like)."""
    try:
        p = proba(X, y, X, seed)
        conf_true = p[np.arange(len(y)), y]
        thresh = np.quantile(conf_true, 0.10)
        return conf_true > thresh
    except Exception:
        return None


def _acquire(t, frames, y_ctx, bought, synth_X, synth_y, k, seed, strategy="uncertainty"):
    avail = [i for i in range(len(t.X_pool)) if i not in set(bought)]
    if strategy == "random":
        return list(np.random.default_rng(seed).choice(avail, size=min(k, len(avail)), replace=False))
    X, y = _rebuild(frames, y_ctx, t, bought, synth_X, synth_y)
    try:
        p = proba(X, y, frames["pool"].iloc[avail], seed)
        margin = np.sort(p, axis=1)
        unc = 1.0 - (margin[:, -1] - (margin[:, -2] if p.shape[1] > 1 else 0))
        order = np.argsort(-unc)
        return [avail[i] for i in order[:k]]
    except Exception:
        return list(np.random.default_rng(seed).choice(avail, size=min(k, len(avail)), replace=False))
