"""Experiment driver. Resumable: every finished job is appended to results.jsonl and
skipped on a rerun, so a dropped tunnel or a killed process costs nothing.

Arms (all share the SAME frozen TabICLv2 and the SAME budget accounting):
  raw           context as given, no agent, zero spend
  xgb / cat     non-TFM baselines on the same context
  buy-rand      spend the entire budget on random real rows          (no LLM)
  buy-unc       spend the entire budget on uncertainty-sampled rows  (no LLM; active learning)
  caafe         LLM feature engineering only, no data purchase       (our CAAFE-on-TabICL)
  agent         cost-aware agent, full action space                  (ours)
  agent-nosynth / agent-nobuy    action-space ablations
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import agent
import data
import tfm

OUT = Path(__file__).parent / "results.jsonl"

ARMS = {
    "buy-rand":      dict(allow=("acquire",),               force={"strategy": "random"}),
    "buy-unc":       dict(allow=("acquire",),               force={"strategy": "uncertainty"}),
    "caafe":         dict(allow=("add_feature", "drop_feature")),
    "agent":         dict(allow=agent.ACTIONS),
    "agent-nosynth": dict(allow=("add_feature", "drop_feature", "curate_context", "acquire")),
    "agent-nobuy":   dict(allow=("add_feature", "drop_feature", "curate_context", "synthesize")),
}


def _tree_baseline(t, kind: str, seed: int) -> float:
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    m = (XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                       tree_method="hist", random_state=seed, verbosity=0)
         if kind == "xgb" else
         CatBoostClassifier(iterations=400, depth=5, learning_rate=0.08,
                            random_seed=seed, verbose=0))
    m.fit(t.X_ctx.to_numpy(), t.y_ctx)
    p = m.predict_proba(t.X_test.to_numpy())
    if t.n_classes == 2:
        return float(roc_auc_score(t.y_test, p[:, 1]))
    return float(roc_auc_score(t.y_test, p, multi_class="ovr", average="macro"))


def _no_llm_buy(t, strategy, budget, seed):
    """Baselines that spend the whole budget on real rows, in one shot."""
    frames = {"ctx": t.X_ctx.copy(), "pool": t.X_pool.copy(),
              "val": t.X_val.copy(), "test": t.X_test.copy()}
    k = min(int(budget / tfm.PRICES.acquire_per_row), len(t.X_pool))
    idx = agent._acquire(t, frames, t.y_ctx, [], [], [], k, seed, strategy=strategy)
    import pandas as pd
    X = pd.concat([t.X_ctx, t.X_pool.iloc[idx]], ignore_index=True)
    y = np.concatenate([t.y_ctx, t.y_pool[idx]])
    return agent.Run(
        task=t.name, arm=f"buy-{strategy}", semantic=t.semantic, budget=budget,
        val0=tfm.score(t.X_ctx, t.y_ctx, t.X_val, t.y_val, seed),
        test0=tfm.score(t.X_ctx, t.y_ctx, t.X_test, t.y_test, seed),
        val=tfm.score(X, y, t.X_val, t.y_val, seed),
        test=tfm.score(X, y, t.X_test, t.y_test, seed),
        spent=float(k * tfm.PRICES.acquire_per_row), rows_bought=len(idx), llm_calls=0,
    )


def job(spec: dict) -> dict:
    ds, arm, seed, budget, anon = (spec["ds"], spec["arm"], spec["seed"],
                                   spec["budget"], spec["anon"])
    t0 = time.time()
    did, name, sem, _ = next(r for r in data.REGISTRY if r[1] == ds)
    t = data.load(did, name, sem, seed=seed)
    if anon:
        t = data.anonymize(t)

    if arm == "raw":
        R = agent.Run(t.name, "raw", t.semantic, 0.0,
                      val0=tfm.score(t.X_ctx, t.y_ctx, t.X_val, t.y_val, seed),
                      test0=tfm.score(t.X_ctx, t.y_ctx, t.X_test, t.y_test, seed))
        R.val, R.test = R.val0, R.test0
    elif arm in ("xgb", "cat"):
        s = _tree_baseline(t, arm, seed)
        R = agent.Run(t.name, arm, t.semantic, 0.0, val0=0.0, test0=s, val=0.0, test=s)
    elif arm in ("buy-rand", "buy-unc"):
        R = _no_llm_buy(t, "random" if arm == "buy-rand" else "uncertainty", budget, seed)
        R.arm = arm
    else:
        R = agent.run(t, arm, budget, seed=seed, allow=ARMS[arm]["allow"], max_iters=8)

    d = asdict(R)
    d.update(ds=ds, arm=arm, seed=seed, budget=budget, anon=anon,
             secs=round(time.time() - t0, 1), tfm=tfm.tfm_calls())
    return d


def worker(spec):
    try:
        return job(spec)
    except Exception:
        return {**spec, "error": traceback.format_exc()[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    names = [r[1] for r in data.REGISTRY]
    sem = [r[1] for r in data.REGISTRY if r[2]]
    jobs: list[dict] = []

    if a.stage == "main":                      # main table: B=100, 3 seeds
        for ds in names:
            for seed in (0, 1, 2):
                for arm in ("raw", "xgb", "cat", "buy-rand", "buy-unc", "caafe", "agent"):
                    jobs.append(dict(ds=ds, arm=arm, seed=seed, budget=100.0, anon=False))
    elif a.stage == "curve":                   # accuracy-vs-cost
        for ds in names:
            for B in (25.0, 50.0, 200.0):
                for arm in ("buy-unc", "caafe", "agent"):
                    jobs.append(dict(ds=ds, arm=arm, seed=0, budget=B, anon=False))
    elif a.stage == "anon":                    # the semantic ablation
        for ds in sem:
            for seed in (0, 1, 2):
                for arm in ("agent", "caafe"):
                    jobs.append(dict(ds=ds, arm=arm, seed=seed, budget=100.0, anon=True))
    elif a.stage == "ablate":                  # action-space ablation
        for ds in names:
            for seed in (0, 1, 2):
                for arm in ("agent-nosynth", "agent-nobuy"):
                    jobs.append(dict(ds=ds, arm=arm, seed=seed, budget=100.0, anon=False))

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["ds"], r["arm"], r["seed"], r["budget"], r["anon"]))
            except json.JSONDecodeError:
                pass
    jobs = [j for j in jobs
            if (j["ds"], j["arm"], j["seed"], j["budget"], j["anon"]) not in done]
    print(f"stage={a.stage}: {len(jobs)} jobs to run ({len(done)} already done)", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            with OUT.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            tag = "ERR " if "error" in r else ""
            print(f"[{i}/{len(jobs)}] {tag}{r['ds']:<20s} {r['arm']:<14s} s{r['seed']} "
                  f"B={r['budget']:.0f} anon={r['anon']} "
                  f"test={r.get('test', float('nan')):.4f} "
                  f"(base {r.get('test0', float('nan')):.4f}) "
                  f"spent={r.get('spent', 0)} [{(time.time()-t0)/60:.1f}m]", flush=True)


if __name__ == "__main__":
    main()
