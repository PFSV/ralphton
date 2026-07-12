"""The other half of the story: don't fix the prior, fix the context.

Everything in agent_loop.py tries to repair what the frozen model *learned from* — its
synthetic prior — and every arm loses to leaving the checkpoint alone. The reason appears to
be that the prior's unrealism is domain randomisation, and narrowing it destroys coverage.

But a tabular foundation model has a second, much more accessible surface: the in-context set.
`fit()` is not training — it is loading a context. So the same LLM agent can act at inference
time instead, with the model completely untouched: derive features from what the column names
MEAN, curate which rows the model sees, synthesise rows to fill a gap, or buy real labelled
rows. Each action has a price; one credit buys one real row. Nothing is trained; every
candidate context costs one forward pass.

On credit-g this reached 0.7789 from a 0.7652 baseline for 33 of 100 credits, spending them
on three world-knowledge features (monthly_installment_burden, repayment_strain,
liquidity_adjusted_burden) and 30 uncertainty-sampled rows. This runs that agent across
TabArena and asks whether it holds up.

Arms, all on the SAME frozen TabICLv2 and the SAME credit budget:
  raw        the context as given                        (no agent, no spend)
  buy-unc    spend the whole budget on real rows,
             chosen by uncertainty                        (active learning; no LLM)
  caafe      LLM feature engineering only                 (our CAAFE-on-TabICL)
  agent      the full cost-aware agent                    (ours)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import agent as A
import tabarena
import tfm

HERE = Path(__file__).parent
OUT = HERE / "context_bench.jsonl"

ARMS = {
    "buy-unc": ("acquire",),
    "caafe":   ("add_feature", "drop_feature"),
    "agent":   A.ACTIONS,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--arms", default="raw,buy-unc,caafe,agent")
    ap.add_argument("--limit", type=int, default=0, help="first N TEST tasks (0 = all)")
    a = ap.parse_args()

    _, test = tabarena.load_split(a.seed)
    if a.limit:
        test = test[: a.limit]
    arms = a.arms.split(",")

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["ds"], r["arm"], r["seed"], r["budget"]))
            except json.JSONDecodeError:
                pass

    print(f"TabArena: {len(test)} tasks | budget {a.budget:.0f} credits "
          f"(1 credit = 1 real labelled row) | seed {a.seed}\n", flush=True)

    for t in test:
        for arm in arms:
            key = (t.name, arm, a.seed, a.budget)
            if key in done:
                continue
            t0 = time.time()
            if arm == "raw":
                v0 = tfm.score(t.X_ctx, t.y_ctx, t.X_val, t.y_val, a.seed)
                s0 = tfm.score(t.X_ctx, t.y_ctx, t.X_test, t.y_test, a.seed)
                R = A.Run(t.name, "raw", t.semantic, 0.0, val0=v0, test0=s0, val=v0, test=s0)
            else:
                R = A.run(t, arm, a.budget, seed=a.seed, allow=ARMS[arm],
                          max_iters=a.iters)
            d = asdict(R)
            d.update(ds=t.name, arm=arm, seed=a.seed, budget=a.budget,
                     secs=round(time.time() - t0, 1))
            with OUT.open("a") as f:
                f.write(json.dumps(d, default=float) + "\n")
            print(f"  {t.name:<42s} {arm:<9s} "
                  f"{R.test0:.4f} -> {R.test:.4f} ({R.test - R.test0:+.4f})  "
                  f"spent {R.spent:>5.1f}  feats {R.feats_added} bought {R.rows_bought} "
                  f"synth {R.rows_synth}  [{time.time()-t0:.0f}s]", flush=True)

    # summary
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["seed"] == a.seed and r["budget"] == a.budget]
    base = {r["ds"]: r["test"] for r in rows if r["arm"] == "raw"}
    print(f"\n{'arm':<10s} {'mean TEST':>10s} {'vs raw':>9s} {'wins':>8s} {'mean spend':>11s}")
    print("-" * 52)
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm and r["ds"] in base]
        if not rs:
            continue
        d = np.array([r["test"] - base[r["ds"]] for r in rs])
        print(f"{arm:<10s} {np.mean([r['test'] for r in rs]):>10.4f} {d.mean():>+9.4f} "
              f"{int((d > 0).sum()):>4d}/{len(d)} "
              f"{np.mean([r['spent'] for r in rs]):>11.1f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
