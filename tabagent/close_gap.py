"""Close the distribution gap: the LLM writes the generator, the C2ST grades it, repeat.

This is the fast inner loop. No GPU training, no downstream evaluation -- just: does the
prior look like real data yet? A classifier separates the released prior from real TabArena
tables with AUC 1.000. Every round the LLM rewrites the realisation function, we re-measure,
and the score goes back to it. Whatever survives is then worth spending a LoRA run on.

Grading is honest in one specific way: C2ST is scored against real tables the LLM never sees.
It only ever receives the *statistics* of the gap, never a row.

    python close_gap.py --rounds 6
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import analysis2 as A2
import priortrain as pt
import realism

HERE = Path(__file__).parent


def prior_meta_with(cfg, realize, n=40, seed=0) -> pd.DataFrame:
    """Meta-features of datasets sampled from the prior, after the realiser runs."""
    src = (realism.make_realistic_prior(cfg, 4, realize) if realize
           else pt.make_prior(cfg, batch_size=4))
    rows, it = [], iter(src)
    while len(rows) < n:
        X, y, *_ = next(it)
        Xn, yn = X.cpu().numpy(), y.cpu().numpy()
        for b in range(Xn.shape[0]):
            xb = Xn[b]
            keep = ~np.all(np.abs(xb) < 1e-12, axis=0)
            if keep.sum() >= 2 and len(np.unique(yn[b])) >= 2:
                rows.append(A2.meta(xb[:, keep], yn[b]))
            if len(rows) >= n:
                break
    return pd.DataFrame(rows)


# Axes the realiser cannot possibly fix, so they must not be scored. n_rows and
# rows_per_feature are set by OUR train/test splitting (we cap real tables at 2000 rows), not
# by the prior. Leaving them in guarantees a classifier separates the two sets perfectly no
# matter what the LLM writes -- which is exactly what happened: C2ST sat at 1.000 for every
# round and carried no gradient at all.
ARTIFACT = {"n_rows", "rows_per_feature", "n_features", "n_classes"}
SCORED = [k for k in A2.KEYS if k not in ARTIFACT]


def gap_score(prior_df, real_df) -> tuple[float, "pd.DataFrame"]:
    """Mean KS distance over the axes a realiser can actually move. Unlike C2ST this is
    graded: it falls smoothly as the marginals converge, so it can be optimised against."""
    axes = A2.per_axis(prior_df, real_df)
    axes = axes[axes.axis.isin(SCORED)]
    return float(axes.ks.mean()), axes


def diagnose(prior_df, real_df, seed=0) -> tuple[float, float, str]:
    """Returns (gap, c2st, text). The gap is the objective; C2ST is reported, not optimised."""
    gap, axes = gap_score(prior_df, real_df)
    auc, p, imp = A2.c2st(prior_df, real_df, n_perm=40, seed=seed)

    lines = [f"Distribution gap = {gap:.3f}  (mean KS distance over the properties you can "
             f"change; 0.0 = the synthetic marginals match real ones. THIS IS THE SCORE.)",
             f"A classifier still separates synthetic from real with AUC {auc:.3f}.",
             "",
             "Per property (worst first) -- KS is how far apart the two distributions are:"]
    for r in axes.sort_values("ks", ascending=False).head(9).itertuples():
        d = "TOO HIGH" if r.cohens_d > 0 else "TOO LOW"
        lines.append(f"  KS {r.ks:.2f}  {r.axis:<20s} synthetic {r.prior:9.3f}  "
                     f"real {r.real:9.3f}   {d}")
    return gap, auc, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    out = HERE / f"close_gap_{a.seed}.json"
    real = A2.real_meta(a.seed)
    cfg = dict(pt.BASE_CFG)

    gap0, auc0, diag = diagnose(prior_meta_with(cfg, None, a.n, a.seed), real, a.seed)
    print(f"baseline: released prior vs real  ->  gap {gap0:.3f}  (C2ST {auc0:.3f})\n",
          flush=True)
    print(diag + "\n", flush=True)

    history, best = [], (gap0, None, "released prior (no realiser)", auc0)
    code, feedback = None, None

    for r in range(a.rounds):
        t0 = time.time()
        print(f"── round {r} ─────────────────────────────", flush=True)
        code, why = realism.write_realizer(diag, previous=code, feedback=feedback)
        fn, msg = realism.compile_realizer(code)
        if fn is None:
            print(f"  rejected: {msg}", flush=True)
            feedback = f"Your code was rejected: {msg}. It never ran. Fix that and resend."
            history.append(dict(round=r, rejected=msg, code=code))
            continue

        df = prior_meta_with(cfg, fn, a.n, a.seed)
        gap, auc, diag_new = diagnose(df, real, a.seed)
        print(f"  {why}", flush=True)
        print(f"  gap {gap0:.3f} -> {gap:.3f}  ({gap - gap0:+.3f})   "
              f"C2ST {auc:.3f}   [{time.time()-t0:.0f}s]", flush=True)

        feedback = (f"Your realiser moved the distribution gap from {gap0:.3f} to {gap:.3f} "
                    f"({'better' if gap < gap0 else 'WORSE'}; 0.0 is the goal). What is still "
                    f"wrong:\n{diag_new}")
        history.append(dict(round=r, why=why, gap=gap, c2st=auc, delta=gap - gap0, code=code,
                            secs=round(time.time() - t0, 1)))
        if gap < best[0]:
            best = (gap, code, why, auc)
            print("  ** new best **", flush=True)
        diag = diag_new
        out.write_text(json.dumps(dict(baseline_gap=gap0, baseline_c2st=auc0,
                                       history=history, best_gap=best[0],
                                       best_code=best[1], best_why=best[2],
                                       best_c2st=best[3]), indent=1, default=float))

    print(f"\n══ best: gap {gap0:.3f} -> {best[0]:.3f}  ({best[0]-gap0:+.3f})   "
          f"C2ST {auc0:.3f} -> {best[3]:.3f}")
    print(f"   {best[2]}")
    if best[1]:
        print("\n--- the generator the LLM wrote ---")
        print(best[1])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
