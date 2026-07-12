"""One closed loop: audit the prior, revise it, adapt to it, test on a real benchmark,
read the errors, revise again.

    audit ─────────────┐
                       ▼
    errors ────────► agent ──► prior cfg ──► LoRA ──► TabArena DEV ──┐
      ▲                                                              │
      └──────────────────────────────────────────────────────────────┘
                                                        (best on DEV)
                                                              │
                                                              ▼
                                                       TabArena TEST
                                                    (seen exactly once)

What makes this different from a hyper-parameter search: the agent is not only told *that*
a prior scored 0.83. It is told *how the prior's synthetic tables differ from real ones*
(the audit) and *which kinds of real task it is losing on* (the error analysis, expressed in
the same axes). Those two signals are commensurable, so the agent can reason from "I am
failing on the categorical-heavy tasks" to "my prior emits no categorical columns" to a
concrete edit of cat_prob -- rather than hill-climbing a scalar.

    python pipeline.py --rounds 6 --seed 0

Everything is checkpointed to pipeline_<seed>.json; rerunning resumes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import llm
import priortrain as pt
import tabarena
from prior_audit import AXES, describe
from stage1 import evaluate

HERE = Path(__file__).parent


# ─────────────────────────────────────────────────────────────── 1. audit the prior

def audit(cfg: dict, n_datasets: int = 24, seed: int = 0) -> dict:
    """Sample the prior and measure it on the same axes as real tables."""
    rows, bs = [], 4
    it = iter(pt.make_prior(cfg, batch_size=bs))
    while len(rows) < n_datasets:
        X, y, *_ = next(it)
        Xn, yn = X.cpu().numpy(), y.cpu().numpy()
        for b in range(Xn.shape[0]):
            xb = Xn[b]
            keep = ~np.all(np.abs(xb) < 1e-12, axis=0)   # drop zero-padding columns
            if keep.sum() >= 2:
                rows.append(describe(xb[:, keep], yn[b]))
            if len(rows) >= n_datasets:
                break
    return pd.DataFrame(rows).median().to_dict()


def real_targets() -> dict:
    """What the prior is *supposed* to look like — from TabArena's own metadata."""
    return tabarena.real_stats()


def mismatch(prior_stats: dict, real: dict) -> dict:
    """Signed, scale-free gap per axis. Positive = the prior overshoots reality."""
    out = {}
    for k, rv in real.items():
        if k not in prior_stats:
            continue
        denom = max(abs(rv), 1e-6)
        out[k] = round((prior_stats[k] - rv) / denom, 3)
    return out


# ────────────────────────────────────────────────── 2. why is it losing, and on what?

def error_analysis(dev, dev_scores: dict) -> tuple[str, dict]:
    """Express failure in the SAME axes as the audit, so the agent can connect them."""
    rows = []
    for t in dev:
        X = t.X_ctx.to_numpy()
        card = np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])
        _, cnt = np.unique(t.y_ctx, return_counts=True)
        rows.append(dict(task=t.name, auc=dev_scores[t.name],
                         n_features=X.shape[1], n_classes=t.n_classes,
                         cat_frac=float((card <= 10).mean()),
                         minority=float(cnt.min() / cnt.sum())))
    df = pd.DataFrame(rows).sort_values("auc")

    # which task property predicts failure?
    corr = {}
    for k in ("n_features", "n_classes", "cat_frac", "minority"):
        if df[k].nunique() > 1:
            r, p = stats.spearmanr(df[k], df.auc)
            corr[k] = (round(float(r), 3), round(float(p), 3))

    lines = ["  worst tasks (anonymised):"]
    for i, r in enumerate(df.head(4).itertuples(), 1):
        lines.append(f"    #{i} AUC {r.auc:.3f} | {r.n_features} features, "
                     f"{r.n_classes} classes, {r.cat_frac:.2f} categorical, "
                     f"minority {r.minority:.2f}")
    lines.append("  correlation of task property with AUC (Spearman rho, p):")
    for k, (r, p) in corr.items():
        flag = "  <-- failing on these" if (r < -0.4 and p < 0.3) else ""
        lines.append(f"    {k:<12s} rho {r:+.2f} (p={p:.2f}){flag}")
    return "\n".join(lines), corr


# ────────────────────────────────────────────────────────────── 3. the agent's turn

def revise(cfg, audit_stats, real, gap, err_text, history) -> tuple[dict, str]:
    hist = "\n".join(
        f"  round {i}: DEV {h['dev_mean']:.4f} | edits {json.dumps(h['diff'])}"
        for i, h in enumerate(history)) or "  (first round)"

    prompt = f"""You are tuning the SYNTHETIC PRIOR that a frozen tabular foundation model (TabICL) is
LoRA-adapted to. You cannot touch the model. You can only change the distribution of
synthetic datasets it is adapted on. Goal: make it better on REAL tables.

CURRENT PRIOR CONFIG
{json.dumps(cfg, indent=1)}

WHAT YOUR PRIOR ACTUALLY GENERATES vs WHAT REAL TABLES LOOK LIKE
(measured: 24 datasets sampled from your prior, against the TabArena benchmark's real
distribution. gap = (prior - real) / real; positive means your prior overshoots.)
{json.dumps({k: {"prior": round(audit_stats.get(k, float("nan")), 3),
                 "real": round(real[k], 3), "gap": gap.get(k)} for k in real}, indent=1)}

WHERE THE ADAPTED MODEL IS LOSING ON REAL DEVELOPMENT TASKS
{err_text}

SEARCH HISTORY
{hist}

KNOBS (hard bounds; anything outside is clipped)
{json.dumps({k: [v[0], v[1], v[2].__name__] for k, v in pt.KNOBS.items()}, indent=1)}

Reason in two steps, then act:
 1. Which axis of the prior is most wrong, and does it line up with the tasks you are losing
    on? (e.g. if AUC falls with cat_frac AND your prior emits no categorical columns, that is
    one story, not two.)
 2. Edit the knobs that move that axis. Do not thrash knobs that are already close.

Return the FULL config (every key), plus the one-sentence reason.

JSON: {{"cfg": {{...all keys...}}, "why": "<one sentence>"}}"""

    r = llm.ask_json(prompt, {})
    if not isinstance(r, dict) or not isinstance(r.get("cfg"), dict):
        return None, "llm failed"
    return pt.clip_cfg(r["cfg"]), str(r.get("why", ""))[:160]


# ─────────────────────────────────────────────────────────────────────── 4. the loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter", default="lora")
    a = ap.parse_args()

    out = HERE / f"pipeline_{a.seed}.json"
    state = json.loads(out.read_text()) if out.exists() else {"history": []}
    history = state["history"]

    dev, test = tabarena.load_split(a.seed)
    real = real_targets()
    print(f"TabArena: DEV {len(dev)} / TEST {len(test)} | real targets {real}\n", flush=True)

    base_dev = evaluate(None, dev, a.seed, a.adapter)
    base_dev_mean = float(np.mean(list(base_dev.values())))
    print(f"released checkpoint, no adaptation: DEV {base_dev_mean:.4f}\n", flush=True)

    cfg = dict(pt.BASE_CFG) if not history else history[-1]["cfg"]

    for r in range(len(history), a.rounds):
        t0 = time.time()
        print(f"── round {r} ─────────────────────────────────────────", flush=True)

        # (1) what does this prior actually produce?
        astats = audit(cfg, seed=a.seed)
        gap = mismatch(astats, real)
        print(f"  audit gap vs real: {gap}", flush=True)

        # (2) adapt to it, and score on the real benchmark
        clf = pt.load_backbone(seed=a.seed)
        clf, st = pt.train_lora(clf, cfg, steps=a.steps, batch_size=4, lr=a.lr,
                                seed=a.seed, log_every=0, adapter=a.adapter)
        adapter_state = pt.lora_state(clf.model_)
        dev_scores = evaluate(adapter_state, dev, a.seed, a.adapter)
        dm = float(np.mean(list(dev_scores.values())))
        print(f"  DEV {dm:.4f} ({dm - base_dev_mean:+.4f} vs released)  "
              f"[loss {st.final_loss:.3f}, {st.secs}s]", flush=True)

        # (3) where is it losing, in the same language as the audit?
        err_text, corr = error_analysis(dev, dev_scores)
        print(err_text, flush=True)

        history.append(dict(round=r, cfg=cfg, diff=pt.clip_cfg(cfg) and
                            {k: v for k, v in cfg.items()
                             if not np.isclose(float(v), float(pt.BASE_CFG[k]), rtol=1e-3)},
                            audit=astats, gap=gap, dev=dev_scores, dev_mean=dm,
                            corr=corr, loss=st.final_loss, secs=st.secs))
        state["history"] = history
        out.write_text(json.dumps(state, indent=1, default=float))

        # (4) the agent reads audit + errors and rewrites the prior
        if r + 1 < a.rounds:
            new_cfg, why = revise(cfg, astats, real, gap, err_text, history)
            if new_cfg is None:
                print("  agent failed to propose; stopping", flush=True)
                break
            edits = {k: v for k, v in new_cfg.items()
                     if not np.isclose(float(v), float(cfg[k]), rtol=1e-3)}
            print(f"  agent: {why}\n  edits: {json.dumps(edits)}"
                  f"  [{time.time()-t0:.0f}s]\n", flush=True)
            cfg = new_cfg
        del clf, adapter_state

    # ───────────────────────────── final: the best prior meets TEST, exactly once
    best = max(history, key=lambda h: h["dev_mean"])
    print(f"\n══ best prior: round {best['round']}, DEV {best['dev_mean']:.4f} "
          f"({best['dev_mean'] - base_dev_mean:+.4f})", flush=True)
    print(f"   edits vs TabICL default: {json.dumps(best['diff'])}", flush=True)

    clf = pt.load_backbone(seed=a.seed)
    clf, _ = pt.train_lora(clf, best["cfg"], steps=a.steps, batch_size=4, lr=a.lr,
                           seed=a.seed, log_every=0, adapter=a.adapter)
    st_best = pt.lora_state(clf.model_)

    base_test = evaluate(None, test, a.seed, a.adapter)
    best_test = evaluate(st_best, test, a.seed, a.adapter)
    bt = float(np.mean(list(base_test.values())))
    tt = float(np.mean(list(best_test.values())))
    wins = sum(best_test[k] > base_test[k] for k in base_test)

    print(f"\n══ TabArena TEST ({len(test)} held-out tasks, seen once) ══")
    print(f"   released checkpoint : {bt:.4f}")
    print(f"   agent-revised prior : {tt:.4f}  ({tt - bt:+.4f})  wins on {wins}/{len(test)}")

    state.update(base_dev=base_dev, base_dev_mean=base_dev_mean,
                 base_test=base_test, best_test=best_test,
                 base_test_mean=bt, best_test_mean=tt, test_delta=tt - bt,
                 test_wins=wins, n_test=len(test), best_round=best["round"])
    out.write_text(json.dumps(state, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
