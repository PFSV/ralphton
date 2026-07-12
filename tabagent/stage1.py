"""Stage 1 (prior revision) and Stage 2 (held-out transfer).

The released TabICL checkpoint was pre-trained on a prior nobody tuned for the tasks it
would eventually meet. We let an agent edit that prior's generating configuration, adapt
the checkpoint to each candidate prior with LoRA, and score the result on a set of DEV
tasks it is allowed to look at. Stage 2 then reports the winner on TEST tasks that were
never part of the loop.

Three search arms share one search budget (R candidate priors, identical LoRA compute):
  base    keep TabICL's own prior; no search at all
  random  sample R priors uniformly from the knob space   (search, no LLM)
  agent   an LLM revises the prior from the DEV diagnosis (search, with LLM)   <- ours

`random` is the arm that matters: without it, any gain could just be search.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import data
import llm
import priortrain as pt

OUT = Path(__file__).parent / "stage1.jsonl"

# Legacy hand-picked suite, kept so the earlier runs remain reproducible.
DEV_TASKS = ["credit-g", "pc4", "climate", "qsar-biodeg", "banknote"]
TEST_TASKS = ["cmc", "churn", "spambase", "ozone", "phoneme", "bank-marketing-anon"]

# Adaptation setting. lr=1e-4 was destroying the pre-trained weights (it lost 0.0056 test
# AUC purely by over-writing them); 1e-5 is the value the DEV sweep selected. See
# sweep_adapt.py / sweep_full.py.
ADAPTER, LR = "lora", 1e-5


def _load(names, seed):
    out = []
    for n in names:
        did, nm, sem, _ = next(r for r in data.REGISTRY if r[1] == n)
        out.append(data.load(did, nm, sem, seed=seed))
    return out


def load_suite(suite: str, seed: int):
    if suite == "tabarena":
        import tabarena
        return tabarena.load_split(seed)
    return _load(DEV_TASKS, seed), _load(TEST_TASKS, seed)


def evaluate(state, tasks, seed, adapter=ADAPTER) -> dict[str, float]:
    scores = {}
    for t in tasks:
        m = pt.LoRATabICL(state, seed=seed, adapter=adapter).fit(t.X_ctx.to_numpy(), t.y_ctx)
        p = m.predict_proba(t.X_test.to_numpy())
        present = np.unique(t.y_ctx)
        if t.n_classes == 2 and p.shape[1] == 2:
            a = roc_auc_score(t.y_test, p[:, 1])
        else:
            P = np.zeros((len(t.X_test), t.n_classes))
            P[:, present.astype(int)] = p[:, : len(present)]
            P = P / P.sum(1, keepdims=True).clip(1e-9)
            a = roc_auc_score(t.y_test, P, multi_class="ovr", average="macro",
                              labels=list(range(t.n_classes)))
        scores[t.name] = float(a)
    return scores


def train_and_eval(cfg, dev, seed, steps, batch_size, lr=LR, adapter=ADAPTER):
    clf = pt.load_backbone(seed=seed)
    clf, st = pt.train_lora(clf, cfg, steps=steps, batch_size=batch_size, lr=lr,
                            seed=seed, log_every=0, adapter=adapter)
    state = pt.lora_state(clf.model_)
    dev_scores = evaluate(state, dev, seed, adapter)
    return state, dev_scores, st


# ------------------------------------------------------------------ search arms

def random_cfg(rng) -> dict:
    cfg = {}
    for k, (lo, hi, kind) in pt.KNOBS.items():
        if kind is bool:
            cfg[k] = int(rng.integers(0, 2))
        elif kind is int:
            cfg[k] = int(rng.integers(lo, hi + 1))
        else:
            cfg[k] = float(rng.uniform(lo, hi))
    return pt.clip_cfg(cfg)


def _task_card(tasks) -> str:
    """Summary statistics only, and the tasks are never named.

    Withholding the name matters: an LLM has memorised the contents of well-known OpenML
    tables (Bordt et al., 2024), so revealing "credit-g" would let recalled specifics, rather
    than reasoning about table shape, drive the revision. The agent sees only what a
    practitioner could read off any table's header.
    """
    rows = []
    for i, t in enumerate(tasks, 1):
        bal = np.bincount(t.y_ctx) / len(t.y_ctx)
        rows.append(f"  task {i}: {t.X_ctx.shape[1]} features, {t.n_classes} classes, "
                    f"context {len(t.X_ctx)} rows, class balance {np.round(bal, 2).tolist()}")
    return "\n".join(rows)


def agent_cfg(cur_cfg, history, dev, rng) -> dict:
    hist = "\n".join(
        f"  round {i}: dev mean AUC {h['dev_mean']:.4f} | cfg diffs vs TabICL default: "
        f"{json.dumps(h['diff'])}"
        for i, h in enumerate(history)
    ) or "  (none yet)"
    worst = ""
    if history:
        last = history[-1]["dev"]
        idx = {t.name: i + 1 for i, t in enumerate(dev)}
        order = sorted(last.items(), key=lambda kv: kv[1])
        worst = "\n".join(f"  task {idx[k]}: AUC {v:.4f}" for k, v in order)

    prompt = f"""You are tuning the SYNTHETIC PRIOR that a tabular foundation model is adapted to.

The model (TabICL) is pre-trained on random structural causal models. You cannot change the
model. You CAN change the distribution of synthetic datasets it is LoRA-adapted to. Your goal
is to make it better on REAL downstream tables.

The real tasks it will be scored on (development set):
{_task_card(dev)}

Per-task AUC in the most recent round (worst first):
{worst or "  (no round yet)"}

Search history:
{hist}

Knobs you may set, with hard bounds:
{json.dumps({k: [v[0], v[1], v[2].__name__] for k, v in pt.KNOBS.items()}, indent=1)}

TabICL's own default prior (the starting point):
{json.dumps(pt.BASE_CFG, indent=1)}

Reason about the MISMATCH between that prior and the real tasks above — dataset size, number
of features, class count and imbalance, how nonlinear the true functions plausibly are, how
much noise real tables carry, whether categorical columns are common. Then propose a prior
that over-represents the regime these real tasks actually live in.

Return the FULL config (every key). Change what you can justify; keep the rest.

JSON: {{"cfg": {{...all keys...}}, "why": "<one sentence>"}}"""
    r = llm.ask_json(prompt, {})
    cfg = r.get("cfg") if isinstance(r, dict) else None
    if not isinstance(cfg, dict) or not cfg:
        return random_cfg(rng)          # LLM failed -> do not stall the run
    return pt.clip_cfg(cfg)


def diff_vs_base(cfg) -> dict:
    return {k: v for k, v in cfg.items()
            if not np.isclose(float(v), float(pt.BASE_CFG[k]), rtol=1e-3)}


# ------------------------------------------------------------------ driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pretrained", "base", "random", "agent"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--suite", default="tabarena", choices=["tabarena", "legacy"])
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--adapter", default=ADAPTER, choices=["lora", "full"])
    a = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    dev, test = load_suite(a.suite, a.seed)
    rec = dict(arm=a.arm, seed=a.seed, rounds=a.rounds, steps=a.steps,
               suite=a.suite, lr=a.lr, adapter=a.adapter)

    if a.arm == "pretrained":
        rec["dev"] = evaluate(None, dev, a.seed, a.adapter)
        rec["test"] = evaluate(None, test, a.seed, a.adapter)
        rec["lora_runs"] = 0
    else:
        history, best = [], None
        rounds = 1 if a.arm == "base" else a.rounds
        for r in range(rounds):
            if a.arm == "base":
                cfg = dict(pt.BASE_CFG)
            elif a.arm == "random":
                cfg = random_cfg(rng)
            else:
                cfg = dict(pt.BASE_CFG) if r == 0 else agent_cfg(cfg, history, dev, rng)

            state, dev_scores, st = train_and_eval(cfg, dev, a.seed, a.steps, a.batch_size,
                                                   lr=a.lr, adapter=a.adapter)
            dm = float(np.mean(list(dev_scores.values())))
            history.append(dict(cfg=cfg, diff=diff_vs_base(cfg), dev=dev_scores,
                                dev_mean=dm, loss=st.final_loss, secs=st.secs))
            print(f"[{a.arm} s{a.seed}] round {r}: dev mean AUC {dm:.4f} "
                  f"(loss {st.final_loss:.3f}, {st.secs}s) diff={json.dumps(diff_vs_base(cfg))}",
                  flush=True)
            if best is None or dm > best[1]:
                best = (state, dm, cfg, r)

        state, dm, cfg, r_best = best
        rec.update(best_round=r_best, best_dev_mean=dm, best_cfg=cfg,
                   best_diff=diff_vs_base(cfg),
                   dev=history[r_best]["dev"], test=evaluate(state, test, a.seed),
                   history=[{k: h[k] for k in ("dev_mean", "diff", "loss")} for h in history],
                   lora_runs=rounds, llm_calls=llm.n_calls())

    rec["test_mean"] = float(np.mean(list(rec["test"].values())))
    rec["secs"] = round(time.time() - t0, 1)
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n== {a.arm} seed {a.seed}: TEST mean AUC {rec['test_mean']:.4f} "
          f"| {json.dumps({k: round(v,4) for k,v in rec['test'].items()})} "
          f"| {rec['secs']}s", flush=True)


if __name__ == "__main__":
    main()
