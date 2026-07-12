"""Validate the agent, not the idea.

The claim under test is narrow and falsifiable:

    An LLM agent, told only that the prior's distribution is suspect and shown how it differs
    from real data, can author a SET of diverse synthetic priors whose mixture -- once a frozen
    TabICL is LoRA-adapted to it -- beats the released checkpoint on a real benchmark. And it
    beats an agent-free control given the same number of priors and the same GPU seconds.

The control is the whole experiment. `random` draws the same K priors uniformly from the same
knob space and gets the identical compute. If `agent` does not beat `random`, the LLM added
nothing and we say so.

Each round the agent is shown two kinds of evidence, in the same units:
  - DISTRIBUTIONAL: per-axis KS distance between what its prior emits and real tables
  - DOWNSTREAM:     per-task AUC on DEV, and which task properties predict failure
It then rewrites both the knobs and the realiser code. Bad round -> it is told exactly what
got worse, and by how much.

Selection is by MIXTURE, never argmax: one adapter is trained on all K priors round-robin, so
DEV picks nothing. (Argmax on validation is gradient descent on the validation set; it gained
+0.0042 DEV and gave back -0.0017 TEST when we tried it.)

    python agent_loop.py --arm agent  --rounds 4 --k 3
    python agent_loop.py --arm random --rounds 4 --k 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import analysis2 as A2
import close_gap as CG
import llm
import priortrain as pt
import realism
import tabarena
from stage1 import evaluate, random_cfg

HERE = Path(__file__).parent

# The one adapter setting the diagnostic found that helps rather than harms: a high LoRA
# scale lets the adapter act through a SMALL B, so it does not trample the pre-trained
# weights. alpha=32 costs -0.010 DEV; alpha=128 gains +0.002.
LR, ALPHA, STEPS_PER_PRIOR = 3e-5, 128, 350


# ────────────────────────────────────────────────────── train on a mixture of K priors

class Mixture:
    """Round-robin across priors so one adapter sees all of them.

    The realiser is numpy running per dataset on the CPU, and the training loop was spending
    all its time there — 524% CPU, 11% GPU. So materialise a buffer of already-realised
    batches once, park them on the GPU, and cycle. The transform cost is paid once instead of
    once per step; both arms pay it identically, so the comparison is untouched.
    """

    def __init__(self, priors, n_batches: int, device):
        self.buf = []
        per = max(1, n_batches // len(priors))
        for p in priors:
            it = iter(p)
            for _ in range(per):
                X, y, d, seq_len, train_size = next(it)
                self.buf.append((X.to(device), y.to(device), train_size))
        self.i = 0

    def __next__(self):
        b = self.buf[self.i % len(self.buf)]
        self.i += 1
        return b


def build_prior(spec, batch_size=4):
    """A prior spec is (knobs, realiser-code). The SCM supplies causal structure; the
    realiser, if any, rewrites the marginals into something that looks like a real column."""
    cfg = pt.clip_cfg(spec.get("cfg", {}))
    code = spec.get("code") or ""
    fn, _ = realism.compile_realizer(code) if code else (None, "")
    base = pt.make_prior(cfg, batch_size=batch_size)
    return realism.RealisticPrior(base, fn) if fn else base


def train_mixture(specs, seed, steps, n_batches: int = 192, batch_size: int = 4):
    # batch_size stays at 4. 16 OOMs (38GB peak: attention is quadratic in sequence length),
    # and 8 trips a CUDA "invalid configuration argument" inside TabICL's flattened-batch
    # SDPA kernel. The card looks idle at 4, but the real bottleneck is the CPU-side realiser
    # in the buffer build, not the GPU step — raising the batch does not fix that.
    clf = pt.load_backbone(seed=seed)
    model = clf.model_
    params = pt.inject_lora(model, alpha=ALPHA)
    model.to(pt.DEV).train()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)

    it = Mixture([build_prior(s, batch_size) for s in specs], n_batches, pt.DEV)
    losses = []
    for _ in range(steps):
        X, y, train_size = next(it)
        ts = int(train_size[0].item()) if torch.is_tensor(train_size) else int(train_size)
        y_tr, y_te = y[:, :ts], y[:, ts:]
        if y_te.numel() == 0:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(pt.DEV == "cuda")):
            pred = model(X, y_tr, None)
            loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(),
                                   y_te.reshape(-1).long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
    model.eval()
    st = pt.lora_state(model)
    del clf, model
    torch.cuda.empty_cache()
    return st, float(np.mean(losses[-20:])) if losses else float("nan")


# ─────────────────────────────────────────────────────────── the two kinds of evidence

def distribution_report(specs, real_meta, seed, n=12) -> tuple[float, str]:
    """How far is the mixture's output from real tables, per axis?"""
    frames = [CG.prior_meta_with(pt.clip_cfg(s.get("cfg", {})),
                                 realism.compile_realizer(s["code"])[0] if s.get("code") else None,
                                 n=max(8, n // len(specs)), seed=seed)
              for s in specs]
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    gap, axes = CG.gap_score(df, real_meta)
    lines = [f"Distribution gap of your prior mixture = {gap:.3f} "
             f"(mean KS vs real tables; 0 = identical marginals)."]
    for r in axes.sort_values("ks", ascending=False).head(6).itertuples():
        d = "TOO HIGH" if r.cohens_d > 0 else "TOO LOW"
        lines.append(f"  KS {r.ks:.2f}  {r.axis:<20s} yours {r.prior:9.3f}  "
                     f"real {r.real:9.3f}   {d}")
    return gap, "\n".join(lines)


def downstream_report(dev, scores, base_scores) -> str:
    """Where is the adapted model losing, and does it correlate with a task property?"""
    import pandas as pd
    from scipy import stats

    rows = []
    for t in dev:
        X = t.X_ctx.to_numpy()
        card = np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])
        _, cnt = np.unique(t.y_ctx, return_counts=True)
        rows.append(dict(auc=scores[t.name], delta=scores[t.name] - base_scores[t.name],
                         n_features=X.shape[1], n_classes=t.n_classes,
                         cat_frac=float((card <= 10).mean()),
                         minority=float(cnt.min() / cnt.sum())))
    df = pd.DataFrame(rows).sort_values("delta")

    lines = [f"Downstream: DEV AUC {np.mean([r['auc'] for r in rows]):.4f} "
             f"({np.mean([r['delta'] for r in rows]):+.4f} vs the untouched checkpoint). "
             f"You helped {sum(r['delta'] > 0 for r in rows)}/{len(rows)} tasks.",
             "  tasks you HURT most (anonymised):"]
    for r in df.head(3).itertuples():
        lines.append(f"    delta {r.delta:+.4f} | {r.n_features} features, {r.n_classes} classes,"
                     f" {r.cat_frac:.2f} categorical, minority {r.minority:.2f}")
    lines.append("  does a task property predict where you help? (Spearman rho with delta)")
    for k in ("n_features", "n_classes", "cat_frac", "minority"):
        if df[k].nunique() > 1:
            rho, p = stats.spearmanr(df[k], df.delta)
            lines.append(f"    {k:<12s} rho {rho:+.2f} (p={p:.2f})")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────── the agent

def propose(k, dist_txt, down_txt, history, prev_specs):
    hist = "\n".join(f"  round {h['round']}: gap {h['gap']:.3f}, DEV {h['dev']:+.4f} vs base"
                     for h in history) or "  (first round)"
    prev = json.dumps([{"cfg_edits": s.get("edits", {}), "has_realiser": bool(s.get("code"))}
                       for s in prev_specs], indent=1) if prev_specs else "  (none)"

    prompt = f"""You are repairing the synthetic PRIOR of a frozen tabular foundation model (TabICL). You
cannot change the model. You choose the distribution of synthetic datasets it is LoRA-adapted
to. A mixture of your {k} priors is trained round-robin into ONE adapter, then scored on real
tables you never see.

YOUR OBJECTIVE IS THE DOWNSTREAM AUC. NOTHING ELSE.

Read this carefully, because it inverts the obvious strategy. We measured it:

    round 0:  distribution gap 0.616  ->  downstream -0.0097
    round 1:  distribution gap 0.458  ->  downstream -0.0117
              (you closed the gap, and the model got WORSE)

Making your synthetic tables look more like real tables has, so far, made the model worse at
real tables. The released prior's *unrealism* appears to be load-bearing -- it is domain
randomisation, and narrowing onto realism throws away coverage the model was relying on.

So the distribution gap below is DIAGNOSTIC, not a target. Do not chase it. Chase the
downstream number. If realism is hurting, then generate priors that are harder, broader or
stranger -- whatever the downstream evidence says works. Let the AUC tell you what to do,
even when it contradicts the distribution report.

EVIDENCE -- distributional
{dist_txt}

EVIDENCE -- downstream
{down_txt}

HISTORY
{hist}

WHAT YOU PROPOSED LAST ROUND
{prev}

KNOBS (hard bounds, clipped)
{json.dumps({kk: [v[0], v[1], v[2].__name__] for kk, v in pt.KNOBS.items()}, indent=1)}

Each prior is a config PLUS an optional `realize(X, y, rng)` function that rewrites the
marginals of the SCM's output (monotone / rank-preserving maps only -- exp, powers,
quantile-binning, thresholding -- so the causal signal survives; never permute values;
numpy is `np`; return the same shape; never touch y).

Propose {k} priors that are DIFFERENT FROM EACH OTHER. Say in one sentence what each is for.

JSON: {{"priors": [{{"cfg": {{...all knobs...}}, "code": "def realize(X, y, rng):\\n    ...", "role": "<what this one covers>"}}, ...]}}"""

    r = llm.ask_json(prompt, {})
    specs = r.get("priors") if isinstance(r, dict) else None
    if not isinstance(specs, list) or not specs:
        return None
    out = []
    for s in specs[:k]:
        if not isinstance(s, dict):
            continue
        cfg = pt.clip_cfg(s.get("cfg", {}))
        code = s.get("code", "")
        fn, msg = realism.compile_realizer(code) if code else (None, "no realiser")
        out.append(dict(cfg=cfg, code=(code if fn else ""), role=str(s.get("role", ""))[:80],
                        realiser_ok=bool(fn), realiser_msg=msg,
                        edits={kk: v for kk, v in cfg.items()
                               if not np.isclose(float(v), float(pt.BASE_CFG[kk]), rtol=1e-3)}))
    return out or None


def random_specs(k, rng):
    """The control. Same K priors, same knob space, same compute — no LLM anywhere."""
    return [dict(cfg=random_cfg(rng), code="", role="random", realiser_ok=False,
                 edits={}) for _ in range(k)]


# ────────────────────────────────────────────────────────────────────────── the loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["agent", "random", "anchored"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    # Small values make a SMOKE run: enough to prove the loop closes end to end, not enough
    # to measure anything. Anything reported as a result must use the full budget.
    ap.add_argument("--steps-per-prior", type=int, default=STEPS_PER_PRIOR)
    ap.add_argument("--buffer", type=int, default=192)
    ap.add_argument("--audit-n", type=int, default=12)
    a = ap.parse_args()

    out = HERE / f"agentloop_{a.arm}_{a.seed}.json"
    rng = np.random.default_rng(a.seed)
    dev, test = tabarena.load_split(a.seed)
    real_meta = A2.real_meta(a.seed)

    # The loop may only ever see DEV tasks' VALIDATION rows. Their test rows are the
    # answer, and an agent that steers on the answer is just overfitting it slowly.
    base_dev = evaluate(None, dev, a.seed, alpha=ALPHA, split="val")
    base_dev_mean = float(np.mean(list(base_dev.values())))
    print(f"[{a.arm} seed {a.seed}] released checkpoint: DEV(val) {base_dev_mean:.4f}\n", flush=True)

    # Compute is matched to the arm, not to the agent: K priors, STEPS_PER_PRIOR each.
    steps = a.steps_per_prior * a.k
    history, specs = [], None

    for r in range(a.rounds):
        t0 = time.time()
        print(f"── round {r} ──────────────────────────────────", flush=True)

        if a.arm == "random":
            specs = random_specs(a.k, rng)
        elif a.arm == "anchored":
            # Keep TabICL's own prior in the mixture and ADD to it, rather than replacing it.
            # Every arm so far threw the released prior away -- and every arm lost. The thing
            # we kept measuring is that its breadth is load-bearing, so stop discarding it.
            if r == 0:
                gap, dist_txt = distribution_report(
                    [dict(cfg=dict(pt.BASE_CFG), code="")], real_meta, a.seed, n=a.audit_n)
                down_txt = (f"Downstream: the released checkpoint scores DEV "
                            f"{base_dev_mean:.4f}. You have not changed anything yet.")
            else:
                gap, dist_txt = history[-1]["gap"], history[-1]["dist_txt"]
                down_txt = history[-1]["down_txt"]
            new = propose(a.k - 1, dist_txt, down_txt, history,
                          specs[1:] if specs else None)
            if new is None:
                new = random_specs(a.k - 1, rng)
            # the released prior is the anchor; the agent's priors are additions to it
            specs = [dict(cfg=dict(pt.BASE_CFG), code="", role="TabICL's own prior (anchor)",
                          realiser_ok=False, edits={})] + new
            for i, sp in enumerate(specs):
                print(f"  prior {i}: {sp['role']}  "
                      f"realiser={'ok' if sp['realiser_ok'] else 'none'}"[:120], flush=True)
        else:
            if r == 0:
                gap, dist_txt = distribution_report(
                    [dict(cfg=dict(pt.BASE_CFG), code="")], real_meta, a.seed, n=a.audit_n)
                down_txt = ("Downstream: the released checkpoint scores DEV "
                            f"{base_dev_mean:.4f}. You have not changed anything yet.")
            else:
                gap, dist_txt = history[-1]["gap"], history[-1]["dist_txt"]
                down_txt = history[-1]["down_txt"]
            new = propose(a.k, dist_txt, down_txt, history, specs)
            if new is None:
                print("  agent produced nothing; falling back to random for this round",
                      flush=True)
                new = random_specs(a.k, rng)
            specs = new
            for i, s in enumerate(specs):
                print(f"  prior {i}: {s['role']}  realiser={'ok' if s['realiser_ok'] else 'none'}"
                      f" edits={json.dumps(s['edits'])}"[:150], flush=True)

        gap, dist_txt = distribution_report(specs, real_meta, a.seed, n=a.audit_n)
        st, loss = train_mixture(specs, a.seed, steps, n_batches=a.buffer)
        sc = evaluate(st, dev, a.seed, alpha=ALPHA, split="val")
        dm = float(np.mean(list(sc.values())))
        down_txt = downstream_report(dev, sc, base_dev)

        print(f"  gap {gap:.3f} | DEV {dm:.4f} ({dm - base_dev_mean:+.4f}) "
              f"| loss {loss:.3f} | {time.time()-t0:.0f}s", flush=True)

        history.append(dict(round=r, specs=specs, gap=gap, dev=dm - base_dev_mean,
                            dev_mean=dm, dev_scores=sc, loss=loss,
                            dist_txt=dist_txt, down_txt=down_txt,
                            secs=round(time.time() - t0, 1)))
        out.write_text(json.dumps(dict(arm=a.arm, seed=a.seed, k=a.k, steps=steps,
                                       base_dev=base_dev, base_dev_mean=base_dev_mean,
                                       history=[{kk: vv for kk, vv in h.items()
                                                 if kk not in ("dist_txt", "down_txt")}
                                                for h in history]),
                                  indent=1, default=float))

    # The final model is the LAST round's mixture, not the best-on-DEV one. Picking the best
    # round by DEV would be the argmax-on-validation move we are trying to avoid.
    final = history[-1]
    st, _ = train_mixture(final["specs"], a.seed, steps, n_batches=a.buffer)
    base_test = evaluate(None, test, a.seed, alpha=ALPHA, split="test")
    test_sc = evaluate(st, test, a.seed, alpha=ALPHA, split="test")
    bt = float(np.mean(list(base_test.values())))
    tt = float(np.mean(list(test_sc.values())))
    wins = sum(test_sc[k] > base_test[k] for k in base_test)

    print(f"\n══ TabArena TEST ({len(test)} held-out tasks, scored once) ══")
    print(f"   released checkpoint : {bt:.4f}")
    print(f"   {a.arm:<18s}: {tt:.4f}  ({tt-bt:+.4f})  wins {wins}/{len(test)}")

    d = json.loads(out.read_text())
    d.update(base_test=base_test, test=test_sc, base_test_mean=bt, test_mean=tt,
             test_delta=tt - bt, test_wins=wins, n_test=len(test))
    out.write_text(json.dumps(d, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
