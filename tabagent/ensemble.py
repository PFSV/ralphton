"""Do not argmax on validation. Ensemble.

A search loop that picks the single candidate with the best DEV score is running gradient
descent on the validation set with extra steps, and it shows: full fine-tuning gained +0.0042
on DEV and gave back -0.0017 on TEST. The DEV signal is real but it is not free to consume.

So use DEV to *weight*, never to *select*:

  argmax   the one best DEV prior                                   (the thing we distrust)
  soup     average the LoRA weights across candidates, DEV-weighted (model soup)
  vote     average the predicted probabilities across candidates    (classic ensembling)
  mixture  train ONE adapter on a mixture of the candidate priors   (Mitra's argument:
           diversity of the prior beats fidelity of any single one)

All four are scored on the same held-out TEST tasks. If the user's hypothesis is right,
argmax is the worst of the four and mixture or soup is the best.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import priortrain as pt
import tabarena
from stage1 import evaluate

HERE = Path(__file__).parent


# ───────────────────────────────────────────────────────────────── the four combiners

def soup(states: list[dict], weights: np.ndarray) -> dict:
    """Weight-average the adapters themselves. Only valid because every adapter starts from
    the same frozen backbone, so they live in one basin."""
    keys = states[0].keys()
    return {k: sum(w * s[k].float() for w, s in zip(weights, states)) for k in keys}


def vote_predict(states, task, seed, weights, alpha=32):
    """Average the probabilities each adapter assigns. More expensive than soup at inference,
    but it does not assume the adapters are averageable."""
    P = None
    for w, st in zip(weights, states):
        m = pt.LoRATabICL(st, seed=seed, alpha=alpha).fit(task.X_ctx.to_numpy(), task.y_ctx)
        p = m.predict_proba(task.X_test.to_numpy())
        P = w * p if P is None else P + w * p
    return P


class MixturePrior:
    """Round-robin over several priors, so one adapter sees all of them."""

    def __init__(self, cfgs, batch_size):
        self.priors = [iter(pt.make_prior(c, batch_size=batch_size)) for c in cfgs]
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        p = self.priors[self.i % len(self.priors)]
        self.i += 1
        return next(p)


def train_on_mixture(cfgs, steps, lr, seed, batch_size=4, alpha=32):
    """Same total step budget as any single-prior arm — the mixture gets no extra compute."""
    import torch.nn.functional as F

    clf = pt.load_backbone(seed=seed)
    model = clf.model_
    params = pt.inject_lora(model, alpha=alpha)
    model.to(pt.DEV).train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    it = iter(MixturePrior(cfgs, batch_size))
    losses = []
    for _ in range(steps):
        X, y, d, seq_len, train_size = next(it)
        ts = int(train_size[0].item()) if torch.is_tensor(train_size) else int(train_size)
        X, y = X.to(pt.DEV), y.to(pt.DEV)
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
    return pt.lora_state(model), float(np.mean(losses[-20:]))


def score_states(state, tasks, seed, alpha=32):
    s = evaluate(state, tasks, seed, alpha=alpha)
    return float(np.mean(list(s.values()))), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    # Defaults are the only setting the diagnostic found that actually helps: a HIGH LoRA
    # scale (alpha 128) lets the adapter have its effect with a *small* B, so it does not
    # trample the pre-trained weights. At alpha 32 the same training costs -0.010 DEV; at
    # alpha 128 it gains +0.002. Ensembling an adapter that is already destructive would
    # only tell us about the damage.
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--candidates", type=int, default=5)
    a = ap.parse_args()

    dev, test = tabarena.load_split(a.seed)
    rng = np.random.default_rng(a.seed)

    # candidate priors: TabICL's own, plus perturbations toward what real tables look like.
    # (When close_gap.py has produced an LLM realiser, its configs go here instead.)
    from stage1 import random_cfg
    cfgs = [dict(pt.BASE_CFG)] + [random_cfg(rng) for _ in range(a.candidates - 1)]

    print(f"seed {a.seed}: {len(cfgs)} candidate priors, {a.steps} steps each, lr {a.lr}\n",
          flush=True)

    states, dev_means = [], []
    for i, c in enumerate(cfgs):
        t0 = time.time()
        clf = pt.load_backbone(seed=a.seed)
        clf, st = pt.train_lora(clf, c, steps=a.steps, batch_size=4, lr=a.lr,
                                alpha=a.alpha, seed=a.seed, log_every=0)
        s = pt.lora_state(clf.model_)
        dm, _ = score_states(s, dev, a.seed, a.alpha)
        states.append(s)
        dev_means.append(dm)
        print(f"  candidate {i}: DEV {dm:.4f}  (loss {st.final_loss:.3f}, "
              f"{time.time()-t0:.0f}s)", flush=True)
        del clf

    dev_means = np.array(dev_means)
    base_dev, _ = score_states(None, dev, a.seed, a.alpha)
    base_test, base_test_per = score_states(None, test, a.seed, a.alpha)
    print(f"\nreleased checkpoint: DEV {base_dev:.4f}  TEST {base_test:.4f}\n", flush=True)

    # DEV may set the weights; it may not pick the winner. But a sharp temperature turns a
    # weighted ensemble back INTO argmax -- exp(-0.02/0.005) = 0.018 is a one-hot in
    # disguise -- so report the uniform ensemble too. If DEV-weighting beats uniform, the
    # weighting is earning its keep; if it merely reproduces argmax, we have learned that.
    tau = 0.02
    w = np.exp((dev_means - dev_means.max()) / tau)
    w /= w.sum()
    u = np.full(len(states), 1.0 / len(states))
    print(f"DEV-weighted (tau={tau}): {np.round(w, 3).tolist()}", flush=True)
    print(f"uniform               : {np.round(u, 3).tolist()}", flush=True)
    if w.max() > 0.9:
        print("  !! DEV weights have collapsed to one-hot; 'soup'/'vote' == 'argmax'",
              flush=True)
    print(flush=True)

    results = {}

    i_best = int(dev_means.argmax())
    tm, per = score_states(states[i_best], test, a.seed, a.alpha)
    results["argmax"] = dict(test=tm, per_task=per, note=f"candidate {i_best}")
    print(f"argmax    (best on DEV, cand {i_best})     : TEST {tm:.4f}  ({tm-base_test:+.4f})",
          flush=True)

    for tag, ww in (("soup-dev", w), ("soup-unif", u)):
        tm, per = score_states(soup(states, ww), test, a.seed, a.alpha)
        results[tag] = dict(test=tm, per_task=per)
        print(f"{tag:<9s} (LoRA weight average)     : TEST {tm:.4f}  ({tm-base_test:+.4f})",
              flush=True)

    for tag, ww in (("vote-dev", w), ("vote-unif", u)):
        per = {}
        for t in test:
            P = vote_predict(states, t, a.seed, ww, alpha=a.alpha)
            if t.n_classes == 2 and P.shape[1] == 2:
                per[t.name] = float(roc_auc_score(t.y_test, P[:, 1]))
            else:
                Q = P / P.sum(1, keepdims=True).clip(1e-9)
                per[t.name] = float(roc_auc_score(t.y_test, Q, multi_class="ovr",
                                                  average="macro",
                                                  labels=list(range(t.n_classes))))
        tm = float(np.mean(list(per.values())))
        results[tag] = dict(test=tm, per_task=per)
        print(f"{tag:<9s} (probability ensemble)    : TEST {tm:.4f}  ({tm-base_test:+.4f})",
              flush=True)

    # Compute-match the mixture. soup/vote/argmax all draw on 5 adapters x `steps` each, so
    # giving the mixture only `steps` would make it lose on budget, not on merit.
    mix_steps = a.steps * len(cfgs)
    st_mix, loss = train_on_mixture(cfgs, mix_steps, a.lr, a.seed, alpha=a.alpha)
    tm, per = score_states(st_mix, test, a.seed, a.alpha)
    results["mixture"] = dict(test=tm, per_task=per, loss=loss, steps=mix_steps)
    print(f"mixture   (1 adapter, all {len(cfgs)} priors, {mix_steps} steps)"
          f" : TEST {tm:.4f}  ({tm-base_test:+.4f})", flush=True)

    out = HERE / f"ensemble_{a.seed}.json"
    out.write_text(json.dumps(dict(seed=a.seed, base_dev=base_dev, base_test=base_test,
                                   base_test_per=base_test_per, dev_means=dev_means.tolist(),
                                   weights=w.tolist(), results=results), indent=1, default=float))
    order = sorted(results.items(), key=lambda kv: -kv[1]["test"])
    print(f"\nranking on TEST: " + " > ".join(f"{k} {v['test']:.4f}" for k, v in order))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
