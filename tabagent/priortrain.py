"""Stage 1: revise the prior, adapt the frozen TFM to it with LoRA.

TabICL was pre-trained on a synthetic prior of random structural causal models. That prior
is fixed at publication time and knows nothing about the tasks it will actually meet. Here
the prior itself becomes the object of search: an agent edits its generating configuration,
we LoRA-adapt the released checkpoint to the revised prior, and Stage 2 measures whether
downstream real tasks got better.

Only the LoRA parameters move (~0.5% of the model). The base weights, and the pre-training
that produced them, are never touched.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabicl import TabICLClassifier
from tabicl.prior import PriorDataset
from tabicl.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------- prior config

# The knobs the agent may edit, with hard bounds. Anything outside is clipped, so a bad
# proposal costs a training run but cannot produce an invalid prior. Bounds are identical for
# the random and the agent arm. max_seq_len is capped at 1024 because that is the sequence
# length TabICL itself pre-trained at, and because in-context attention is quadratic in it.
KNOBS: dict[str, tuple] = {
    # (low, high, kind)
    "min_features":     (2, 20, int),
    "max_features":     (20, 100, int),
    "max_classes":      (2, 10, int),
    "min_seq_len":      (200, 800, int),
    "max_seq_len":      (600, 1024, int),
    "min_train_size":   (0.05, 0.5, float),
    "max_train_size":   (0.5, 0.95, float),
    "mix_prob_mlp":     (0.0, 1.0, float),   # -> mix_probs = [p, 1-p]  (mlp vs tree SCM)
    "cat_prob":         (0.0, 0.6, float),   # fraction of columns made categorical
    "balanced":         (0, 1, bool),        # force balanced classes
    "tree_depth_lambda":       (0.1, 3.0, float),
    "tree_n_estimators_lambda": (0.1, 3.0, float),
    "num_causes_max":   (2, 30, int),        # width of the causal graph
    "num_layers_max":   (2, 12, int),        # depth of the SCM
    "hidden_dim_max":   (16, 256, int),
    "noise_std_max":    (0.001, 1.0, float),
    "init_std_max":     (0.5, 20.0, float),
    "dropout_scale":    (0.1, 2.0, float),
    "replay_small":     (0, 1, bool),        # over-sample tiny datasets
}

BASE_CFG: dict = {
    "min_features": 2, "max_features": 100, "max_classes": 10,
    "min_seq_len": 500, "max_seq_len": 1024,
    "min_train_size": 0.1, "max_train_size": 0.9,
    "mix_prob_mlp": 0.7, "cat_prob": 0.2, "balanced": 0,
    "tree_depth_lambda": 0.5, "tree_n_estimators_lambda": 0.5,
    "num_causes_max": 12, "num_layers_max": 6, "hidden_dim_max": 130,
    "noise_std_max": 0.3, "init_std_max": 10.0, "dropout_scale": 0.9,
    "replay_small": 0,
}


def clip_cfg(cfg: dict) -> dict:
    out = dict(BASE_CFG)
    for k, v in (cfg or {}).items():
        if k not in KNOBS:
            continue
        lo, hi, kind = KNOBS[k]
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        v = min(max(v, lo), hi)
        out[k] = int(round(v)) if kind in (int, bool) else float(v)
    out["max_features"] = max(out["max_features"], out["min_features"] + 5)
    out["max_seq_len"] = max(out["max_seq_len"], out["min_seq_len"] + 100)
    out["max_train_size"] = max(out["max_train_size"], out["min_train_size"] + 0.2)
    return out


def make_prior(cfg: dict, batch_size: int, n_jobs: int = 1,
               device: str | None = None) -> PriorDataset:
    """Sample the prior on the GPU. This is not a micro-optimisation: on CPU the sampling
    cost grows with SCM width/depth/length (1.2--1.4 s/step), so an expensive prior would be
    slow to *evaluate* as well as to train on, and wall-clock would confound the search. On
    GPU it is 0.16 s/step and flat across configurations, so every candidate prior costs the
    same to try, whoever proposed it."""
    cfg = clip_cfg(cfg)
    fixed = dict(DEFAULT_FIXED_HP)
    fixed["mix_probs"] = [cfg["mix_prob_mlp"], 1.0 - cfg["mix_prob_mlp"]]
    fixed["cat_prob"] = cfg["cat_prob"]
    fixed["balanced"] = bool(cfg["balanced"])
    fixed["tree_depth_lambda"] = cfg["tree_depth_lambda"]
    fixed["tree_n_estimators_lambda"] = cfg["tree_n_estimators_lambda"]

    sampled = copy.deepcopy(DEFAULT_SAMPLED_HP)
    sampled["num_causes"]["max_mean"] = cfg["num_causes_max"]
    sampled["num_layers"]["max_mean"] = cfg["num_layers_max"]
    sampled["hidden_dim"]["max_mean"] = cfg["hidden_dim_max"]
    sampled["noise_std"]["max_mean"] = cfg["noise_std_max"]
    sampled["init_std"]["max_mean"] = cfg["init_std_max"]
    sampled["mlp_dropout_prob"]["scale"] = cfg["dropout_scale"]

    return PriorDataset(
        batch_size=batch_size, batch_size_per_gp=min(4, batch_size),
        min_features=cfg["min_features"], max_features=cfg["max_features"],
        max_classes=cfg["max_classes"],
        min_seq_len=cfg["min_seq_len"], max_seq_len=cfg["max_seq_len"],
        min_train_size=cfg["min_train_size"], max_train_size=cfg["max_train_size"],
        replay_small=bool(cfg["replay_small"]),
        prior_type="mix_scm", scm_fixed_hp=fixed, scm_sampled_hp=sampled,
        n_jobs=n_jobs, device=device or DEV,
    )


# --------------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32,
                pattern: str = "icl_predictor.tf_icl") -> list[nn.Parameter]:
    """Adapt only the dataset-wise ICL transformer — the block that does the in-context
    inference. Column/row embedders keep their pre-trained behaviour."""
    for p in model.parameters():
        p.requires_grad_(False)
    # out_proj is excluded on purpose: the attention block reads `out_proj.weight`
    # directly (functional MHA), so it cannot be wrapped. The FFN projections can.
    targets = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.startswith(pattern)
               and n.split(".")[-1] in ("linear1", "linear2")]
    for name in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha))
    params = [p for p in model.parameters() if p.requires_grad]
    return params


# --------------------------------------------------------------------- training

@dataclass
class TrainStats:
    steps: int
    final_loss: float
    trainable: int
    total: int
    secs: float


def lora_state(model: nn.Module) -> dict:
    """Just the adapter — a few hundred KB, not a 28M-parameter checkpoint."""
    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


class LoRATabICL:
    """TabICLClassifier.fit() reloads the pre-trained checkpoint every call, which drops
    any adapter we injected. So we re-inject after each fit. Passing state=None gives the
    unmodified released model, which is exactly the no-LoRA baseline.
    """

    def __init__(self, state: dict | None, seed: int = 0, r: int = 16, alpha: int = 32,
                 adapter: str = "lora"):
        self.state, self.seed, self.r, self.alpha = state, seed, r, alpha
        self.adapter = adapter
        self.clf = None

    def fit(self, X, y):
        self.clf = TabICLClassifier(device=DEV, random_state=self.seed)
        self.clf.fit(X, y)
        if self.state:
            if self.adapter == "lora":
                inject_lora(self.clf.model_, r=self.r, alpha=self.alpha)
            missing, unexpected = self.clf.model_.load_state_dict(self.state, strict=False)
            assert not unexpected, f"adapter keys not in model: {unexpected[:3]}"
            self.clf.model_.to(DEV).eval()
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


def load_backbone(seed: int = 0):
    """The released TabICLv2 checkpoint, plus the sklearn wrapper that owns its
    preprocessing (we reuse the wrapper at eval time with the adapted weights)."""
    clf = TabICLClassifier(device=DEV, random_state=seed)
    clf.fit(np.random.randn(64, 5), np.random.randint(0, 2, 64))  # forces the load
    return clf


def train_lora(clf, cfg: dict, steps: int = 200, batch_size: int = 4,
               lr: float = 1e-4, r: int = 16, seed: int = 0, log_every: int = 50,
               adapter: str = "lora"):
    """adapter='lora' trains 2.1% of the model; adapter='full' unfreezes everything.

    Full fine-tuning is the honest control: if adaptation only ever hurts, we need to know
    whether that is LoRA's narrow surface or the fact that any further training on the prior
    disturbs a checkpoint that was already trained on it for far longer.
    """
    import time
    t0 = time.time()
    torch.manual_seed(seed)
    model = clf.model_
    if adapter == "lora":
        params = inject_lora(model, r=r)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
    model.to(DEV).train()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)

    prior = make_prior(cfg, batch_size=batch_size)
    it = iter(prior)
    losses: list[float] = []
    for step in range(steps):
        X, y, d, seq_len, train_size = next(it)
        ts = int(train_size[0].item()) if torch.is_tensor(train_size) else int(train_size)
        X, y = X.to(DEV), y.to(DEV)
        y_tr, y_te = y[:, :ts], y[:, ts:]
        if y_te.numel() == 0:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            # d (feature-group sizes) is rejected by the checkpoint's grouped embedder
            pred = model(X, y_tr, None)
            loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(),
                                   y_te.reshape(-1).long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step+1}/{steps} loss {np.mean(losses[-log_every:]):.4f}",
                  flush=True)

    model.eval()
    return clf, TrainStats(steps, float(np.mean(losses[-20:])) if losses else float("nan"),
                           trainable, total, round(time.time() - t0, 1))
