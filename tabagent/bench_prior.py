"""Where is prior generation actually spending time, and does the GPU fix it?"""
import time

from tabicl.prior import PriorDataset
from tabicl.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
import copy

import priortrain as pt

HEAVY = dict(pt.BASE_CFG, max_seq_len=2048, min_seq_len=800, hidden_dim_max=256,
             num_layers_max=12, max_features=100, num_causes_max=30)


def bench(cfg, device, n_batches=6, bs=4, n_jobs=8, label=""):
    fixed = dict(DEFAULT_FIXED_HP)
    fixed["mix_probs"] = [cfg["mix_prob_mlp"], 1 - cfg["mix_prob_mlp"]]
    fixed["cat_prob"] = cfg["cat_prob"]
    sampled = copy.deepcopy(DEFAULT_SAMPLED_HP)
    sampled["num_causes"]["max_mean"] = cfg["num_causes_max"]
    sampled["num_layers"]["max_mean"] = cfg["num_layers_max"]
    sampled["hidden_dim"]["max_mean"] = cfg["hidden_dim_max"]
    p = PriorDataset(batch_size=bs, batch_size_per_gp=min(4, bs),
                     min_features=cfg["min_features"], max_features=cfg["max_features"],
                     max_classes=cfg["max_classes"],
                     min_seq_len=cfg["min_seq_len"], max_seq_len=cfg["max_seq_len"],
                     min_train_size=cfg["min_train_size"], max_train_size=cfg["max_train_size"],
                     replay_small=bool(cfg["replay_small"]), prior_type="mix_scm",
                     scm_fixed_hp=fixed, scm_sampled_hp=sampled,
                     n_jobs=n_jobs, device=device)
    it = iter(p)
    next(it)                                   # warm
    t0 = time.time()
    for _ in range(n_batches):
        next(it)
    dt = (time.time() - t0) / n_batches
    print(f"{label:<28s} device={device:<5s} n_jobs={n_jobs:<3d} {dt:6.2f} s/step "
          f"-> 200 steps = {dt*200/60:5.1f} min", flush=True)
    return dt


print("--- TabICL default prior ---")
bench(pt.BASE_CFG, "cpu", n_jobs=8, label="base")
bench(pt.BASE_CFG, "cuda", n_jobs=1, label="base")
print("--- worst-case random draw ---")
bench(HEAVY, "cpu", n_jobs=8, label="heavy (seq2048,h256,L12)")
bench(HEAVY, "cuda", n_jobs=1, label="heavy (seq2048,h256,L12)")
