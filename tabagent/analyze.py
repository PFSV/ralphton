"""Turn stage1.jsonl into the numbers the paper prints. Nothing here invents a value:
every cell traces back to a run in the jsonl.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import priortrain as pt

HERE = Path(__file__).parent
ROWS = [json.loads(l) for l in (HERE / "stage1.jsonl").read_text().splitlines() if l.strip()]
ARMS = ["pretrained", "base", "random", "agent"]
LABEL = {"pretrained": "TabICLv2 (released, no adaptation)",
         "base": "LoRA on TabICL's own prior",
         "random": "LoRA + random prior search",
         "agent": "LoRA + agent-revised prior (ours)"}

by = defaultdict(list)
for r in ROWS:
    by[r["arm"]].append(r)

# task names come from the runs themselves, so this works for either suite
TEST_TASKS = sorted({t for rs in by.values() for r in rs for t in r["test"]})
DEV_TASKS = sorted({t for rs in by.values() for r in rs for t in r["dev"]})

print("=" * 96)
print(f"{'arm':<38s} {'seeds':>5s} {'DEV mean':>12s} {'TEST mean':>16s} {'LoRA runs':>10s} {'LLM':>5s}")
print("=" * 96)
summary = {}
for a in ARMS:
    rs = by.get(a, [])
    if not rs:
        continue
    dev = np.array([np.mean(list(r["dev"].values())) for r in rs])
    test = np.array([r["test_mean"] for r in rs])
    runs = np.mean([r.get("lora_runs", 0) for r in rs])
    calls = np.mean([r.get("llm_calls", 0) for r in rs])
    summary[a] = dict(dev=dev, test=test)
    print(f"{LABEL[a]:<38s} {len(rs):>5d} {dev.mean():>7.4f}±{dev.std():<4.4f} "
          f"{test.mean():>9.4f}±{test.std():<5.4f} {runs:>10.0f} {calls:>5.0f}")
print("=" * 96)

if "pretrained" in summary:
    b = summary["pretrained"]["test"].mean()
    print("\nTEST delta vs released checkpoint:")
    for a in ARMS[1:]:
        if a in summary:
            d = summary[a]["test"] - summary["pretrained"]["test"]
            print(f"  {LABEL[a]:<40s} {d.mean():+.4f} ± {d.std():.4f}  (per seed: {np.round(d,4)})")

if "agent" in summary and "random" in summary:
    d = summary["agent"]["test"] - summary["random"]["test"]
    print(f"\nagent - random (same search budget, LLM is the only difference):")
    print(f"  {d.mean():+.4f} ± {d.std():.4f}   per seed {np.round(d, 4)}")

# per-task test AUC
print("\nPer-task TEST AUC (mean over seeds):")
hdr = f"{'task':<22s}" + "".join(f"{a[:9]:>11s}" for a in ARMS if a in summary)
print(hdr)
for t in TEST_TASKS:
    line = f"{t:<22s}"
    for a in ARMS:
        if a not in summary:
            continue
        v = np.mean([r["test"][t] for r in by[a]])
        line += f"{v:>11.4f}"
    print(line)

# what did the agent actually change?
print("\nKnobs the agent moved (winning config, per seed):")
for r in by.get("agent", []):
    print(f"  seed {r['seed']} (best round {r.get('best_round')}): "
          f"{json.dumps(r.get('best_diff', {}))}")

# search trajectory
print("\nDEV mean by round (search efficiency):")
for a in ("random", "agent"):
    for r in by.get(a, []):
        traj = [round(h["dev_mean"], 4) for h in r.get("history", [])]
        print(f"  {a:<7s} seed {r['seed']}: {traj}")

json.dump({a: {"dev": summary[a]["dev"].tolist(), "test": summary[a]["test"].tolist()}
           for a in summary}, (HERE / "summary.json").open("w"), indent=1)
print(f"\nwrote {HERE/'summary.json'}")
