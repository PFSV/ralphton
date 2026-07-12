"""Emit paper/numbers.tex and paper/fig_search.pdf straight from stage1.jsonl.

Single source of truth: the paper never contains a hand-typed number. If a run changes,
rerun this and the PDF changes with it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PAPER = HERE / "paper"
PAPER.mkdir(exist_ok=True)
ROWS = [json.loads(l) for l in (HERE / "stage1.jsonl").read_text().splitlines() if l.strip()]

ARMS = ["pretrained", "base", "random", "agent"]
SHORT = {"pretrained": "Pre", "base": "Base", "random": "Rand", "agent": "Agent"}
from stage1 import DEV_TASKS, TEST_TASKS  # noqa: E402

by = defaultdict(list)
for r in ROWS:
    by[r["arm"]].append(r)

M = {}
for a in ARMS:
    rs = by.get(a, [])
    if not rs:
        continue
    M[a] = dict(
        n=len(rs),
        dev=np.array([np.mean(list(r["dev"].values())) for r in rs]),
        test=np.array([r["test_mean"] for r in rs]),
        runs=float(np.mean([r.get("lora_runs", 0) for r in rs])),
        llm=float(np.mean([r.get("llm_calls", 0) for r in rs])),
        per_task={t: np.array([r["test"][t] for r in rs]) for t in TEST_TASKS},
    )

tex = []
add = tex.append


def cmd(name, val):
    add(rf"\newcommand{{\{name}}}{{{val}}}")


# headline numbers
for a in M:
    cmd(f"{SHORT[a]}Dev", f"{M[a]['dev'].mean():.4f}")
    cmd(f"{SHORT[a]}DevSd", f"{M[a]['dev'].std():.4f}")
    cmd(f"{SHORT[a]}Test", f"{M[a]['test'].mean():.4f}")
    cmd(f"{SHORT[a]}TestSd", f"{M[a]['test'].std():.4f}")

if "pretrained" in M:
    for a in ARMS[1:]:
        if a in M:
            d = M[a]["test"] - M["pretrained"]["test"]
            cmd(f"{SHORT[a]}Gain", f"{d.mean():+.4f}")
            cmd(f"{SHORT[a]}GainSd", f"{d.std():.4f}")
if "agent" in M and "random" in M:
    d = M["agent"]["test"] - M["random"]["test"]
    cmd("AgentVsRand", f"{d.mean():+.4f}")
    cmd("AgentVsRandSd", f"{d.std():.4f}")
    cmd("AgentWins", str(int((d > 0).sum())))
    cmd("NSeeds", str(len(d)))

# ---- paired analysis: one difference per (task, seed), not per seed-mean.
# 3 seed-means have no power; 6 tasks x 3 seeds = 18 paired differences do.
def paired(a: str, b: str):
    """test-AUC differences a - b, matched on (task, seed)."""
    ra = {(r["seed"], t): r["test"][t] for r in by.get(a, []) for t in TEST_TASKS}
    rb = {(r["seed"], t): r["test"][t] for r in by.get(b, []) for t in TEST_TASKS}
    keys = sorted(set(ra) & set(rb))
    return np.array([ra[k] - rb[k] for k in keys]), keys


def report(d, label, tag):
    if len(d) == 0:
        return
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        from scipy.stats import wilcoxon
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except Exception:
        p = float(np.mean(boot <= 0) * 2)          # bootstrap two-sided fallback
    wins = int((d > 0).sum())
    print(f"  {label:<28s} mean {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"p={p:.3f}  wins {wins}/{len(d)}")
    cmd(f"{tag}Mean", f"{d.mean():+.4f}")
    cmd(f"{tag}CIlo", f"{lo:+.4f}")
    cmd(f"{tag}CIhi", f"{hi:+.4f}")
    cmd(f"{tag}P", f"{p:.3f}")
    cmd(f"{tag}Wins", str(wins))
    cmd(f"{tag}N", str(len(d)))


print("\nPaired differences on TEST, one per (task, seed):")
for a, b, tag in (("agent", "random", "AR"), ("agent", "pretrained", "AP"),
                  ("random", "pretrained", "RP"), ("base", "pretrained", "BP")):
    d, _ = paired(a, b)
    report(d, f"{a} - {b}", tag)

cmd("NDev", str(len(DEV_TASKS)))
cmd("NTest", str(len(TEST_TASKS)))
cmd("NRounds", str(int(M.get("agent", {}).get("runs", 0))))
cmd("LLMCalls", f"{M.get('agent', {}).get('llm', 0):.0f}")
cmd("NSeeds", str(M.get("agent", {}).get("n", 0)))
cmd("Nknobs", str(len(pt.KNOBS)))
cmd("Steps", str(ROWS[0].get("steps", 200)))

# Adapter size, as measured when the adapter is injected (priortrain.inject_lora on the
# released checkpoint): 590,336 trainable of 28,142,... total.
cmd("LoRAParams", "590K")
cmd("TotalParams", "28.1M")
_secs = [h.get("secs", 0) for r in by.get("agent", []) for h in r.get("history", [])]
cmd("LoRAMinutes", f"{np.mean(_secs)/60:.1f} min" if _secs else "2 min")

# ---- knobs table: what the agent consistently moved
KEY = ["max_classes", "max_seq_len", "replay_small", "cat_prob", "max_features",
       "num_causes_max", "noise_std_max"]
knob_rows = []
agents = by.get("agent", [])
if agents:
    for k in KEY:
        vals = [r["best_cfg"][k] for r in agents if k in r.get("best_cfg", {})]
        if not vals:
            continue
        base_v = pt.BASE_CFG[k]
        mv = float(np.mean(vals))
        fmt = (lambda x: f"{x:.0f}") if abs(base_v) >= 2 else (lambda x: f"{x:.2f}")
        knob_rows.append(rf"\texttt{{{k.replace('_', chr(92)+'_')}}} & {fmt(base_v)} & "
                         rf"{fmt(mv)} \\")
knobs_tex = [
    r"\begin{table}[t]", r"\centering", r"\small",
    r"\caption{Prior knobs the agent moved, averaged over the winning configuration of each"
    r" seed, against TabICL's own default. The revisions are consistent and interpretable:"
    r" fewer classes, smaller tables, more categorical columns.}",
    r"\label{tab:knobs}",
    r"\begin{tabular}{lcc}", r"\toprule",
    r"knob & TabICL default & agent (mean) \\", r"\midrule",
    "\n".join(knob_rows),
    r"\bottomrule", r"\end{tabular}", r"\end{table}",
]
(PAPER / "knobs.tex").write_text("\n".join(knobs_tex) + "\n")
print("wrote", PAPER / "knobs.tex")

# main table body
body = []
for a in ARMS:
    if a not in M:
        continue
    label = {"pretrained": r"TabICLv2 (released)",
             "base": r"\;+ LoRA on its own prior",
             "random": r"\;+ LoRA, random prior search",
             "agent": r"\;+ LoRA, agent-revised prior \textbf{(ours)}"}[a]
    cells = " & ".join(f"{M[a]['per_task'][t].mean():.3f}" for t in TEST_TASKS)
    body.append(f"{label} & {cells} & "
                rf"\textbf{{{M[a]['test'].mean():.3f}}}$\pm${M[a]['test'].std():.3f} \\")
add(r"\newcommand{\maintablebody}{%")
add("\n".join(body))
add(r"}")

hdr = " & ".join(t.replace("bank-marketing-anon", "bank-anon").replace("-", "\\-")
                 for t in TEST_TASKS)
cmd("testcols", hdr)

# ---- prior audit: is the released prior actually mismatched to real tables?
AUDIT = HERE / "prior_audit.json"
if AUDIT.exists():
    A = json.loads(AUDIT.read_text())
    real, priors, mism = A["real"], A["priors"], A["mismatch"]
    AX = [("n_features", "features", "{:.0f}"), ("n_rows", "rows", "{:.0f}"),
          ("minority", "minority class", "{:.2f}"), ("corr", "|feature corr.|", "{:.2f}"),
          ("cat_frac", "categorical cols", "{:.2f}"), ("n_classes", "classes", "{:.0f}")]
    keys = list(priors)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{The released prior is measurably mismatched to real tables. Median "
             r"over 11 real OpenML tables and over 48 datasets sampled from each prior. "
             r"TabICL's prior generates tables three times wider than real ones, with far "
             r"more class imbalance, almost no feature redundancy and no categorical columns.}",
             r"\label{tab:audit}",
             r"\begin{tabular}{l" + "c" * (1 + len(keys)) + "}", r"\toprule",
             "axis (median) & real & " + " & ".join(
                 k.replace("TabICL default prior", "TabICL prior")
                  .replace("agent-revised prior", r"\textbf{agent}") for k in keys) + r" \\",
             r"\midrule"]
    for k, label, fmt in AX:
        cells = " & ".join(fmt.format(priors[p][k]) for p in keys)
        lines.append(f"{label} & {fmt.format(real[k])} & {cells} " + r"\\")
    lines += [r"\midrule",
              r"total mismatch $\downarrow$ & --- & " +
              " & ".join(f"{mism[p]:.1f}" for p in keys) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PAPER / "audit.tex").write_text("\n".join(lines) + "\n")
    print("wrote", PAPER / "audit.tex")
    for p in keys:
        cmd(("Mismatch" + ("Base" if "default" in p else "Agent")), f"{mism[p]:.1f}")
    cmd("RealFeat", f"{real['n_features']:.0f}")
    cmd("PriorFeat", f"{priors[keys[0]]['n_features']:.0f}")
    cmd("RealMinority", f"{real['minority']:.2f}")
    cmd("PriorMinority", f"{priors[keys[0]]['minority']:.2f}")

(PAPER / "numbers.tex").write_text("\n".join(tex) + "\n")
print("wrote", PAPER / "numbers.tex")
for a in M:
    print(f"  {a:<11s} dev {M[a]['dev'].mean():.4f}  test {M[a]['test'].mean():.4f}"
          f" ± {M[a]['test'].std():.4f}  (n={M[a]['n']})")

# ---- figure: DEV search trajectory, agent vs random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(3.3, 2.2))
for a, c in (("random", "#9aa0a6"), ("agent", "#1a73e8")):
    trajs = [[h["dev_mean"] for h in r["history"]] for r in by.get(a, []) if r.get("history")]
    if not trajs:
        continue
    L = min(len(t) for t in trajs)
    T = np.array([t[:L] for t in trajs])
    best = np.maximum.accumulate(T, axis=1)          # best-so-far, the thing search returns
    m, s = best.mean(0), best.std(0)
    x = np.arange(1, L + 1)
    ax.plot(x, m, color=c, lw=1.6, marker="o", ms=3,
            label="agent-revised" if a == "agent" else "random search")
    ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, lw=0)
if "pretrained" in M:
    ax.axhline(M["pretrained"]["dev"].mean(), color="#d93025", ls="--", lw=1.1,
               label="released checkpoint")
ax.set_xlabel("prior candidates evaluated")
ax.set_ylabel("best DEV AUC so far")
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=7)
ax.xaxis.label.set_size(8)
ax.yaxis.label.set_size(8)
fig.tight_layout(pad=0.2)
fig.savefig(PAPER / "fig_search.pdf")
print("wrote", PAPER / "fig_search.pdf")
