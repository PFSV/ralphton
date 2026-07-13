# Representation Learning Lab

One researcher, one GPU, one output: **findings about persistent representations**, at ICLR quality.

Read first, and treat as binding:
- `CONSTITUTION.md` — mission, evidence rule, what never changes.
- `QUESTIONS.md` — the six permanent questions. Every experiment names the one it moves, or is out
  of scope.

## Repository map

| Path | What it is |
|---|---|
| `laion_data/` | Active line of work: representational alignment between LLMs and human visual cortex (re:vision / Doerig et al., Nat. Mach. Intell. 2025). Code `src/`, tests `tests/`, findings `reports/`. |
| `runs/` | The run ledger. One directory per launched experiment, plus append-only `registry.jsonl`. This is the experiment tracker; there is no other. |
| `reports/claims.yaml` | The claims ledger. Claim id → status → the run ids that are its evidence. The lab's state lives here, not in conversation. |
| `.claude/agents/` | The six roles below. |
| `.claude/skills/` | `/experiment-design`, `/literature-review`, `/referee`. |
| `.claude/hooks/` | Enforcement: GPU pin, evidence rule, session state. |
| `tabagent/` | **Archived. Out of scope.** Prior LLM-agent work (ICML). This OS does not govern it; do not extend it. Left on disk for the paper only. |

## The lab

You are the director. The researcher decides what to work on; these six do the work.

| Agent | Job | Never |
|---|---|---|
| `cartographer` | Prior art, what a paper really claims and measures, whether our result is new | Writes code; cites from memory |
| `theorist` | Question → pre-registered design: hypothesis, null, falsification criterion, seed budget, instrument risk | Runs anything; sees the result first |
| `experimentalist` | Builds the pipeline, registers the run, pins GPU 1, launches, records the exit code | Designs its own experiment; interprets its own result |
| `metrologist` | Re-derives every number from raw artifacts; interrogates the metric that produced it | Modifies code or results |
| `referee` | Attacks whether the evidence licenses the claim | Modifies anything; softens an objection |
| `archivist` | Ledger, reports, retractions | Certifies; launches |

**Producer and judge are never the same agent.** Nothing certifies itself.

## Hard rules

**Evidence rule. YOU MUST NOT write a number you did not read from a file during this session.**
Every number in a report, figure, or paper traces to a file under `runs/` with a run id and a git
SHA. If a number is not in a file, it does not exist. Do not estimate, recall, or interpolate one.

**No null, no result.** Every effect is reported against a control that could have produced it by
accident, and with its seed spread. A mean without its spread is not a number.

**The metric is a hypothesis.** Report which instrument, preprocessing, and noise ceiling produced
the effect, and whether it survives an alternative.

**A claim is a row in `reports/claims.yaml`**, not a memory. Status changes ship in the same commit
as their evidence. Retractions are recorded, never deleted.

**Never write to `laion_data/data/` (118 GB, not regenerable), `.venv/`, or built `*.pdf`.**

**GPU 0 belongs to another user. Only GPU 1 is ours.** Every job pins its device; the hook blocks
what does not.

**Failures must be loud.** A run that cannot reach its data, weights, or GPU exits non-zero. A
silent degradation to random output has already cost this lab one retracted result.

## Environment

- Python: `/home/snucsnl/ralphton/.venv/bin/python`. No `pyproject.toml`; deps installed ad hoc.
- Long jobs detach: `CUDA_VISIBLE_DEVICES=1 setsid nohup <cmd> < /dev/null > runs/<id>/stdout.log 2>&1 &`
- No wandb, no tensorboard. `runs/` is the tracker.

## Workflow

1. **Question** — name the permanent question. If there is none, stop.
2. **Read** — `cartographer`. What is already known, and what would make our result unsurprising.
3. **Design** — `theorist`. Hypothesis, null, falsification criterion, seeds, instrument, cost.
4. **Build and run** — `experimentalist`. Registered before launched, pinned to GPU 1.
5. **Verify** — `metrologist`. Re-derive from raw. Then `referee`. Attack until it holds or dies.
6. **Record** — `archivist`. Ledger and report, in one commit with the evidence.

Prefer the retraction to the defense. Claims have been struck here after audit; that is the process
working, not failing.

## Research policy

Tools over recall. Search arXiv, Semantic Scholar, and OpenAlex before asserting related work; read
repositories with the GitHub tools rather than guessing their contents.
