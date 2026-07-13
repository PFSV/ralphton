# AI Research Lab

Two research projects share this repository, one venv, and one GPU. Findings, not code volume, are the output.

## Repository map

| Path | What it is |
|---|---|
| `tabagent/` | LLM agent that repairs tabular priors. Entrypoints: `agent_loop.py`, `context_bench.py`. Drivers: `one.sh`, `fleet.sh`, `full.sh`, `grid.sh`. Paper: `tabagent/paper/` (ICML). |
| `laion_data/` | re:vision reproduction (Doerig et al., Nat. Mach. Intell. 2025). Code in `src/`, tests in `tests/`, findings in `reports/`. |
| `runs/` | Run ledger. One directory per launched experiment, plus append-only `registry.jsonl`. |
| `reports/claims.yaml` | Claims ledger: claim id -> status (open/reproduced/failed/retracted) -> evidence run ids. |
| `.claude/rules/` | Path-scoped conventions. Loaded when you touch matching files. |
| `.claude/skills/` | Lab workflows: `/new-run`, `/audit-claim`, `/retract`, `/lit-review`, `/figure`, `/status`. |
| `.claude/agents/` | Delegation targets, including `claim-auditor` (adversarial). |

## Hard rules

**Evidence rule. YOU MUST NOT write a number you did not read from a file during this session.**
Every number in a report, README, figure, or paper must be traceable to a file under `runs/`.
If a number is not in a file, it does not exist. Do not estimate, recall, or interpolate one.

**A claim is a row in `reports/claims.yaml`, not a memory.** Changing a claim's status means editing that file in the same commit as the evidence.

**Never write to `laion_data/data/` (118 GB, not regenerable), `.venv/`, or built `*.pdf` artifacts.**

**GPU 0 belongs to another user. Only GPU 1 is ours.** Every CUDA job pins its device explicitly; never launch unpinned.

**Failures must be loud.** A run that cannot reach its LLM, its data, or its GPU must exit non-zero. A silent degradation to random output already cost this lab one retracted result.

## Environment

- Python: `/home/snucsnl/ralphton/.venv/bin/python`, shared by both projects. No `pyproject.toml`; dependencies are installed ad hoc.
- Long jobs detach: `setsid nohup <cmd> < /dev/null > runs/<id>/stdout.log 2>&1 &`
- No experiment tracker (no wandb, no tensorboard). `runs/` is the tracker.

## Workflow

1. **Understand** — read the code and the paper before proposing anything. Delegate wide reading to a subagent; keep the main context clean.
2. **Plan** — state the hypothesis and its falsification criterion before writing code.
3. **Run** — register the run (git SHA, argv, env, GPU, exit code) before reading any result from it.
4. **Verify** — re-derive the headline number from raw output. Ask what else could explain it: test artifact, leakage, multiple comparisons, a dead dependency.
5. **Record** — update `reports/` and `reports/claims.yaml`. A finding that lives only in the conversation is lost at the next compaction.

Prefer the retraction to the defense. Two claims have already been struck here after audit; that is the process working, not failing.

## Research policy

Use tools over recall. Search arXiv and Semantic Scholar before asserting related work; read repositories with the GitHub tools rather than guessing their contents.

## References

- Project specifics: `tabagent/CLAUDE.md`, `laion_data/CLAUDE.md`
- Conventions by file type: `.claude/rules/`
- Permissions, protected paths, environment: `.claude/settings.json`
