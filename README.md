# ralphton

Two independent research projects, one repo. They share nothing but a Python environment and a
working style: every claim is measured, every discarded run is recorded, and nothing is written
down that the experiments have not earned.

| project | question | state |
|---|---|---|
| [`tabagent/`](tabagent/) | Can an LLM agent repair the frozen synthetic prior of a tabular foundation model (TabICL), after the checkpoint has shipped? | **Negative result, measured.** Every arm loses to leaving the checkpoint alone. The live follow-up is repairing the *context* instead of the prior. |
| [`laion_data/`](laion_data/) | Does Doerig et al. (*Nature Machine Intelligence* 2025), "High-level visual representations in the human brain are aligned with large language models", reproduce — and does it generalise off COCO? | **Reproduction in progress.** Figures 1 and 3 reproduce; one published claim retracted after audit; the paper's Fig. 4 activations turn out to be irreproducible from the authors' own code. |

## tabagent — repairing the prior

TabICL/TabPFN are pre-trained on a prior over random SCMs, frozen at publication. The agent is
shown only anonymised statistics of downstream tasks, revises the prior's generating config *and
its generator code*, and LoRA-adapts the released checkpoint to each candidate. Control: random
search over the same knobs, same GPU seconds.

A C2ST says the prior is perfectly distinguishable from real tables (AUC 1.000). The agent
closes 30% of that gap — and the model gets **worse**. The unrealism is domain randomisation;
narrowing it destroys the coverage the model depends on.

→ [`tabagent/RESULTS.md`](tabagent/RESULTS.md) (findings) · [`tabagent/NEXT.md`](tabagent/NEXT.md) (handoff)

## laion_data — re:vision

A reproduction of Doerig et al. 2025 on NSD, plus a generalisation test on LAION-fMRI, off the
COCO distribution the paper's captions come from.

→ [`laion_data/reports/master_report.md`](laion_data/reports/master_report.md) (what has been
verified, and what has not) · [`laion_data/RE_VISION_PLAN.md`](laion_data/RE_VISION_PLAN.md) (the plan, in Korean)

## Also here

- [`tabagent/LLM_에이전트_기반_Tabular_Prior_생성기_문헌조사.md`](tabagent/LLM_%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8_%EA%B8%B0%EB%B0%98_Tabular_Prior_%EC%83%9D%EC%84%B1%EA%B8%B0_%EB%AC%B8%ED%97%8C%EC%A1%B0%EC%82%AC.md)
  — literature review behind tabagent.

## What is not committed

Data and secrets, on purpose. NSD betas are DUA-gated and ~315 GB; COCO derivatives are
regenerable from `laion_data/src/`; model caches, logs, vendored clones and `.env` are
gitignored. Generated paper artifacts (`tabagent/paper/numbers.tex`, `main.pdf`) are gitignored
too, so a stale number can never be mistaken for a result.
