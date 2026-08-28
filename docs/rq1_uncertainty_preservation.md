# RQ1 — Proxy Uncertainty Preservation

> **RQ1.** Does the distilled proxy preserve the uncertainty behavior of the target black-box LLM under both in-distribution (ID) and out-of-distribution (OOD) conditions?

**Auto-generated** by `scripts/rq1_report.py`.

## Setup

- **Teacher** = `qwen3-8b` (the distillation target); **proxy** = the distilled `qwen3-0.6b` (merged). The **same four logit-based estimators** are computed on **both** models' logits over each cached (prompt + response): **EDL-AU, EDL-EU, MSP, Entropy** (one forward pass each, mean over the response span, higher = more uncertain).
- **Split:** ID = **TriviaQA** (the distillation domain, `tqa`); OOD = **everything else** (NaturalQA/PopQA/TruthfulQA + all health). If distillation transferred *uncertainty* (not just outputs), teacher↔proxy agreement should hold ID and — the hypothesis under test — **degrade OOD**.
- Pooled over 3 seeds; 4032 teacher/proxy response pairs total (447 ID, 3585 OOD).


## Agreement — does the proxy's uncertainty track the teacher's?

| Estimator | Split | Spearman ρ | Pearson r | Kendall τ |
|---|---|--:|--:|--:|
| EDL-AU | ID (TriviaQA) | 0.603 | 0.615 | 0.427 |
| EDL-AU | OOD (rest) | 0.679 | 0.671 | 0.487 |
| EDL-EU | ID (TriviaQA) | 0.440 | 0.426 | 0.308 |
| EDL-EU | OOD (rest) | 0.288 | 0.347 | 0.194 |
| MSP | ID (TriviaQA) | 0.549 | 0.576 | 0.386 |
| MSP | OOD (rest) | 0.720 | 0.713 | 0.532 |
| Entropy | ID (TriviaQA) | 0.577 | 0.579 | 0.401 |
| Entropy | OOD (rest) | 0.684 | 0.641 | 0.500 |

## Discrimination & calibration — does the proxy degrade vs the teacher?

| Estimator | Split | Teacher AUROC | Proxy AUROC | Δ(proxy−teacher) | Teacher ECE | Proxy ECE |
|---|---|--:|--:|--:|--:|--:|
| EDL-AU | ID (TriviaQA) | 0.527 | 0.507 | -0.020 | 0.050 | 0.016 |
| EDL-AU | OOD (rest) | 0.510 | 0.556 | +0.046 | 0.005 | 0.008 |
| EDL-EU | ID (TriviaQA) | 0.587 | 0.582 | -0.005 | 0.007 | 0.022 |
| EDL-EU | OOD (rest) | 0.633 | 0.540 | -0.092 | 0.006 | 0.013 |
| MSP | ID (TriviaQA) | 0.605 | 0.556 | -0.049 | 0.050 | 0.024 |
| MSP | OOD (rest) | 0.642 | 0.609 | -0.034 | 0.010 | 0.011 |
| Entropy | ID (TriviaQA) | 0.627 | 0.581 | -0.046 | 0.034 | 0.029 |
| Entropy | OOD (rest) | 0.655 | 0.620 | -0.036 | 0.015 | 0.008 |

## OOD detail — teacher↔proxy agreement by dataset (EDL-AU)

| Dataset | Split | n | Spearman ρ (AU) | Proxy AUROC (AU) | Teacher AUROC (AU) |
|---|---|--:|--:|--:|--:|
| trivia_qa | ID | 447 | 0.603 | 0.507 | 0.527 |
| bioasq | OOD | 450 | 0.183 | 0.899 | 0.772 |
| kqa | OOD | 435 | 0.528 | 0.614 | 0.538 |
| medlfqa | OOD | 450 | 0.377 | 0.538 | 0.477 |
| medqa | OOD | 450 | 0.550 | 0.528 | 0.592 |
| mmlu_med | OOD | 450 | 0.604 | 0.555 | 0.599 |
| natural_qa | OOD | 450 | 0.626 | 0.586 | 0.467 |
| pop_qa | OOD | 450 | 0.542 | 0.657 | 0.704 |
| truthful_qa | OOD | 450 | 0.695 | 0.423 | 0.549 |

## Finding — is uncertainty preserved?

- **Mean teacher↔proxy Spearman agreement:** ID = **0.543**, OOD = **0.593** (Δ = -0.050). Agreement is **similar ID and OOD** — no evidence of an OOD preservation collapse on this axis.

- **Read the Δ(proxy−teacher) AUROC column**: a large negative Δ that worsens OOD is the 'silent proxy degradation' RQ1 predicts — the proxy's uncertainty discriminates errors well ID but loses the teacher's signal OOD.

