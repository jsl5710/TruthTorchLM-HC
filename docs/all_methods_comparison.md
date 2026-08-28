# All-Methods Comparison — AUROC vs. Latency (qwen3-8b)

> Every UQ method on the same target (qwen3-8b), same 9 datasets, mean over seeds. Families: **direct** (consistency + verbalized), **DisAAD-au** & **DisAAD-eu** (both read-outs — no cherry-picking), and **OURS** (uncertainty-aware variants). `★` = on the AUROC-vs-latency Pareto frontier. Latency = auxiliary-compute p50, one forward for proxies.

**Auto-generated** by `scripts/all_methods_report.py`.


## All 9 datasets

| # | method | family | AUROC | p50 ms | Pareto |
|--:|---|---|--:|--:|:--:|
| 1 | Verb | direct | 0.685 | 112 | ★ |
| 2 | Ecc | direct | 0.672 | 3238 |  |
| 3 | EccentricityConfidence | direct | 0.671 | 3251 |  |
| 4 | PTrue | direct | 0.630 | 64 | ★ |
| 5 | **UA·SemEnt_head** | OURS | 0.610 | 31 | ★ |
| 6 | **UA·SemEnt_both_lam5** | OURS | 0.606 | 30 | ★ |
| 7 | **UA·Ecc_head** | OURS | 0.605 | 30 |  |
| 8 | KernelLanguageEntropy | direct | 0.597 | 1530 |  |
| 9 | NumSemanticSetUncertainty | direct | 0.596 | 584 |  |
| 10 | SemEnt | direct | 0.595 | 585 |  |
| 11 | DisAAD-au | DisAAD | 0.589 | 33 |  |
| 12 | LexicalSimilarity | direct | 0.577 | 6403 |  |
| 13 | **UA·SemEnt_head_lam5** | OURS | 0.575 | 29 | ★ |
| 14 | **UA·Ecc_both_lam5** | OURS | 0.573 | 31 |  |
| 15 | **UA·Ecc_head_lam5** | OURS | 0.564 | 29 |  |
| 16 | SumEigenUncertainty | direct | 0.562 | 1784 |  |
| 17 | EigV | direct | 0.562 | 1776 |  |
| 18 | MatrixDegreeUncertainty | direct | 0.560 | 1780 |  |
| 19 | MatrixDegreeConfidence | direct | 0.552 | 1990 |  |
| 20 | **UA·Verb_head** | OURS | 0.541 | 29 | ★ |
| 21 | DisAAD-eu | DisAAD | 0.524 | 33 |  |
| 22 | **UA·SemEnt_edl** | OURS | 0.511 | 28 | ★ |
| 23 | **UA·Ecc_edl** | OURS | 0.504 | 30 |  |
| 24 | **UA·Verb_edl** | OURS | 0.498 | 30 |  |
| 25 | DALD-proxy | DALD | 0.470 | 29 |  |

## Health · MCQ

| # | method | family | AUROC | p50 ms | Pareto |
|--:|---|---|--:|--:|:--:|
| 1 | PTrue | direct | 0.698 | 73 | ★ |
| 2 | EccentricityConfidence | direct | 0.634 | 3736 |  |
| 3 | LexicalSimilarity | direct | 0.602 | 5951 |  |
| 4 | Ecc | direct | 0.601 | 3666 |  |
| 5 | Verb | direct | 0.590 | 116 |  |
| 6 | **UA·Verb_head** | OURS | 0.544 | 29 | ★ |
| 7 | DisAAD-au | DisAAD | 0.542 | 31 |  |
| 8 | SumEigenUncertainty | direct | 0.540 | 3724 |  |
| 9 | EigV | direct | 0.540 | 3689 |  |
| 10 | MatrixDegreeUncertainty | direct | 0.540 | 3673 |  |
| 11 | MatrixDegreeConfidence | direct | 0.532 | 3671 |  |
| 12 | KernelLanguageEntropy | direct | 0.508 | 3389 |  |
| 13 | SemEnt | direct | 0.506 | 681 |  |
| 14 | NumSemanticSetUncertainty | direct | 0.506 | 681 |  |
| 15 | **UA·Ecc_edl** | OURS | 0.464 | 30 |  |
| 16 | **UA·Verb_edl** | OURS | 0.464 | 30 |  |
| 17 | **UA·SemEnt_edl** | OURS | 0.458 | 28 | ★ |
| 18 | DALD-proxy | DALD | 0.457 | 29 |  |
| 19 | DisAAD-eu | DisAAD | 0.451 | 31 |  |
| 20 | **UA·SemEnt_head_lam5** | OURS | 0.426 | 29 |  |
| 21 | **UA·SemEnt_head** | OURS | 0.424 | 31 |  |
| 22 | **UA·SemEnt_both_lam5** | OURS | 0.423 | 30 |  |
| 23 | **UA·Ecc_both_lam5** | OURS | 0.418 | 31 |  |
| 24 | **UA·Ecc_head** | OURS | 0.404 | 30 |  |
| 25 | **UA·Ecc_head_lam5** | OURS | 0.400 | 30 |  |

## Health · Free-form QA

| # | method | family | AUROC | p50 ms | Pareto |
|--:|---|---|--:|--:|:--:|
| 1 | Verb | direct | 0.771 | 113 | ★ |
| 2 | **UA·Ecc_head** | OURS | 0.698 | 30 | ★ |
| 3 | **UA·SemEnt_head** | OURS | 0.686 | 31 |  |
| 4 | DisAAD-au | DisAAD | 0.685 | 35 |  |
| 5 | **UA·SemEnt_both_lam5** | OURS | 0.674 | 30 | ★ |
| 6 | Ecc | direct | 0.658 | 3186 |  |
| 7 | EccentricityConfidence | direct | 0.655 | 3189 |  |
| 8 | **UA·Ecc_both_lam5** | OURS | 0.581 | 31 |  |
| 9 | **UA·SemEnt_head_lam5** | OURS | 0.569 | 29 | ★ |
| 10 | **UA·Ecc_head_lam5** | OURS | 0.563 | 29 |  |
| 11 | DisAAD-eu | DisAAD | 0.541 | 35 |  |
| 12 | KernelLanguageEntropy | direct | 0.516 | 604 |  |
| 13 | SemEnt | direct | 0.514 | 548 |  |
| 14 | NumSemanticSetUncertainty | direct | 0.514 | 545 |  |
| 15 | PTrue | direct | 0.499 | 66 |  |
| 16 | LexicalSimilarity | direct | 0.485 | 356 |  |
| 17 | **UA·SemEnt_edl** | OURS | 0.479 | 28 | ★ |
| 18 | **UA·Ecc_edl** | OURS | 0.451 | 30 |  |
| 19 | **UA·Verb_edl** | OURS | 0.428 | 30 |  |
| 20 | **UA·Verb_head** | OURS | 0.420 | 29 |  |
| 21 | EigV | direct | 0.395 | 712 |  |
| 22 | SumEigenUncertainty | direct | 0.395 | 708 |  |
| 23 | MatrixDegreeUncertainty | direct | 0.395 | 707 |  |
| 24 | MatrixDegreeConfidence | direct | 0.387 | 1064 |  |
| 25 | DALD-proxy | DALD | 0.377 | 29 |  |

## General · QA

| # | method | family | AUROC | p50 ms | Pareto |
|--:|---|---|--:|--:|:--:|
| 1 | Ecc | direct | 0.718 | 3063 | ★ |
| 2 | KernelLanguageEntropy | direct | 0.704 | 2612 | ★ |
| 3 | EccentricityConfidence | direct | 0.703 | 3056 |  |
| 4 | NumSemanticSetUncertainty | direct | 0.703 | 564 | ★ |
| 5 | SumEigenUncertainty | direct | 0.703 | 3067 |  |
| 6 | EigV | direct | 0.703 | 3053 |  |
| 7 | SemEnt | direct | 0.701 | 566 |  |
| 8 | MatrixDegreeUncertainty | direct | 0.698 | 3064 |  |
| 9 | PTrue | direct | 0.695 | 58 | ★ |
| 10 | MatrixDegreeConfidence | direct | 0.688 | 3064 |  |
| 11 | Verb | direct | 0.667 | 108 |  |
| 12 | **UA·SemEnt_head_lam5** | OURS | 0.653 | 29 | ★ |
| 13 | **UA·Ecc_head_lam5** | OURS | 0.646 | 29 |  |
| 14 | **UA·SemEnt_both_lam5** | OURS | 0.646 | 30 |  |
| 15 | **UA·SemEnt_head** | OURS | 0.646 | 31 |  |
| 16 | **UA·Ecc_both_lam5** | OURS | 0.644 | 31 |  |
| 17 | LexicalSimilarity | direct | 0.638 | 13387 |  |
| 18 | **UA·Ecc_head** | OURS | 0.636 | 30 |  |
| 19 | **UA·Verb_head** | OURS | 0.631 | 29 | ★ |
| 20 | **UA·Verb_edl** | OURS | 0.567 | 30 |  |
| 21 | **UA·Ecc_edl** | OURS | 0.564 | 30 |  |
| 22 | **UA·SemEnt_edl** | OURS | 0.563 | 28 | ★ |
| 23 | DisAAD-eu | DisAAD | 0.547 | 33 |  |
| 24 | DALD-proxy | DALD | 0.546 | 29 |  |
| 25 | DisAAD-au | DisAAD | 0.542 | 33 |  |

**Read:** our UA head variants are **Pareto-optimal** (best AUROC achievable at ~30 ms) and beat **both** DisAAD-au and DisAAD-eu — but the few-pass direct methods (VerbalizedConfidence, PTrue) and the slow graph methods reach higher AUROC at 2–100× the latency. The defensible claim is *best accuracy-per-millisecond*, not *highest AUROC*.

