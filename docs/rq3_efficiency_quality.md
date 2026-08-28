# RQ3 — Efficiency–Quality Trade-off

> **RQ3.** Do the efficiency gains of the proxy-based (hybrid) uncertainty-estimation framework justify any trade-off in uncertainty-estimation quality compared with direct black-box uncertainty-estimation methods?

**Auto-generated** by `scripts/rq3_report.py`. Organised **Domain → Task type → Dataset**.

## Setup — a common footing

- **Target model:** `qwen3-8b` — the *only* fair target, because the DisAAD proxy is distilled from it (per-target method). Every method scores the **same** cached generations.
- **Method types:** *proxy (hybrid)* = DisAAD (evidential **AU**/**EU** on the distilled 0.6B proxy — one forward pass); *direct* = consistency family (multi-sample) + verbalized (PTrue, VerbalizedConfidence).
- **Taxonomy:** Health {MCQ · Free-form QA} · General/Open {QA}. **Dialogue** (tau-bench) is a task type in the benchmark but **out of RQ3 scope** — its trajectories are precomputed gpt-4o/sonnet, not qwen3-8b, so the per-target proxy was never distilled/scored there.
- **Metrics:** *quality* = AUROC, AUPR, ECE (calibration); *task performance* = generator accuracy (correct-rate) per dataset; *efficiency* = auxiliary-compute latency p50/p95 + the compute-cost column (true per-query cost). Each method at its **best-AUROC N**.
- **Calibration** (ECE): cross-fitted isotonic normalizer, identical for both arms. Status: **12 cells loaded**.


## Health — MCQ


### Aggregate (mean over this group's datasets × seeds)

| Method | Type | N | AUROC | AUPR | ECE | Brier | p50 ms | p95 ms | Deployment cost |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| **DisAAD-au** | **proxy** | 1 | 0.542 | 0.709 | 0.059 | 0.231 | 31 | — | 1 proxy fwd (0.6B) + offline distill (one-time ~12h) |
| **DisAAD-eu** | **proxy** | 1 | 0.451 | 0.662 | 0.065 | 0.234 | 31 | — | 1 proxy fwd (0.6B) + offline distill (one-time ~12h) |
| PTrue | direct | 5 | 0.698 | 0.838 | — | — | 73 | 103 | ~3-5 target calls |
| EccentricityConfidence | direct | 10 | 0.634 | 0.804 | 0.053 | 0.221 | 3736 | 3912 | N target gens + O(N^2) NLI graph |
| LexicalSimilarity | direct | 10 | 0.602 | 0.843 | 0.050 | 0.227 | 5951 | 11332 | N target gens + O(N^2) ROUGE |
| EccentricityUncertainty | direct | 10 | 0.601 | 0.745 | 0.051 | 0.224 | 3666 | 3902 | N target gens + O(N^2) NLI graph |
| VerbalizedConfidence | direct | 5 | 0.590 | 0.813 | — | — | 116 | 122 | 1 extra target call |
| EigV | direct | 10 | 0.540 | 0.696 | 0.054 | 0.228 | 3689 | 3923 | N target gens + O(N^2) NLI graph |
| SumEigenUncertainty | direct | 10 | 0.540 | 0.696 | 0.054 | 0.228 | 3724 | 3892 | N target gens + O(N^2) NLI graph |
| MatrixDegreeUncertainty | direct | 10 | 0.540 | 0.696 | 0.054 | 0.228 | 3673 | 3900 | N target gens + O(N^2) NLI graph |
| MatrixDegreeConfidence | direct | 10 | 0.532 | 0.693 | 0.053 | 0.227 | 3671 | 3900 | N target gens + O(N^2) NLI graph |
| KernelLanguageEntropy | direct | 10 | 0.508 | 0.847 | 0.000 | 0.220 | 3389 | 3615 | N target gens + O(N^2) NLI kernel |
| DiscreteSemanticEntropy | direct | 10 | 0.506 | 0.846 | 0.000 | 0.220 | 681 | 729 | N target gens + 1 NLI cluster |
| NumSemanticSetUncertainty | direct | 10 | 0.506 | 0.846 | 0.000 | 0.220 | 681 | 728 | N target gens + 1 NLI cluster |

### Per-dataset detail


**MedQA** · task accuracy 67%

| Method | Type | AUROC | AUPR | ECE | p50 ms |
|---|---|--:|--:|--:|--:|
| **DisAAD-au** | proxy | 0.529 | 0.720 | 0.059 | 31 |
| **DisAAD-eu** | proxy | 0.496 | 0.664 | 0.065 | 31 |
| PTrue | direct | 0.701 | 0.813 | — | 88 |
| EccentricityConfidence | direct | 0.565 | 0.750 | 0.053 | 4233 |
| LexicalSimilarity | direct | 0.564 | 0.824 | 0.050 | 7371 |
| EigV | direct | 0.563 | 0.680 | 0.054 | 4121 |
| MatrixDegreeUncertainty | direct | 0.563 | 0.680 | 0.054 | 4126 |
| SumEigenUncertainty | direct | 0.563 | 0.680 | 0.054 | 4241 |
| EccentricityUncertainty | direct | 0.561 | 0.724 | 0.051 | 4122 |
| MatrixDegreeConfidence | direct | 0.559 | 0.678 | 0.053 | 4122 |
| VerbalizedConfidence | direct | 0.510 | 0.758 | — | 119 |
| DiscreteSemanticEntropy | direct | 0.500 | 0.837 | 0.000 | 804 |
| NumSemanticSetUncertainty | direct | 0.500 | 0.837 | 0.000 | 805 |
| KernelLanguageEntropy | direct | 0.500 | 0.837 | 0.000 | 4026 |

**MMLU-Med** · task accuracy 71%

| Method | Type | AUROC | AUPR | ECE | p50 ms |
|---|---|--:|--:|--:|--:|
| **DisAAD-au** | proxy | 0.554 | 0.697 | — | 30 |
| **DisAAD-eu** | proxy | 0.405 | 0.660 | — | 30 |
| EccentricityConfidence | direct | 0.703 | 0.857 | — | 3240 |
| PTrue | direct | 0.695 | 0.863 | — | 59 |
| VerbalizedConfidence | direct | 0.671 | 0.867 | — | 113 |
| EccentricityUncertainty | direct | 0.642 | 0.767 | — | 3210 |
| LexicalSimilarity | direct | 0.639 | 0.863 | — | 4531 |
| EigV | direct | 0.517 | 0.712 | — | 3257 |
| SumEigenUncertainty | direct | 0.517 | 0.712 | — | 3208 |
| MatrixDegreeUncertainty | direct | 0.516 | 0.712 | — | 3220 |
| KernelLanguageEntropy | direct | 0.515 | 0.857 | — | 2752 |
| DiscreteSemanticEntropy | direct | 0.511 | 0.856 | — | 559 |
| NumSemanticSetUncertainty | direct | 0.511 | 0.856 | — | 557 |
| MatrixDegreeConfidence | direct | 0.504 | 0.707 | — | 3220 |

## Health — Free-form QA


### Aggregate (mean over this group's datasets × seeds)

| Method | Type | N | AUROC | AUPR | ECE | Brier | p50 ms | p95 ms | Deployment cost |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| **DisAAD-au** | **proxy** | 1 | 0.685 | 0.916 | 0.044 | 0.098 | 35 | — | 1 proxy fwd (0.6B) + offline distill (one-time ~12h) |
| **DisAAD-eu** | **proxy** | 1 | 0.541 | 0.885 | 0.018 | 0.096 | 35 | — | 1 proxy fwd (0.6B) + offline distill (one-time ~12h) |
| VerbalizedConfidence | direct | 5 | 0.771 | 0.944 | — | — | 113 | 114 | 1 extra target call |
| EccentricityUncertainty | direct | 10 | 0.658 | 0.923 | 0.032 | 0.097 | 3186 | 3210 | N target gens + O(N^2) NLI graph |
| EccentricityConfidence | direct | 10 | 0.655 | 0.925 | 0.033 | 0.098 | 3189 | 3208 | N target gens + O(N^2) NLI graph |
| KernelLanguageEntropy | direct | 5 | 0.516 | 0.922 | 0.033 | 0.096 | 604 | 609 | N target gens + O(N^2) NLI kernel |
| DiscreteSemanticEntropy | direct | 10 | 0.514 | 0.942 | 0.015 | 0.098 | 548 | 554 | N target gens + 1 NLI cluster |
| NumSemanticSetUncertainty | direct | 10 | 0.514 | 0.942 | 0.013 | 0.097 | 545 | 552 | N target gens + 1 NLI cluster |
| PTrue | direct | 5 | 0.499 | 0.938 | — | — | 66 | 79 | ~3-5 target calls |
| LexicalSimilarity | direct | 5 | 0.485 | 0.917 | 0.028 | 0.097 | 356 | 1743 | N target gens + O(N^2) ROUGE |
| EigV | direct | 5 | 0.395 | 0.906 | 0.038 | 0.100 | 712 | 719 | N target gens + O(N^2) NLI graph |
| SumEigenUncertainty | direct | 5 | 0.395 | 0.906 | 0.038 | 0.100 | 708 | 714 | N target gens + O(N^2) NLI graph |
| MatrixDegreeUncertainty | direct | 5 | 0.395 | 0.906 | 0.038 | 0.100 | 707 | 713 | N target gens + O(N^2) NLI graph |
| MatrixDegreeConfidence | direct | 5 | 0.387 | 0.904 | 0.031 | 0.100 | 1064 | 1073 | N target gens + O(N^2) NLI graph |

### Per-dataset detail


**K-QA** · task accuracy 77%

| Method | Type | AUROC | AUPR | ECE | p50 ms |
|---|---|--:|--:|--:|--:|
| **DisAAD-au** | proxy | 0.611 | 0.852 | 0.092 | 37 |
| **DisAAD-eu** | proxy | 0.510 | 0.765 | 0.040 | 37 |
| KernelLanguageEntropy | direct | 0.631 | 0.871 | 0.078 | 605 |
| LexicalSimilarity | direct | 0.626 | 0.860 | 0.057 | 485 |
| VerbalizedConfidence | direct | 0.616 | 0.880 | — | 113 |
| EccentricityConfidence | direct | 0.598 | 0.857 | 0.070 | 3183 |
| EigV | direct | 0.587 | 0.842 | 0.073 | 703 |
| SumEigenUncertainty | direct | 0.587 | 0.842 | 0.073 | 704 |
| MatrixDegreeUncertainty | direct | 0.586 | 0.842 | 0.073 | 708 |
| MatrixDegreeConfidence | direct | 0.585 | 0.841 | 0.062 | 1063 |
| EccentricityUncertainty | direct | 0.583 | 0.851 | 0.056 | 3185 |
| PTrue | direct | 0.543 | 0.882 | — | 74 |
| DiscreteSemanticEntropy | direct | 0.518 | 0.884 | 0.022 | 540 |
| NumSemanticSetUncertainty | direct | 0.518 | 0.884 | 0.022 | 543 |

**MedLFQA** · task accuracy 89%

| Method | Type | AUROC | AUPR | ECE | p50 ms |
|---|---|--:|--:|--:|--:|
| **DisAAD-au** | proxy | 0.537 | 0.898 | 0.041 | 35 |
| **DisAAD-eu** | proxy | 0.490 | 0.893 | 0.014 | 35 |
| VerbalizedConfidence | direct | 0.705 | 0.954 | — | 112 |
| EccentricityConfidence | direct | 0.590 | 0.919 | 0.030 | 3186 |
| EccentricityUncertainty | direct | 0.574 | 0.918 | 0.041 | 3178 |
| DiscreteSemanticEntropy | direct | 0.526 | 0.944 | 0.023 | 552 |
| EigV | direct | 0.526 | 0.895 | 0.042 | 716 |
| SumEigenUncertainty | direct | 0.526 | 0.895 | 0.042 | 710 |
| NumSemanticSetUncertainty | direct | 0.526 | 0.944 | 0.017 | 545 |
| MatrixDegreeUncertainty | direct | 0.526 | 0.895 | 0.041 | 703 |
| MatrixDegreeConfidence | direct | 0.519 | 0.890 | 0.032 | 1064 |
| LexicalSimilarity | direct | 0.484 | 0.898 | 0.026 | 291 |
| PTrue | direct | 0.473 | 0.936 | — | 65 |
| KernelLanguageEntropy | direct | 0.448 | 0.898 | 0.022 | 603 |

**BioASQ** · task accuracy 99%

| Method | Type | AUROC | AUPR | ECE | p50 ms |
|---|---|--:|--:|--:|--:|
| **DisAAD-au** | proxy | 0.906 | 0.999 | 0.000 | 33 |
| **DisAAD-eu** | proxy | 0.624 | 0.997 | 0.000 | 33 |
| VerbalizedConfidence | direct | 0.990 | 1.000 | — | 113 |
| EccentricityUncertainty | direct | 0.817 | 0.999 | 0.000 | 3195 |
| EccentricityConfidence | direct | 0.776 | 0.998 | 0.000 | 3197 |
| DiscreteSemanticEntropy | direct | 0.498 | 0.997 | 0.000 | 551 |
| NumSemanticSetUncertainty | direct | 0.498 | 0.997 | 0.000 | 547 |
| PTrue | direct | 0.481 | 0.996 | — | 60 |
| KernelLanguageEntropy | direct | 0.470 | 0.996 | 0.000 | 605 |
| LexicalSimilarity | direct | 0.346 | 0.994 | 0.000 | 291 |
| EigV | direct | 0.072 | 0.981 | 0.000 | 715 |
| MatrixDegreeUncertainty | direct | 0.072 | 0.981 | 0.000 | 710 |
| SumEigenUncertainty | direct | 0.072 | 0.981 | 0.000 | 709 |
| MatrixDegreeConfidence | direct | 0.056 | 0.980 | 0.000 | 1065 |

## General / Open-domain — QA


### Aggregate (mean over this group's datasets × seeds)

| Method | Type | N | AUROC | AUPR | ECE | Brier | p50 ms | p95 ms | Deployment cost |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| **DisAAD-eu** | **proxy** | 1 | 0.547 | 0.494 | — | — | 33 | — | 1 proxy fwd (0.6B) + offline distill (one-time ~12h) |
| **DisAAD-au** | **proxy** | 1 | 0.542 | 0.536 | — | — | 33 | — | 1 proxy fwd (0.6B) + offline distill (one-time ~12h) |
| EccentricityUncertainty | direct | 10 | 0.718 | 0.659 | — | — | 3063 | 3100 | N target gens + O(N^2) NLI graph |
| KernelLanguageEntropy | direct | 10 | 0.704 | 0.708 | — | — | 2612 | 2634 | N target gens + O(N^2) NLI kernel |
| EccentricityConfidence | direct | 10 | 0.703 | 0.666 | — | — | 3056 | 3088 | N target gens + O(N^2) NLI graph |
| NumSemanticSetUncertainty | direct | 10 | 0.703 | 0.758 | — | — | 564 | 1448 | N target gens + 1 NLI cluster |
| EigV | direct | 10 | 0.703 | 0.613 | — | — | 3053 | 3089 | N target gens + O(N^2) NLI graph |
| SumEigenUncertainty | direct | 10 | 0.703 | 0.613 | — | — | 3067 | 3111 | N target gens + O(N^2) NLI graph |
| DiscreteSemanticEntropy | direct | 10 | 0.701 | 0.758 | — | — | 566 | 1452 | N target gens + 1 NLI cluster |
| MatrixDegreeUncertainty | direct | 10 | 0.698 | 0.610 | — | — | 3064 | 3095 | N target gens + O(N^2) NLI graph |
| PTrue | direct | 5 | 0.695 | 0.745 | — | — | 58 | 73 | ~3-5 target calls |
| MatrixDegreeConfidence | direct | 10 | 0.688 | 0.604 | — | — | 3064 | 3098 | N target gens + O(N^2) NLI graph |
| VerbalizedConfidence | direct | 5 | 0.667 | 0.712 | — | — | 108 | 110 | 1 extra target call |
| LexicalSimilarity | direct | 10 | 0.638 | 0.652 | — | — | 13387 | 19264 | N target gens + O(N^2) ROUGE |

### Per-dataset detail


**TriviaQA** · task accuracy 44%

| Method | Type | AUROC | AUPR | ECE | p50 ms |
|---|---|--:|--:|--:|--:|
| **DisAAD-eu** | proxy | 0.582 | 0.472 | — | 33 |
| **DisAAD-au** | proxy | 0.507 | 0.470 | — | 33 |
| EccentricityUncertainty | direct | 0.756 | 0.687 | — | 3069 |
| EccentricityConfidence | direct | 0.750 | 0.722 | — | 3064 |
| KernelLanguageEntropy | direct | 0.743 | 0.771 | — | 2606 |
| LexicalSimilarity | direct | 0.741 | 0.755 | — | 24929 |
| EigV | direct | 0.720 | 0.598 | — | 3057 |
| SumEigenUncertainty | direct | 0.720 | 0.598 | — | 3078 |
| PTrue | direct | 0.719 | 0.748 | — | 58 |
| VerbalizedConfidence | direct | 0.717 | 0.748 | — | 109 |
| NumSemanticSetUncertainty | direct | 0.715 | 0.754 | — | 528 |
| MatrixDegreeUncertainty | direct | 0.714 | 0.595 | — | 3061 |
| DiscreteSemanticEntropy | direct | 0.712 | 0.752 | — | 525 |
| MatrixDegreeConfidence | direct | 0.708 | 0.588 | — | 3062 |

**NaturalQA** · task accuracy 51%

| Method | Type | AUROC | AUPR | ECE | p50 ms |
|---|---|--:|--:|--:|--:|
| **DisAAD-au** | proxy | 0.585 | 0.623 | — | 34 |
| **DisAAD-eu** | proxy | 0.510 | 0.490 | — | 34 |
| EccentricityUncertainty | direct | 0.744 | 0.719 | — | 3061 |
| EigV | direct | 0.728 | 0.655 | — | 3045 |
| SumEigenUncertainty | direct | 0.728 | 0.655 | — | 3037 |
| EccentricityConfidence | direct | 0.727 | 0.720 | — | 3050 |
| NumSemanticSetUncertainty | direct | 0.724 | 0.778 | — | 524 |
| MatrixDegreeUncertainty | direct | 0.723 | 0.653 | — | 3040 |
| DiscreteSemanticEntropy | direct | 0.717 | 0.774 | — | 532 |
| KernelLanguageEntropy | direct | 0.717 | 0.746 | — | 2596 |
| MatrixDegreeConfidence | direct | 0.706 | 0.635 | — | 3064 |
| PTrue | direct | 0.705 | 0.778 | — | 59 |
| VerbalizedConfidence | direct | 0.671 | 0.752 | — | 108 |
| LexicalSimilarity | direct | 0.622 | 0.693 | — | 21472 |

**PopQA** · task accuracy 38%

| Method | Type | AUROC | AUPR | ECE | p50 ms |
|---|---|--:|--:|--:|--:|
| **DisAAD-au** | proxy | 0.654 | 0.563 | — | 32 |
| **DisAAD-eu** | proxy | 0.554 | 0.465 | — | 32 |
| EigV | direct | 0.797 | 0.658 | — | 3029 |
| SumEigenUncertainty | direct | 0.797 | 0.658 | — | 3073 |
| MatrixDegreeUncertainty | direct | 0.788 | 0.651 | — | 3077 |
| MatrixDegreeConfidence | direct | 0.784 | 0.650 | — | 3059 |
| DiscreteSemanticEntropy | direct | 0.778 | 0.743 | — | 680 |
| NumSemanticSetUncertainty | direct | 0.777 | 0.743 | — | 681 |
| EccentricityUncertainty | direct | 0.772 | 0.640 | — | 3039 |
| KernelLanguageEntropy | direct | 0.757 | 0.653 | — | 2628 |
| EccentricityConfidence | direct | 0.746 | 0.628 | — | 3032 |
| LexicalSimilarity | direct | 0.716 | 0.591 | — | 1235 |
| PTrue | direct | 0.711 | 0.702 | — | 54 |
| VerbalizedConfidence | direct | 0.623 | 0.623 | — | 108 |

**TruthfulQA** · task accuracy 53%

| Method | Type | AUROC | AUPR | ECE | p50 ms |
|---|---|--:|--:|--:|--:|
| **DisAAD-eu** | proxy | 0.542 | 0.548 | — | 34 |
| **DisAAD-au** | proxy | 0.422 | 0.489 | — | 34 |
| VerbalizedConfidence | direct | 0.659 | 0.727 | — | 109 |
| PTrue | direct | 0.644 | 0.753 | — | 61 |
| EccentricityUncertainty | direct | 0.600 | 0.592 | — | 3082 |
| DiscreteSemanticEntropy | direct | 0.599 | 0.761 | — | 526 |
| KernelLanguageEntropy | direct | 0.598 | 0.661 | — | 2619 |
| NumSemanticSetUncertainty | direct | 0.596 | 0.760 | — | 525 |
| EccentricityConfidence | direct | 0.589 | 0.594 | — | 3078 |
| MatrixDegreeUncertainty | direct | 0.565 | 0.539 | — | 3076 |
| EigV | direct | 0.565 | 0.539 | — | 3079 |
| SumEigenUncertainty | direct | 0.565 | 0.539 | — | 3079 |
| MatrixDegreeConfidence | direct | 0.552 | 0.541 | — | 3069 |
| LexicalSimilarity | direct | 0.471 | 0.567 | — | 5912 |

## Dialogue / agentic (tau-bench) — out of RQ3 scope

tau-bench is the benchmark's dialogue/agentic regime, but it is **excluded from this RQ3 comparison**: its runs are precomputed gpt-4o and claude-3.5-sonnet trajectories, so there is no qwen3-8b proxy to compare (DisAAD is per-target). It is reported separately in the main frontier. Extending RQ3 to dialogue needs a proxy distilled from a dialogue target.


## Efficiency–quality summary — is the trade-off justified?

- **Health · MCQ:** best direct = **PTrue** (0.698 AUROC @ 73 ms); proxy **DisAAD-au = 0.542 @ 31 ms** (2× faster). → a **real trade-off** — proxy gives up 0.157 AUROC for the speed.

- **Health · Free-form QA:** best direct = **VerbalizedConfidence** (0.771 AUROC @ 113 ms); proxy **DisAAD-au = 0.685 @ 35 ms** (3× faster). → a **real trade-off** — proxy gives up 0.086 AUROC for the speed.

- **General / Open-domain · QA:** best direct = **EccentricityUncertainty** (0.718 AUROC @ 3063 ms); proxy **DisAAD-au = 0.542 @ 33 ms** (92× faster). → a **real trade-off** — proxy gives up 0.176 AUROC for the speed.

- **Compute cost:** the proxy pays a **one-time** offline distillation (~12 h) then **one 0.6 B forward pass** per query — target-decoupled, so its marginal cost is ~constant regardless of how slow the target is. Direct multi-sample methods pay **N target generations + O(N²) clustering** on *every* query; verbalized pays 1–5 extra target calls.


*Latency is auxiliary-compute only; the compute-cost column gives true deployment cost. DisAAD is qwen3-8b-only by construction (per-target proxy). Each method shown at its best-AUROC N.*

