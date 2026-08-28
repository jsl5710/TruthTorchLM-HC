# Multi-Target RQ — by Domain + Latency

> Companion to `multitarget_comparison.md` (per-student). **Table (a)** is per-domain AUROC, each family shown at its **best over students/variants** (the ceiling it reaches). **Table (b)** is p50 latency per method. PTrue is excluded everywhere (grey-box: reads the target's logprobs).

**Auto-generated** by `scripts/mt_report_domains.py`.


## (a) AUROC by domain × method family


### qwen3-32b

| domain | best Direct | DALD | DisAAD | Ours | winner |
|---|---|--:|--:|--:|---|
| General | EccentricityConfidence 0.753 | 0.581 | 0.595 | **0.635** | Direct |
| **Medical** | VerbalizedConfidence 0.583 | 0.550 | 0.549 | **0.637** | Ours ✅ |
| Math | MatrixDegreeConfidence 0.708 | 0.767 | 0.761 | **0.650** | DALD |
| Adversarial | VerbalizedConfidence 0.686 | 0.680 | 0.677 | **0.669** | Direct |
| Factual | MatrixDegreeConfidence 0.594 | 0.631 | 0.631 | **0.521** | DisAAD |

### llama3.3-70b

| domain | best Direct | DALD | DisAAD | Ours | winner |
|---|---|--:|--:|--:|---|
| General | EccentricityConfidence 0.720 | 0.751 | 0.750 | **0.676** | DALD |
| **Medical** | VerbalizedConfidence 0.771 | 0.703 | 0.674 | **0.737** | Direct |
| Math | MatrixDegreeConfidence 0.676 | 0.710 | 0.725 | **0.529** | DisAAD |
| Adversarial | VerbalizedConfidence 0.797 | 0.640 | 0.629 | **0.710** | Direct |
| Factual | VerbalizedConfidence 0.557 | 0.606 | 0.604 | **0.449** | DALD |

> Medical is the health-coaching target domain. `Ours` = best over students/variants; the pending corner-head cells will refine these.


## (b) Latency per method (p50)


**Proxy tier (target-decoupled):** median forward = **28 ms** (range 13–51 across student sizes) — **zero** extra target calls.


**Direct methods on qwen3-32b** (scoring p50; each also needs target generations):

| method | scoring p50 (ms) | extra target calls |
|---|--:|---|
| EccentricityConfidence | 794 | + N−1 target samples |
| MatrixDegreeConfidence | 793 | + N−1 target samples |
| EccentricityUncertainty | 483 | + N−1 target samples |
| SumEigenUncertainty | 480 | + N−1 target samples |
| EigV | 480 | + N−1 target samples |
| MatrixDegreeUncertainty | 480 | + N−1 target samples |
| KernelLanguageEntropy | 428 | + N−1 target samples |
| VerbalizedConfidence | 287 | +1 live target call |
| DiscreteSemanticEntropy | 210 | + N−1 target samples |
| NumSemanticSetUncertainty | 208 | + N−1 target samples |
| LexicalSimilarity | 183 | + N−1 target samples |

**Direct methods on llama3.3-70b** (scoring p50; each also needs target generations):

| method | scoring p50 (ms) | extra target calls |
|---|--:|---|
| MatrixDegreeConfidence | 820 | + N−1 target samples |
| EccentricityConfidence | 820 | + N−1 target samples |
| SumEigenUncertainty | 500 | + N−1 target samples |
| MatrixDegreeUncertainty | 499 | + N−1 target samples |
| EccentricityUncertainty | 499 | + N−1 target samples |
| EigV | 498 | + N−1 target samples |
| KernelLanguageEntropy | 443 | + N−1 target samples |
| VerbalizedConfidence | 286 | +1 live target call |
| NumSemanticSetUncertainty | 215 | + N−1 target samples |
| DiscreteSemanticEntropy | 214 | + N−1 target samples |
| LexicalSimilarity | 188 | + N−1 target samples |

> Even scoring-only, direct methods are 7–30× the ~27 ms proxy; adding the N−1 target samples (seconds on a 70B, ~19 s on an API target) is the dominant, target-scaling cost the proxy avoids entirely.

