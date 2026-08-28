# RQ2 — Uncertainty Estimator Evaluation

> **RQ2.** Does Evidential Deep Learning (EDL) provide superior uncertainty estimation compared with established logit-based estimators when applied to the **same proxy logits**?

**Auto-generated** by `scripts/rq2_report.py` (reuses the proxy-side logits from `rq1_preservation.py`).

## Setup — isolate the estimator, hold the input fixed

- **Fixed input:** the distilled **qwen3-0.6b proxy** logits over each cached (prompt + response) — identical for every estimator, so any difference is the *estimator's* contribution, not the representation.
- **Estimators:** EDL-AU, EDL-EU (evidential, top-k Dirichlet) · MSP (1−max softmax) · predictive Entropy · Energy (−logsumexp) · LogTokU (full-vocab evidential*). All oriented higher = more uncertain, mean-aggregated over the response span.
- **Quality:** AUROC & AUPR at flagging wrong answers (task correctness = LLM-judge / MCQ label); calibration = cross-fitted isotonic ECE. `✦` marks the EDL family.
- *LogTokU is a best-effort implementation of Ma et al. 2025 (full-vocabulary evidential epistemic); flagged `*` as 'where applicable'.
- Pooled over 3 seeds; 4032 proxy responses.


### All datasets (proxy logits)

| Estimator | Family | AUROC | AUPR | ECE |
|---|---|--:|--:|--:|
| Entropy | softmax | 0.590 | 0.454 | 0.006 |
| MSP | softmax | 0.575 | 0.427 | 0.003 |
| Energy | energy | 0.564 | 0.412 | 0.012 |
| EDL-EU ✦ | evidential | 0.555 | 0.405 | 0.019 |
| EDL-AU ✦ | evidential | 0.524 | 0.369 | 0.008 |
| LogTokU | evidential* | 0.493 | 0.343 | 0.002 |

### Health · MCQ

| Estimator | Family | AUROC | AUPR | ECE |
|---|---|--:|--:|--:|
| EDL-AU ✦ | evidential | 0.541 | 0.368 | 0.029 |
| MSP | softmax | 0.520 | 0.364 | 0.017 |
| Entropy | softmax | 0.513 | 0.369 | 0.022 |
| Energy | energy | 0.502 | 0.335 | 0.015 |
| LogTokU | evidential* | 0.470 | 0.306 | 0.025 |
| EDL-EU ✦ | evidential | 0.459 | 0.306 | 0.016 |

### Health · Free-form QA

| Estimator | Family | AUROC | AUPR | ECE |
|---|---|--:|--:|--:|
| Energy | energy | 0.678 | 0.193 | 0.014 |
| EDL-AU ✦ | evidential | 0.663 | 0.177 | 0.015 |
| MSP | softmax | 0.638 | 0.185 | 0.007 |
| Entropy | softmax | 0.620 | 0.167 | 0.010 |
| LogTokU | evidential* | 0.577 | 0.151 | 0.012 |
| EDL-EU ✦ | evidential | 0.518 | 0.118 | 0.015 |

### General · QA

| Estimator | Family | AUROC | AUPR | ECE |
|---|---|--:|--:|--:|
| Entropy | softmax | 0.586 | 0.616 | 0.016 |
| MSP | softmax | 0.570 | 0.597 | 0.013 |
| Energy | energy | 0.537 | 0.581 | 0.009 |
| EDL-EU ✦ | evidential | 0.530 | 0.563 | 0.014 |
| EDL-AU ✦ | evidential | 0.523 | 0.554 | 0.013 |
| LogTokU | evidential* | 0.465 | 0.506 | 0.015 |

## Finding — is EDL superior?

- On the same proxy logits, the best non-EDL estimator (**Entropy 0.590**) **beats** the best EDL measure (0.555) overall — EDL is **not** superior once the input representation is controlled; its reported edge in the paper may be an input-confound, not an estimator win.

- Compare AU vs EU: they read different failure modes (data-ambiguity vs evidence-scarcity); the winner shifts by task type (see the per-group tables).

- ECE columns show which estimator is best *calibrated* on the proxy logits, independent of ranking ability.

