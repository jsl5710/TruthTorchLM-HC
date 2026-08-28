# Multi-Target Proxy RQ — AUROC + Latency Across Targets

> Does the proxy advantage hold across target models and **widen on the larger/slower target**? Two teachers (Qwen3-32B, Llama-3.3-70B), each with a 3-student sweep. Proxy read-outs (DALD-au/eu, DisAAD-au/eu, Ours-EDL best config) are target-decoupled (~30 ms); direct methods run on the target (~750 ms). AUROC = mean over the 7-dataset subset.

**Auto-generated** by `scripts/mt_report.py`.


## Teacher: qwen3-32b

- best direct method: **VerbalizedConfidence** AUROC=0.616; mean direct latency p50 ≈ 799 ms


| student | DALD-au | DALD-eu | DisAAD-au | DisAAD-eu | edl·ecc λ1 | edl·ecc λ5 | edl·ecc λ10 | **Ours (best)** | best direct | proxy p50 ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qwen3-0.6b | 0.544 | 0.504 | 0.542 | 0.523 | — | 0.504 | — | **0.504** (edl·ecc·λ5) | 0.616 VerbalizedConfidence | 33 |
| qwen3-1.7b | 0.542 | 0.465 | 0.547 | 0.468 | 0.470 | 0.486 | 0.499 | **0.542** (head_ecc_lam5) | 0.616 VerbalizedConfidence | 31 |
| qwen3-4b | 0.560 | 0.513 | 0.547 | 0.529 | — | 0.539 | — | **0.550** (head_dse_lam5) | 0.616 VerbalizedConfidence | 42 |

## Teacher: llama3.3-70b

- best direct method: **VerbalizedConfidence** AUROC=0.707; mean direct latency p50 ≈ 823 ms


| student | DALD-au | DALD-eu | DisAAD-au | DisAAD-eu | edl·ecc λ1 | edl·ecc λ5 | edl·ecc λ10 | **Ours (best)** | best direct | proxy p50 ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| llama3.2-1b | 0.639 | 0.481 | 0.590 | 0.489 | — | 0.528 | — | **0.580** (head_ecc_lam5) | 0.707 VerbalizedConfidence | 18 |
| llama3.2-3b | 0.607 | 0.513 | 0.584 | 0.507 | 0.590 | 0.606 | 0.627 | **0.627** (ecc_lam10) | 0.707 VerbalizedConfidence | 26 |
| llama3.1-8b | 0.551 | 0.533 | 0.519 | 0.533 | — | 0.618 | — | **0.618** (edl·ecc·λ5) | 0.707 VerbalizedConfidence | 32 |

## Ours λ/oracle sweep — AUROC by config

| teacher | student | config (oracle_lamλ) | AUROC |
|---|---|---|--:|
| qwen3-32b | qwen3-0.6b | edl(default) | 0.504 |
| qwen3-32b | qwen3-0.6b | head_dse_lam1 | 0.501 |
| qwen3-32b | qwen3-0.6b | head_dse_lam5 | 0.502 |
| qwen3-32b | qwen3-0.6b | head_ecc_lam1 | 0.487 |
| qwen3-32b | qwen3-0.6b | head_ecc_lam5 | 0.487 |
| qwen3-32b | qwen3-1.7b | edl(default) | 0.486 |
| qwen3-32b | qwen3-1.7b | dse_lam1 | 0.467 |
| qwen3-32b | qwen3-1.7b | dse_lam10 | 0.489 |
| qwen3-32b | qwen3-1.7b | dse_lam2 | 0.463 |
| qwen3-32b | qwen3-1.7b | dse_lam5 | 0.469 |
| qwen3-32b | qwen3-1.7b | ecc_lam1 | 0.470 |
| qwen3-32b | qwen3-1.7b | ecc_lam10 | 0.499 |
| qwen3-32b | qwen3-1.7b | ecc_lam15 | 0.481 |
| qwen3-32b | qwen3-1.7b | ecc_lam2 | 0.473 |
| qwen3-32b | qwen3-1.7b | ecc_lam20 | 0.476 |
| qwen3-32b | qwen3-1.7b | ecc_lam5 | 0.486 |
| qwen3-32b | qwen3-1.7b | head_dse_lam1 | 0.539 |
| qwen3-32b | qwen3-1.7b | head_dse_lam5 | 0.525 |
| qwen3-32b | qwen3-1.7b | head_ecc_lam1 | 0.516 |
| qwen3-32b | qwen3-1.7b | head_ecc_lam5 | 0.542 |
| qwen3-32b | qwen3-4b | edl(default) | 0.539 |
| qwen3-32b | qwen3-4b | head_dse_lam1 | 0.540 |
| qwen3-32b | qwen3-4b | head_dse_lam5 | 0.550 |
| qwen3-32b | qwen3-4b | head_ecc_lam1 | 0.510 |
| qwen3-32b | qwen3-4b | head_ecc_lam5 | 0.535 |
| llama3.3-70b | llama3.2-1b | edl(default) | 0.528 |
| llama3.3-70b | llama3.2-1b | head_dse_lam1 | 0.417 |
| llama3.3-70b | llama3.2-1b | head_dse_lam5 | 0.426 |
| llama3.3-70b | llama3.2-1b | head_ecc_lam1 | 0.439 |
| llama3.3-70b | llama3.2-1b | head_ecc_lam5 | 0.580 |
| llama3.3-70b | llama3.2-3b | edl(default) | 0.607 |
| llama3.3-70b | llama3.2-3b | dse_lam1 | 0.582 |
| llama3.3-70b | llama3.2-3b | dse_lam10 | 0.619 |
| llama3.3-70b | llama3.2-3b | dse_lam2 | 0.578 |
| llama3.3-70b | llama3.2-3b | dse_lam5 | 0.594 |
| llama3.3-70b | llama3.2-3b | ecc_lam1 | 0.590 |
| llama3.3-70b | llama3.2-3b | ecc_lam10 | 0.627 |
| llama3.3-70b | llama3.2-3b | ecc_lam15 | 0.593 |
| llama3.3-70b | llama3.2-3b | ecc_lam2 | 0.597 |
| llama3.3-70b | llama3.2-3b | ecc_lam20 | 0.608 |
| llama3.3-70b | llama3.2-3b | ecc_lam5 | 0.606 |
| llama3.3-70b | llama3.2-3b | head_dse_lam1 | 0.433 |
| llama3.3-70b | llama3.2-3b | head_dse_lam5 | 0.442 |
| llama3.3-70b | llama3.2-3b | head_ecc_lam1 | 0.513 |
| llama3.3-70b | llama3.2-3b | head_ecc_lam5 | 0.499 |
| llama3.3-70b | llama3.1-8b | edl(default) | 0.618 |
| llama3.3-70b | llama3.1-8b | head_dse_lam1 | 0.414 |
| llama3.3-70b | llama3.1-8b | head_dse_lam5 | 0.428 |
| llama3.3-70b | llama3.1-8b | head_ecc_lam1 | 0.445 |
| llama3.3-70b | llama3.1-8b | head_ecc_lam5 | 0.412 |

**Read:** claim (a) = Ours-EDL (best config) ≥ DALD/DisAAD on each cell; claim (b) = the proxy's *latency* edge over direct methods grows with target size (a ~30 ms proxy forward is target-decoupled; direct methods pay the target's per-sample cost). If (a) fails even at the swept optimum, the honest claim is Pareto/latency dominance, not raw-AUROC dominance.

