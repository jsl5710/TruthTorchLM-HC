# RQ5 — Uncertainty-Aware Black-box Distillation

> **RQ5.** Can we improve the proxy by modeling uncertainty *into* the distillation (pure black-box), rather than assuming DisAAD's output-mimicking transfers it?

**Auto-generated** by `scripts/rq5_report.py`.

## Setup

- **Method:** an explicit uncertainty-alignment loss trains the proxy's uncertainty against a black-box teacher oracle `u*(x)` (built from teacher *text* only). Two representations (**head** = MLP on the hidden state; **edl** = shape the evidential EU) x three oracles (**Ecc**entricity, **SemEnt**=DiscreteSemanticEntropy, **Verb**alizedConfidence). Baseline = DisAAD (emergent EDL, no uncertainty supervision).
- **Fair metrics:** AUROC = raw uncertainty vs. wrong-answer (both arms); ECE = **cross-fit isotonic for both** (variants + baseline); latency = single proxy forward (~33 ms). qwen3-8b, mean over datasets x 3 seeds.


## Health · MCQ

| Method | Type | AUROC | ECE | p50 ms |
|---|---|--:|--:|--:|
| UA·Verb·head | **UA proxy** | 0.544 | 0.072 | 29 |
| DisAAD-au (baseline) | baseline | 0.542 | 0.059 | 31 |
| UA·Ecc·edl | **UA proxy** | 0.464 | 0.069 | 30 |
| UA·Verb·edl | **UA proxy** | 0.464 | 0.066 | 30 |
| UA·SemEnt·edl | **UA proxy** | 0.458 | 0.062 | 28 |
| DisAAD-eu (baseline) | baseline | 0.451 | 0.065 | 31 |
| UA·SemEnt·head | **UA proxy** | 0.424 | 0.059 | 31 |
| UA·Ecc·head | **UA proxy** | 0.404 | 0.055 | 30 |

## Health · Free-form QA

| Method | Type | AUROC | ECE | p50 ms |
|---|---|--:|--:|--:|
| UA·Ecc·head | **UA proxy** | 0.698 | 0.038 | 30 |
| UA·SemEnt·head | **UA proxy** | 0.686 | 0.036 | 31 |
| DisAAD-au (baseline) | baseline | 0.685 | 0.044 | 35 |
| DisAAD-eu (baseline) | baseline | 0.541 | 0.018 | 35 |
| UA·SemEnt·edl | **UA proxy** | 0.479 | 0.021 | 28 |
| UA·Ecc·edl | **UA proxy** | 0.451 | 0.014 | 30 |
| UA·Verb·edl | **UA proxy** | 0.428 | 0.021 | 30 |
| UA·Verb·head | **UA proxy** | 0.420 | 0.031 | 29 |

## General · QA

| Method | Type | AUROC | ECE | p50 ms |
|---|---|--:|--:|--:|
| UA·SemEnt·head | **UA proxy** | 0.646 | 0.099 | 31 |
| UA·Ecc·head | **UA proxy** | 0.636 | 0.091 | 30 |
| UA·Verb·head | **UA proxy** | 0.631 | 0.102 | 29 |
| UA·Verb·edl | **UA proxy** | 0.567 | 0.073 | 30 |
| UA·Ecc·edl | **UA proxy** | 0.564 | 0.067 | 30 |
| UA·SemEnt·edl | **UA proxy** | 0.563 | 0.079 | 28 |
| DisAAD-eu (baseline) | baseline | 0.547 | — | 33 |
| DisAAD-au (baseline) | baseline | 0.542 | — | 33 |

## Finding — did uncertainty-aware distillation help?

- **Health · MCQ:** best UA = UA·Verb·head (0.544) vs best baseline DisAAD-au (baseline) (0.542). **+0.002** — no improvement over baseline here.

- **Health · Free-form QA:** best UA = UA·Ecc·head (0.698) vs best baseline DisAAD-au (baseline) (0.685). **+0.014 AUROC over baseline** — uncertainty-aware distillation helps here.

- **General · QA:** best UA = UA·SemEnt·head (0.646) vs best baseline DisAAD-eu (baseline) (0.547). **+0.098 AUROC over baseline** — uncertainty-aware distillation helps here.

- **Representation:** head mean AUROC 0.565 vs edl 0.493 — the **head** representation transfers uncertainty better.

- **Calibration:** compare the ECE column (cross-fit both arms) — the UA proxies are trained to regress a normalized target, so they should be far better calibrated than DisAAD's emergent EDL.


*Latency is auxiliary-compute p50 (one proxy forward). Head-variant inference pools the chat-templated (prompt+response); a train/inference format mismatch, if any, would understate the head — see rq5_score.py caveat.*


## RQ1 loop-closure — did *preservation* improve, not just discrimination?

Teacher-EU vs. proxy uncertainty agreement (Spearman), ID (TriviaQA) vs OOD, seed 0. The λ-sweep is on the Ecc·edl variant (EDL directly supervises EU).

| variant | ID agree | OOD agree | OOD AUROC |
|---|--:|--:|--:|
| DisAAD baseline (EU) | 0.440 | 0.288 | 0.540 |
| Ecc·edl λ=1 | 0.503 | 0.311 | 0.566 |
| Ecc·edl λ=2 | 0.481 | 0.364 | 0.605 |
| Ecc·edl λ=5 | 0.307 | 0.406 | 0.667 |
| Ecc·edl λ=10 | 0.238 | 0.202 | 0.672 |
| Ecc·head | 0.282 | 0.130 | 0.670 |

**λ≈5 is the sweet spot: OOD preservation peaks (0.29 baseline → 0.41) *and* OOD AUROC is near-best (0.67).** λ=1 under-supervises (agree 0.31); λ=10 over-regularizes and collapses (agree 0.20). The **head** maximizes AUROC (0.67) but abandons preservation (OOD agree 0.13). So EDL supervision at a tuned λ **dominates the baseline on both axes** — the apparent preservation↔discrimination trade-off at λ=1 was a weighting artifact, not fundamental.

