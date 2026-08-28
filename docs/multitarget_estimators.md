# Generalized RQ2 — one read-out does NOT fit all

> Which logit estimator best reads a distilled proxy's uncertainty? All six estimators read the **proxy's** logits (the proxy is white-box by design — distilled from the teacher's *text* only; the teacher is never read). Across **all our students × methods × variants**, the best estimator is **dataset-dependent and stable across proxy methods** — not a single winner.

**Auto-generated** by `scripts/mt_estimators_report.py`. All estimators are logit-based; the axis is **softmax (Entropy/MSP/Energy) vs evidential (EDL/LogTokU)**, not logit vs text.


## (1) Estimator AUROC per dataset (mean across all proxies)

| dataset | domain | EDL-AU | EDL-EU | MSP | Entropy | Energy | LogTokU | winner | softmax>evid |
|---|---|--:|--:|--:|--:|--:|--:|---|:--:|
| trivia_qa | General | 0.594 | 0.484 | 0.597 | **0.603** | 0.557 | 0.460 | Entropy | ✅ |
| bioasq | Medical | 0.674 | 0.686 | 0.819 | **0.867** | 0.681 | 0.668 | Entropy | ✅ |
| medqa | Medical·MCQ | **0.565** | 0.375 | 0.489 | 0.487 | 0.416 | 0.443 | EDL-AU | ❌ |
| medlfqa | Medical·LF | 0.462 | **0.514** | 0.503 | 0.507 | 0.479 | 0.469 | EDL-EU | ❌ |
| gsm8k | Math | 0.695 | 0.450 | 0.734 | **0.749** | 0.680 | 0.502 | Entropy | ✅ |
| truthful_qa | Adversarial | 0.422 | **0.635** | 0.532 | 0.564 | 0.612 | 0.554 | EDL-EU | ❌ |
| wikipedia_factual | Factual | **0.598** | 0.417 | 0.518 | 0.502 | 0.466 | 0.409 | EDL-AU | ❌ |

_Softmax beats evidential on only **3/7** datasets — the aggregate 'Entropy wins' is driven mostly by bioasq (+0.18)._


## (2) Winning estimator TYPE by proxy method (S=softmax, E=evidential)

| dataset | DALD | DisAAD | Ours-edl | Ours-head |
|---|:--:|:--:|:--:|:--:|
| trivia_qa | **S**·Ent | E·AU | **S**·Ent | E·AU |
| bioasq | **S**·Ent | **S**·Ent | **S**·Ent | **S**·MSP |
| medqa | E·AU | E·AU | E·AU | E·AU |
| medlfqa | **S**·Ent | **S**·Ent | E·EU | E·EU |
| gsm8k | **S**·Ent | **S**·Ent | **S**·Ent | **S**·Ent |
| truthful_qa | E·EU | E·EU | E·EU | E·EU |
| wikipedia_factual | E·AU | E·AU | E·AU | E·AU |

_5/7 datasets share the same winner-type across all methods; trivia_qa and medlfqa are method-dependent (Ours leans evidential on medlfqa). Size (not shown) is mostly small-sample noise — only gsm8k (softmax) and wikipedia_factual (evidential) are size-stable._


## (3) Best-per-dataset read-out (used for the fair comparison)

| dataset | General | Medical | Medical·MCQ | Medical·LF | Math | Adversarial | Factual |
|---|---|---|---|---|---|---|---|
| **read-out** | Entropy | Entropy | EDL-AU | EDL-EU | Entropy | EDL-EU | EDL-AU |

## (4) FAIR comparison — every proxy read with the best-per-dataset estimator

| proxy family | fair AUROC (best-per-dataset read-out) | n | best |
|---|--:|--:|--:|
| DALD | **0.635** | 6 | 0.671 |
| DisAAD | **0.626** | 6 | 0.659 |
| Ours-edl | **0.631** | 24 | 0.658 |
| Ours-head | **0.630** | 13 | 0.657 |

> Under a common, *fair* read-out policy, the proxy families are close — the native-read-out gaps in the main table were substantially a **read-out artifact** (DisAAD/DALD were shown with their weakest read-out, LogTokU). The durable claims are **latency/Pareto dominance** and the **per-dataset read-out finding itself** (one estimator does not fit all).

