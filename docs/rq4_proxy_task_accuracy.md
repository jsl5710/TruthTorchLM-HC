# RQ4 — Proxy Task Accuracy (companion to RQ1)

> **RQ4.** Is the distilled **proxy** (student `qwen3-0.6b`) as **accurate on the tasks** as the **teacher** (`qwen3-8b`) it was distilled from — across QA, MCQ, and health datasets?

**Linked to RQ1:** RQ1 asks whether the proxy preserves the teacher's *uncertainty*; RQ4 is the counterpart — whether it preserves the teacher's *task accuracy*. Read together they separate 'good uncertainty proxy' from 'good answerer'.

Proxy run as a generator (greedy, thinking off), judged identically to the teacher (MCQMatch for MCQ, Qwen3-4B judge otherwise). Mean over 3 seeds. **Auto-generated** by `scripts/proxy_accuracy_report.py`.


## Health — MCQ

| Dataset | Proxy acc | Teacher acc | Δ (proxy−teacher, pts) |
|---|--:|--:|--:|
| MedQA | 34.0% | 67.3% | -33.3 |
| MMLU-Med | 41.3% | 70.7% | -29.3 |

## Health — Free-form QA

| Dataset | Proxy acc | Teacher acc | Δ (proxy−teacher, pts) |
|---|--:|--:|--:|
| K-QA | 40.7% | 77.2% | -36.6 |
| MedLFQA | 42.7% | 88.7% | -46.0 |
| BioASQ | 81.3% | 99.3% | -18.0 |

## General — QA

| Dataset | Proxy acc | Teacher acc | Δ (proxy−teacher, pts) |
|---|--:|--:|--:|
| TriviaQA | 14.8% | 43.6% | -28.9 |
| NaturalQA | 13.5% | 50.7% | -37.2 |
| PopQA | 24.0% | 38.0% | -14.0 |
| TruthfulQA | 23.3% | 52.7% | -29.4 |

## Summary

- **Health · MCQ:** proxy 37.7% vs teacher 69.0% (Δ -31.3 pts)

- **Health · Free-form QA:** proxy 54.9% vs teacher 88.4% (Δ -33.5 pts)

- **General · QA:** proxy 18.9% vs teacher 46.2% (Δ -27.3 pts)


**Overall:** mean proxy−teacher accuracy gap = **-30.3 pts**. Is the 0.6B proxy as accurate as the 8B teacher? **no** — the proxy is 30.3 pts below the teacher on average. Distillation transfers *outputs* on the distillation domain (TriviaQA), but this measures whether task *accuracy* holds across all task types — the larger the gap, the more the proxy is a good *uncertainty* proxy without being a good *answerer*.

