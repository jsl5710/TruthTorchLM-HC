# Open vs Closed targets — the comparison arm

> The study distills proxies for **open** targets (qwen3-32b, llama-70b). The **closed** frontier targets (gpt-4o, claude-haiku) are the reference: direct UQ needs N≈10 samples from the target, and on an API target each is a ~1–2 s round-trip — so direct UQ costs **seconds per item**, while a distilled proxy stays **~27 ms** (target-decoupled). Closed targets run the consistency direct methods only (no API-teacher proxy). PTrue excluded (grey-box).

**Auto-generated** by `scripts/mt_report_closed.py`.


| target | type | best pure-BB direct (AUROC) | target gen / call | direct UQ gen cost (≈N calls) | proxy |
|---|---|---|--:|--:|--:|
| qwen3-32b (open, 32B) | open | VerbalizedConfidence 0.616 | — | **—** | 27 ms |
| llama-3.3-70b (open, 70B) | open | VerbalizedConfidence 0.707 | — | **—** | 27 ms |
| gpt-4o (closed, API) | closed | EccentricityConfidence 0.582 | 1121 ms | **≈11.2 s** | 27 ms |
| claude-haiku-4.5 (closed, API) | closed | MatrixDegreeUncertainty 0.524 | 1857 ms | **≈18.6 s** | 27 ms |

**Read:** on the closed API targets, direct UQ pays **~11–19 s of target generation per item** (N samples × 1–2 s/call) before any scoring; the proxy needs **zero** target calls and one ~27 ms forward — a **~400–700× latency gap** that only widens with target cost. AUROC-wise the consistency methods transfer to the API targets (see per-target rows). The open targets' per-call generation latency is not cached (hc_benchmark schema); a small measurement job can fill it, but the proxy's 27 ms is target-independent regardless.


### qwen3-32b (open, 32B) — direct methods (pure-BB)

| method | AUROC |
|---|--:|
| VerbalizedConfidence | 0.616 |
| MatrixDegreeConfidence | 0.598 |
| EccentricityConfidence | 0.597 |
| SumEigenUncertainty | 0.587 |
| EigV | 0.587 |
| MatrixDegreeUncertainty | 0.587 |

### llama-3.3-70b (open, 70B) — direct methods (pure-BB)

| method | AUROC |
|---|--:|
| VerbalizedConfidence | 0.707 |
| MatrixDegreeConfidence | 0.634 |
| EccentricityConfidence | 0.625 |
| EccentricityUncertainty | 0.622 |
| LexicalSimilarity | 0.609 |
| MatrixDegreeUncertainty | 0.601 |

### gpt-4o (closed, API) — direct methods (pure-BB)

| method | AUROC |
|---|--:|
| EccentricityConfidence | 0.582 |
| LexicalSimilarity | 0.578 |
| EccentricityUncertainty | 0.575 |
| MatrixDegreeConfidence | 0.563 |
| SumEigenUncertainty | 0.549 |
| EigV | 0.549 |

### claude-haiku-4.5 (closed, API) — direct methods (pure-BB)

| method | AUROC |
|---|--:|
| MatrixDegreeUncertainty | 0.524 |
| SumEigenUncertainty | 0.524 |
| EigV | 0.524 |
| MatrixDegreeConfidence | 0.514 |
| EccentricityUncertainty | 0.509 |
| EccentricityConfidence | 0.507 |
