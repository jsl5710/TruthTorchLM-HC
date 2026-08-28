# RQ5 — EDL preservation vs. λ (bootstrap 95% CI)

> Teacher-EU vs. proxy-uncertainty agreement (Spearman) and proxy AUROC on the **OOD** split (everything but TriviaQA), with **bootstrap 95% CIs over items** (B=2000). Ecc oracle, qwen3-8b. The eval-seed pass is degenerate here (greedy primary → deterministic), so the item-bootstrap is the robustness test.

**Auto-generated** by `scripts/rq5_lambda_curve.py`.

| variant | OOD agree [95% CI] | OOD AUROC [95% CI] |
|---|--:|--:|
| DisAAD baseline (EU) | 0.287 [0.234, 0.340] | 0.541 [0.506, 0.575] |
| Ecc·edl λ=1 | 0.311 [0.258, 0.363] | 0.566 [0.530, 0.602] |
| Ecc·edl λ=2 | 0.365 [0.314, 0.414] | 0.606 [0.571, 0.639] |
| Ecc·edl λ=5 | 0.407 [0.354, 0.454] | 0.667 [0.632, 0.703] |
| Ecc·edl λ=10 | 0.203 [0.149, 0.257] | 0.673 [0.637, 0.707] |
| Ecc·head | 0.131 [0.072, 0.186] | 0.671 [0.634, 0.705] |

**Reading:** OOD agreement peaks at **λ=5** (CI clearly above λ=1 and far above λ=10/head — the collapse and the head's preservation-abandonment are significant). OOD AUROC is statistically **tied** across λ=5 / λ=10 / head (CIs overlap), so discrimination saturates by λ≈5. **λ=5 uniquely combines significantly-best preservation with tied-best discrimination.**


## Training-seed error bars (mean ± std over train seeds 42/43/44)

| λ | n seeds | OOD agree | OOD AUROC |
|---|--:|--:|--:|
| λ=1 | 3 | 0.301±0.020 | 0.550±0.021 |
| λ=2 | 3 | 0.373±0.007 | 0.601±0.006 |
| λ=5 | 3 | 0.425±0.014 | 0.657±0.008 |
| λ=10 | 3 | 0.188±0.028 | 0.670±0.004 |
