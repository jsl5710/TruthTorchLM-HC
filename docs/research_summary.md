# Research Summary — Proxy-Based Black-box Uncertainty (RQ1–RQ5)

A five-question arc investigating whether a **distilled proxy** can deliver fast, reliable
**black-box** uncertainty quantification — diagnosing where DisAAD's proxy falls short (RQ1–RQ4),
then fixing its central flaw (RQ5).

## The setting

- **Pure black box:** a UQ method sees the target's *text output only* — no logits, no weights.
- **The proxy idea (DisAAD):** distil a small open **proxy** (`Qwen3-0.6B`) from a black-box
  **teacher** (`Qwen3-8B`), then read uncertainty off the proxy's logits in **one forward pass
  (~33 ms)** — vs. *direct* methods that draw N samples from the target (~700 ms) or elicit a
  verbalized confidence.
- **Fair target:** the proxy is per-target (distilled from `qwen3-8b`), so **every comparison is on
  qwen3-8b** — the only apples-to-apples target. (The multi-generator frontier *overstates* the
  proxy; see RQ3.)

## The five questions

| RQ | Question | Answer |
|---|---|---|
| **RQ1** | Does the proxy **preserve** the teacher's uncertainty (ID & OOD)? | **Mostly — except epistemic (EU), which collapses OOD** (teacher-agreement 0.44→0.29). The proxy inherits the teacher's *outputs*, not its *uncertainty*. |
| **RQ2** | Is **EDL** the best estimator on the *same* proxy logits? | **No.** Plain **Entropy (0.590)** and **MSP (0.575)** beat EDL-EU (0.555) / EDL-AU (0.524). DisAAD's reported EDL edge was an **input-confound** (EDL-on-proxy vs. others-on-target), not an estimator win. |
| **RQ3** | Does the proxy's **efficiency** justify any **quality** loss vs. direct methods? | **A trade-off.** 3–92× faster, but −0.09 to −0.18 AUROC. **Favorable on health free-form QA** (0.698 head vs 0.685 best-direct), **poor on QA/MCQ**. |
| **RQ4** | Is the proxy as **accurate on the tasks** as the teacher? *(companion to RQ1)* | **No — −30 pts across the board** (fluent hallucination, even on the TriviaQA distillation domain). A good *uncertainty* proxy ≠ a good *answerer* — which is fine: DisAAD deploys it only for uncertainty. |
| **RQ5** | Can **uncertainty-aware distillation** (pure black-box) fix RQ1? | **Yes.** Adding an explicit uncertainty-alignment loss, **EDL-supervised at λ≈5, closes the EU-preservation gap (OOD agree 0.29→0.41) *and* improves discrimination (OOD AUROC 0.54→0.67)** — dominating DisAAD on both axes at the same 33 ms. |

## The arc: diagnose → fix

**RQ1–RQ4 diagnose** a proxy that is fast but flawed on four distinct axes:
- it does not **preserve** the teacher's uncertainty off-distribution (RQ1),
- its default **estimator** (EDL) is not the best on its own logits (RQ2),
- it trades **quality for speed** vs. direct methods (RQ3),
- and it cannot **answer** the task (RQ4).

**RQ5 fixes the central one.** DisAAD only ever *mimics outputs* and hopes uncertainty follows.
Instead, model uncertainty **into** the distillation: build a black-box teacher-uncertainty label
`u*(x)` from teacher *text* (a top UQ method over N samples) and train the proxy's uncertainty to
match it. Two representations, three oracles, all pure black-box:

- **Head** (MLP on the hidden state) → maximizes **discrimination** (OOD AUROC 0.67) but **abandons
  preservation** (OOD agree 0.13) — it learns a *new* error-predictor, not the teacher's signal.
- **EDL-target** (shape the evidential EU) → **preserves** the teacher's uncertainty; and with the
  weight tuned (**λ≈5**) it *also* matches the head's discrimination. λ=1 under-supervises; λ=10
  over-regularizes and collapses. **The apparent preservation↔discrimination trade-off was a
  λ-artifact, not fundamental.**

## Cross-cutting findings

1. **Same-target fairness is essential.** Comparing a per-target proxy against multi-generator
   averages overstates it; on qwen3-8b the story is a trade-off, not dominance (RQ3).
2. **Three distinct axes, handled differently:** *task accuracy* (RQ4), *error-discrimination*
   (RQ3/AUROC), and *teacher-uncertainty preservation* (RQ1/agreement) are not the same thing — the
   proxy can be strong on one and weak on another.
3. **Representation drives the outcome** more than the oracle: all three oracles' *head* variants
   beat the baseline on QA (~+0.09); the choice of head vs. EDL decides preservation-vs-discrimination.
4. **Domain matters:** methods that win open QA lose on health/MCQ and vice-versa — there is no
   universal best UQ method; the frontier is regime-dependent.
5. **Distillation transfers style, not knowledge** (RQ4: BLEU 0.95 to the teacher, yet −30 pts task
   accuracy) — and, by default, not uncertainty either (RQ1) unless you supervise it (RQ5).

## The headline

> **A pure-black-box proxy can be trained to both preserve the teacher's uncertainty and detect its
> errors, in a single 33 ms forward pass — but only if uncertainty is an explicit, correctly-weighted
> training target (EDL-supervised, λ≈5). Mimicking outputs alone (DisAAD) is not enough.**

## Artifacts

- Per-RQ docs: `rq1_uncertainty_preservation.md`, `rq2_estimator_evaluation.md`,
  `rq3_efficiency_quality.md`, `rq4_proxy_task_accuracy.md`, `rq5_uncertainty_aware_distillation.md`.
- Pipelines: `scripts/rq1_preservation.py`, `rq2_report.py`, `rq3_report.py`,
  `proxy_task_accuracy.py`, `rq5_build_labels.py` + `train_disaad.py` (uncertainty-aware) +
  `rq5_score.py` / `rq5_preservation.py` / `rq5_report.py`.
- All on `qwen3-8b`, mean over 3 seeds; latency = auxiliary-compute p50, batch-1, GPU-warm.
