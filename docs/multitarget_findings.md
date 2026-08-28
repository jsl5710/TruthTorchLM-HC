# Multi-Target Proxy UQ — Consolidated Findings

**Setup.** Two axes of target: **open** teachers (Qwen3-32B, Llama-3.3-70B) and **closed** frontier
API teachers (GPT-4o, Claude-Haiku-4.5). For each, a **3-student sweep** of distilled proxies
(Qwen3 {0.6B,1.7B,4B}; Llama {1B,3B,8B}) × **3 objectives**: `DALD` (masked SFT), `DisAAD`
(SFT+adversarial), `Ours` (SFT+adversarial+**uncertainty-aware**). Eval = **7 datasets**, one per
category (General/Medical×3/Math/Adversarial/Factual), 150 items each. Pure **black-box**: methods
see only the target's *text*; the proxy is white-box by design (distilled from teacher text only).

---

## Results (consolidated)

Four tables that survive read-out control and LODO. Detailed analysis in §1–§8. All AUROC; the two
best-student cells (Qwen3-32B→qwen3-4b, Llama-3.3-70B→llama3.2-3b).

**Table 1 — Proxy comparison, three tiers of honesty (mean AUROC).** Ceiling = per-dataset
config·read-out cherry-pick (a *mirage* — fails LODO, §6c); best-var = one fixed Ours config; deploy =
`base · Perplexity`, chosen a priori.

| | Qwen ceiling | Qwen best-var | **Qwen deploy** | Llama ceiling | Llama best-var | **Llama deploy** |
|---|--:|--:|--:|--:|--:|--:|
| **Ours** | 0.690 | 0.616 | 0.612 | 0.715 | 0.626 | 0.609 |
| **DALD** | 0.668 | 0.604 | 0.603 | 0.697 | **0.632** | **0.632** |
| **DisAAD** | 0.675 | **0.613** | **0.613** | 0.683 | 0.611 | 0.611 |

The ceiling leads Ours (+0.02); **at any deployable setting the three tie** (Qwen→DisAAD, Llama→DALD) —
the objective does not separate them (§3, §6).

**Table 2 — Deployable per-dataset (single-best fixed config·read-out).** Qwen: Ours=`head_ecc_λ1·Ppl`,
DALD=`·Entropy`, DisAAD=`·Ppl`. Llama: all `·Perplexity`.

| dataset | Qwen Ours | Qwen DALD | Qwen DisAAD | Llama Ours | Llama DALD | Llama DisAAD |
|---|--:|--:|--:|--:|--:|--:|
| trivia (General) | 0.623 | 0.605 | **0.634** | **0.664** | 0.631 | 0.647 |
| bioasq (Medical) | — | — | — | 0.987 | **0.993** | 0.987 |
| medqa (Med-MCQ) | **0.713** | 0.578 | 0.690 | 0.528 | **0.600** | 0.452 |
| medlfqa (Med-LF) | **0.526** | 0.519 | 0.500 | 0.448 | **0.450** | 0.444 |
| gsm8k (Math) | 0.773 | **0.854** | 0.781 | **0.696** | 0.694 | 0.693 |
| truthful_qa (Adversarial) | 0.477 | **0.510** | 0.477 | **0.554** | 0.551 | 0.554 |
| wikipedia (Factual) | 0.585 | 0.560 | **0.598** | 0.506 | **0.507** | 0.499 |
| **MEAN** | **0.616** | 0.604 | 0.613 | 0.626 | **0.632** | 0.611 |

**Latency (measured p50, batch = 1)** — identical across objectives (one forward, same student):

| student | Ours | DALD | DisAAD |
|---|--:|--:|--:|
| qwen3-4b | 41.4 ms | 43.1 ms | 44.3 ms |
| llama3.2-3b | 24.1 ms | 31.9 ms | 34.1 ms |

vs direct methods (full target generation): seconds on a 70B, ≈11 s (GPT-4o) / ≈19 s (Haiku) per item →
**~400–700× cheaper**.

**Table 3 — Read-out finding (best single read-out on Ours-base, mean AUROC).** Read-out ≫ objective; a
likelihood read-out (Perplexity) beats all six shape-estimators, no retraining (§3).

| read-out | Qwen | Llama |
|---|--:|--:|
| EDL-AU (old default) | 0.510 | 0.590 |
| Entropy | 0.607 | 0.591 |
| **Perplexity** (new) | **0.612** | **0.609** |
| MaxNLL (new) | 0.552 | 0.602 |

**Table 4 — Best *direct* pure-BB method (mean AUROC).** Target-coupled (pays a generation).

| | Qwen | Llama |
|---|--:|--:|
| VerbalizedConfidence (N=1) | 0.616 | **0.707** |
| spectral-consistency @N10 | 0.634 | 0.677 |
| *PTrue (grey-box — excluded)* | *0.662* | *0.659* |

VerbalizedConfidence dominates Medical+Adversarial (Llama 0.778) but costs a full target call; the proxy
trades ~0.1 AUROC for the latency above.

---

## 1. Headline — a domain division of labour, exploited by a router

The single-number aggregate is misleading. **Per domain**, uncertainty-aware and logit-based proxies
are **complementary**:

| domain | best-proxy winner | note |
|---|---|---|
| **Medical** (bioasq, medqa, medlfqa) | **Ours** | decisive on Llama-70B teacher; baselines fall *below random* on large students |
| **Adversarial** (truthful_qa) | **Ours** (open-Llama) / DALD-*eu* (open-Qwen) | Ours is confident-wrong-aware |
| General (trivia) | DALD/DisAAD | |
| Math (gsm8k) | DALD/DisAAD | simple consistency suffices |
| Factual (wikipedia) | DALD/DisAAD | |

**The payoff — a domain-router** (send Medical + Adversarial queries to `Ours-base`, the rest to
`DALD-au`) **beats either method alone**, keying only on the *query domain* (known at inference):

| | Qwen-32B teacher | Llama-70B teacher |
|---|--:|--:|
| DALD-au alone | 0.557 | 0.599 |
| Ours-base alone | 0.510 | 0.585 |
| **Domain-router** | **0.576** (+0.019) | **0.662** (+0.063) |
| oracle ceiling (max/dataset) | 0.617 | 0.676 |

The router captures ~93% of the oracle ceiling on Llama, at ~27 ms (both are the same proxy class).

> **Caveat (see §3): this domain split was measured with each proxy's *native* read-out.** Once every
> proxy is read with **Perplexity** (the best read-out), the Ours advantage on Medical+Adversarial
> **evaporates** — DALD·Perplexity ties Ours on Qwen and *beats* it on Llama, incl. Med/Adv. So the
> durable claim is the **latency/Pareto** story and the **read-out** finding, not "Ours owns Med/Adv."

---

## 2. Which "Ours" config? — BASE is the deployable one

The best Ours config is **(student-size × domain)-dependent**, but `base` (edl · Eccentricity · λ5)
is the single best deployable choice — most wins overall, and it **owns the large-student regime**:

- **Large students (l8b, q4b): `base` dominates** (l8b: base wins General, Medical **0.99**, MCQ,
  Adversarial). The base uncertainty signal is strong enough that tuning doesn't help.
- **Small students need tuning** (head-MLP or higher λ): head_ecc on MCQ, ecc_λ10 on General/Adv.
- **Domain signatures:** Medical → `base`/`dse_λ10`; MCQ → `head_ecc`; Factual → `head_dse`.
- **`head` is fragile** — it *inverts* on bioasq (AUROC → 0.03) on some cells; `base` is robust.

→ Report **`base`** as "Ours"; the head/λ configs are a marginal per-(size,domain) refinement,
not a better universal method.

---

## 3. The read-out matters more than the objective (generalized RQ2)

Scanning all 6 logit estimators over 49 distilled proxies' own logits: **no single estimator wins;
the best is dataset-dependent and stable across proxy method** (softmax Entropy for open free-form QA;
evidential EDL for MCQ/adversarial/factual). **LogTokU — the DisAAD/DALD default — is near-worst
everywhere.** Under a *fair* best-per-dataset read-out, the four proxy families **tie (~0.63)** — i.e.
the native-read-out gaps were largely a read-out artifact, and uncertainty-aware training did not
produce better *logits*, just ones better read by EDL on some domains. (All estimators read the
**proxy's** logits — legitimate; the teacher is never read.)

**Update — a *likelihood* read-out beats them all, and it's free.** Adding three fast likelihood-based
read-outs that use *which token the target emitted* (not just the distribution shape) — **Perplexity**
(mean NLL), **MaxNLL**, **LogitMargin** — **Perplexity is the single best read-out on the deployable
Ours-base proxy for BOTH teachers** (Qwen **0.612**, Llama **0.609** mean AUROC), beating all six
shape-estimators (old EDL-AU default: 0.510 / 0.590). So the concrete win is a **read-out swap on the
same proxy — no retraining**. Two corollaries: (a) the best-performing tuned configs (`head_dse_lam5`,
`ecc_lam10`) do **not** beat base once both are read with Perplexity → **use Ours-base, read via
Perplexity**, the config sweep is unnecessary; (b) on the router's Ours domains (Medical+Adversarial),
Perplexity lifts the mean +0.107 (Qwen, medqa 0.52→**0.71**) / +0.068 (Llama, bioasq 0.67→**0.99**),
though not uniformly (Llama medqa 0.57→0.42) — so it's the best *default*, but the per-domain winner
still varies.

**...and under that fair Perplexity read-out, "Ours owns Medical+Adversarial" does NOT survive.**
Reading **all three proxies with Perplexity** (apples-to-apples, best-student cells): on Qwen it's a
three-way tie (Ours 0.612 / DALD 0.603 / DisAAD 0.613; Med/Adv 0.565 / 0.551 / 0.555 — Ours marginally
best). On **Llama, DALD·Perplexity (0.632) beats Ours·Perplexity (0.609), including on Med/Adv (0.649
vs 0.602)** — the domains §1 credited to Ours. Perplexity lifts DALD's Med/Adv from 0.561 (native au)
to 0.649 (**+0.088 — it helps DALD *more* than Ours**); the earlier Ours-Med/Adv lead was the artifact
of comparing Ours·Perplexity vs DALD·native-*au*. Net: uncertainty-aware training did not yield better
*logits*, and once the read-out is controlled with Perplexity, **DALD ≥ Ours**. The safety-domain
champion is actually **VerbalizedConfidence** (direct; Med/Adv 0.608 Qwen / **0.778** Llama), not a proxy.
(Variant note: under Perplexity all Ours configs cluster at 0.61±0.01 — read-out ≫ variant; `head_ecc`
edges base by +0.004/+0.017, within noise on Qwen.)

**Grey-box comparison (target logits + PTrue) — the proxy loses almost nothing.** Running the same 9
read-outs on the **target's own logits** (grey-box ceiling): fair best-per-ds lift is **−0.011 (Qwen)
/ −0.014 (Llama)** — our black-box proxy's logits are *as informative as the target's*. The only
grey-box method that beats the best proxy read-out is **PTrue (0.66 vs Perplexity 0.61, ≈+0.05)**.
A neat mechanism confirms the proxy: **Perplexity is *weaker* on the target (0.570 / 0.578) than on the
proxy (0.612 / 0.609)** — the target's *self*-perplexity is contaminated (it generated the answer),
whereas the proxy's *cross*-perplexity is a clean error signal. (Scripts: `mt_estimators.py`
[+`--no-adapter/--load-in-4bit`], `mt_est_bbgb.sh`, `mt_est_bb_vs_gb.py`.)

---

## 4. Teacher-dependence — it does NOT transfer to frontier API teachers

| teacher | Ours best-proxy wins (of 7 domains) |
|---|---|
| Llama-3.3-70B (open) | **4** — all 3 Medical + Adversarial, deployably (base) |
| Qwen3-32B (open) | 2 — Medical-LF + General; adversarial flips to DALD-*eu* |
| **GPT-4o** (API) | **1** — Adversarial only |
| **Claude-Haiku** (API) | **0** |

On frontier API teachers **DALD dominates** (GPT-4o Math 0.820) — a better teacher yields better
distilled logits that LogTokU-au reads cleanly. Caveat: the **closed Ours ran the base config only**
(no head/λ sweep), so it is *under-tuned* vs the open arm.

---

## 5. Latency — the universal claim

Every proxy read-out is **~27 ms** (target-decoupled: one small-model forward). Direct methods pay
the target's per-sample cost — **~180–820 ms of NLI scoring PLUS N target generations**: seconds on a
70B, **≈11 s (GPT-4o) / ≈19 s (Haiku) per item** on API targets. **~400–700× cheaper**, widening with
target cost. This holds regardless of the AUROC story.

---

## 6. Trying to beat the router — three negative results (a common cause)

We tried three ways to push past a single fixed choice on the best-student cells (Qwen3-32B→qwen3-4b,
Llama-70B→llama3.2-3b). **All failed for the same reason: any selection/combination fit across our
7 small datasets (n≈110–150) does not transfer out-of-sample.** The oracle ceiling (per-dataset best,
0.68–0.70) is a **mirage** — it evaporates under leave-one-dataset-out (LODO).

**(a) A learned 5-signal score-level stacker loses to the hard router.** A leave-one-dataset-out
logistic stacker over all five read-outs per item `[Ours-EDL, DALD-au, DALD-eu, DisAAD-au,
DisAAD-eu] → P(incorrect)`:

| | Qwen-32B → q4b | Llama-70B → l3b |
|---|--:|--:|
| best single (DALD-au) | 0.570 | 0.607 |
| hard domain-router | **0.591** | **0.692** |
| **LODO learned stacker** | 0.558 | 0.648 |
| oracle ceiling | 0.640 | 0.703 |

The stacker **loses 0.03–0.04 AUROC**. It only wins where a held-out domain *resembles* a training
domain; it collapses on the idiosyncratic ones. The reason is exactly the read-out finding (§3): the
right signal-weighting **flips by domain**, so a global combiner can't transfer to an unseen domain.
The router encodes the domain prior explicitly (keyed on the query domain, free at inference) — that
information is **load-bearing and can't be recovered from the five scores alone**. Score-level stacking
helps *within* known domains, not *across* them.

**(b) VerbalizedConfidence as the distillation oracle fails — a good *direct* detector makes a poor
*distillation target*.** VC is the best pure-BB **direct** method (Llama 0.707; §5-adjacent), so we
distilled `Ours` toward it (`--oracle VerbalizedConfidence`, edl, λ5) instead of Eccentricity. Even on
**Llama — the strong-VC teacher — it underperforms the base**:

| Llama-70B → l3b | base (ecc) | VC-oracle | direct VC |
|---|--:|--:|--:|
| MEAN AUROC | **0.607** | 0.583 | 0.707 |

VC-oracle lands **−0.024 below base** and **−0.124 below the direct signal it was distilled from** —
distillation recovers only ⅓ of it. It's a *leveling* effect (helps where base is weak: medqa, medlfqa,
gsm8k, wikipedia; hurts where base is strong: bioasq 0.99→0.83, truthful_qa, trivia). This confirms the
single-target RQ5 verdict (VC was the *worst* oracle there: edl 0.498, head 0.541 vs Ecc/DSE ~0.61) on
the best-case teacher. **Mechanism:** VC is an *introspective scalar* — its discrimination lives in the
teacher knowing itself, which doesn't survive distillation into a small proxy's logits; Ecc/DSE come
from sample **dispersion**, which the proxy *can* regress. (The Qwen control was unusable — 4-bit VC
elicitation collapsed to a constant label.) **VerbalizedConfidence stays useful only as a direct
signal, never as an oracle.**

**(c) A domain→read-out router (and, by extension, a student MoE) fails LODO — worse than a fixed
read-out.** Since the best read-out is dataset-dependent (§3), the natural idea is to route each query
to its domain's best read-out (or, at model level, to a domain-specialised expert). LODO-tested on a
fixed proxy (read-out chosen per query-type from the *other* datasets):

| teacher :: proxy | always-Perplexity (fixed) | GLOBAL-best read-out (LODO) | **type-router (LODO)** | oracle |
|---|--:|--:|--:|--:|
| Qwen :: DALD | 0.603 | 0.531 | **0.502** | 0.668 |
| Qwen :: Ours-base | 0.612 | 0.587 | **0.503** | 0.679 |
| Llama :: DALD | 0.632 | 0.602 | **0.560** | 0.697 |
| Llama :: Ours-base | 0.609 | 0.533 | **0.568** | 0.688 |

The router **loses to no-routing in 3 of 4 cells** (the read-out best "on average for a domain-type" is
wrong for specific members — e.g. Perplexity is great for medqa but *destroys* adversarial, both
"evidential-type"). Worse, **even CV-selecting the read-out (GLOBAL-best LODO, 0.53–0.60) underperforms
committing to one read-out a priori** (always-Perplexity 0.60–0.63): the datasets are too small and
high-variance for *any* adaptive read-out choice to survive. **This refutes the MoE idea directly** — an
MoE gate is the same learned cross-domain selector, so it would chase the same unreachable ceiling and
regress to the best single expert. Three independent LODO refutations now (stacker, best-variant config
selection §2, read-out router) all say: **fit-across-domains does not transfer; commit to one fixed
choice.** Scripts: `mt_est_variants.sh`/`mt_variant_ppl.py`, `mt_proxy_fair.py`.

Remaining untried lever (a bigger bet, not a quick test): a **multi-head joint proxy** trained to predict
several oracles at once inside one backbone — distinct from the failed *post-hoc* combiners above because
the sharing is end-to-end, though §3 (read-out ≫ objective) tempers the expected payoff.

---

## 7. Honest caveats (on the record)
- **PTrue excluded** — it reads the *target's* logprobs (grey-box), not text. Best pure-BB direct
  method is VerbalizedConfidence (open: 0.616 / 0.707).
- **bioasq is degenerate on Qwen3-32B** (all ≈0.500; direct LexSim hit 1.000 — trivially separable);
  it inverts under some proxies. **Drop it for the Qwen teacher.**
- **"Ours-best" per dataset and the oracle ceiling are mirages** (label-based per-dataset selection);
  they fail LODO (§6c). Deployable numbers use a **single fixed config + single fixed read-out** (~0.61–0.63).
- Read-out control equalizes the proxy families — the durable claims are the **latency/Pareto win** (§5)
  and the **read-out finding** (§3: Perplexity best on the proxy; read-out ≫ objective), **not** "Ours
  beats the baselines" and **not** the domain-router (which itself fails LODO, §6c).

---

## 8. Deployable recommendation

The whole investigation converges on **simplicity** — every attempt to combine methods, configs,
signals, or read-outs (stacker, best-variant, domain/read-out router, MoE) **failed to transfer
out-of-sample** (§6). What deploys is one fixed choice:

> **One proxy, one fixed read-out, no routing / stacking / MoE.**
> - **Proxy:** a single **DALD** proxy (simplest; ties or beats Ours once the read-out is controlled, §3).
> - **Read-out (pick one a priori, don't select):** **Perplexity** for the best *mean* (~0.61–0.63);
>   or a **logit read-out** (Energy/EDL) if adversarial/safety is the priority — Perplexity is
>   structurally blind to confident-wrong answers, and *no router can fix that out-of-sample* (§6c).
> - **Latency:** ~27 ms, target-decoupled (§5) — the durable Pareto win, independent of the AUROC story.
> - **If you can afford a target call:** reserve **VerbalizedConfidence** (direct, target-coupled) for
>   safety-critical Medical/Adversarial queries, where it genuinely dominates (0.61–0.78, §3).

### Ceiling → single-best-variant → deployable (the collapse, honestly)

Three tiers of honesty, mean AUROC on the two best-student cells. **Ceiling** = best variant · best
read-out *per dataset* (label-cheating, fails LODO). **Single-best-variant** = one fixed Ours config
(`head_ecc` · Perplexity) chosen by overall mean — deployable, mildly optimistic. **base · Perplexity**
= one fixed config + one fixed read-out chosen a priori (fully honest). Latency is one forward on the
same student → **identical across objectives** (measured p50: ~43 ms qwen3-4b, ~30 ms llama3.2-3b).

| | Qwen ceiling | Qwen best-var | Qwen base·Ppl | Llama ceiling | Llama best-var | Llama base·Ppl |
|---|--:|--:|--:|--:|--:|--:|
| **Ours** | 0.690 | 0.616 | 0.612 | 0.715 | 0.626 | 0.609 |
| **DALD** | 0.668 | 0.604 | 0.603 | 0.697 | **0.632** | **0.632** |
| **DisAAD** | 0.675 | **0.613** | **0.613** | 0.683 | 0.611 | 0.611 |

At the **ceiling** Ours leads both teachers (+0.02). The moment you commit to a **single fixed config**,
the lead is gone: Ours drops −0.07/−0.09 (the most of the three, because its ceiling leaned hardest on
per-dataset config+read-out cherry-picking), landing in a three-way tie on Qwen (Ours 0.616 ≈ DisAAD
0.613) and *behind* on Llama (**DALD 0.632** > Ours 0.626). base·Perplexity confirms it. Per-dataset
winners stay split (Ours: medqa/medlfqa on Qwen, trivia/gsm8k/truthful on Llama; DALD: the medical QA on
Llama; DisAAD: general/factual on Qwen) — **no proxy owns a domain deployably, and latency is equal
throughout.**

**Do not report the oracle ceiling (0.68–0.70) as a result** — it requires per-dataset label-based
read-out selection that fails LODO (§6c). The honest headline number is the **fixed-read-out ~0.61–0.63**.
The uncertainty-aware objective's one narrow, read-out-invariant edge over both baselines is **Factual
(wikipedia) on both teachers, and Medical-MCQ under logit read-outs**; "owns Medical+Adversarial" does
not survive read-out control. On frontier API teachers, DALD dominates outright (Ours under-tuned there).
