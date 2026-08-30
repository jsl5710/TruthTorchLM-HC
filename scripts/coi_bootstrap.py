#!/usr/bin/env python
"""Per-cell bootstrap confidence intervals for the CoI results.

Two estimands, both from the SAME resampled items so the comparison is paired:
  (1) AUROC of each rung, with a percentile CI;
  (2) the DELTA (best-verify minus best-baseline), with a CI and a two-sided
      bootstrap p-value for delta<=0 -- this is what decides whether a "win" is real.

Per-item scores come from the rung raw logs (`rawlog_<gen>_<ds>_n<K>_seed<seed>.jsonl`,
field `tv`, one record per item in cache order); labels come from the Stage-B sidecar
(`.labels.json`, key `correct_llm_judge`) or a precomputed labels file for LongFact.

    ~/JasonLucas/envs/ttlm/bin/python scripts/coi_bootstrap.py
"""
import glob, json, os
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

W = os.path.expanduser("~/JasonLucas/outputs")
B = 10000
RNG = np.random.default_rng(0)


def auc_rows(Y, S):
    """Vectorized tie-aware AUROC for each row of (Y, S) via the Mann-Whitney rank identity.
    Ties matter here: n_7 scores take only K+1 distinct values, so rank-averaging is required."""
    R = rankdata(S, axis=1)
    npos = Y.sum(1)
    nneg = Y.shape[1] - npos
    sr = (R * Y).sum(1)
    out = np.full(Y.shape[0], np.nan)
    ok = (npos > 0) & (nneg > 0)
    out[ok] = (sr[ok] - npos[ok] * (npos[ok] + 1) / 2.0) / (npos[ok] * nneg[ok])
    return out

# (generator, dataset, results-dir, labels-source)
CELLS = [
    ("llama-3.1-8b", "longfact", "results_coi_longfact",
     f"{W}/cache_coi_longfact/labels_longfact_llama-3.1-8b_seed1.json"),
    ("qwen3-8b", "longfact", "results_coi_longfact",
     f"{W}/cache_coi_longfact/labels_longfact_qwen3-8b_seed1.json"),
    ("llama-3.1-8b", "trivia_qa", "results_coi_verify", "sidecar:cache_full"),
    ("qwen3-8b", "trivia_qa", "results_coi_verify", "sidecar:cache_full"),
    ("llama-3.1-8b", "medlfqa", "results_coi_verify", "sidecar:cache_full_health"),
    ("qwen3-8b", "medlfqa", "results_coi_verify", "sidecar:cache_full_health"),
]
RUNGS = [1, 3, 6, 7]


def load_labels(src, gen, ds):
    if src.startswith("sidecar:"):
        root = src.split(":", 1)[1]
        hits = glob.glob(f"{W}/{root}/stageA_{ds}_{gen}_*_seed1.labels.json")
        if not hits:
            return None
        labs = json.load(open(hits[0]))["labels"]
        return [int(labs[k]["correct_llm_judge"]) for k in sorted(labs, key=lambda x: int(x))]
    lab = json.load(open(src))
    return [int(v) for _, v in sorted(lab.items(), key=lambda kv: int(kv[0].replace("lf", "")))]


SEARCH = ["results_coi_longfact", "results_coi_verify", "results_coi", "results_coi_extend",
          "results_coi_gen"]


def load_scores(rdir, gen, ds, n):
    """Rung raw logs live in whichever run produced that rung; search the known dirs."""
    for d in [rdir] + [x for x in SEARCH if x != rdir]:
        p = f"{W}/{d}/rawlog_{gen}_{ds}_n{n}_seed1.jsonl"
        if os.path.exists(p):
            v = [json.loads(l)["tv"] for l in open(p) if l.strip()]
            if v:
                return v
    return None


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def main():
    print(f"Paired bootstrap, B={B}, percentile CIs\n")
    hdr = f"{'cell':<26}{'rung':>6}{'AUROC':>8}  {'95% CI':<16}"
    for gen, ds, rdir, lsrc in CELLS:
        y = load_labels(lsrc, gen, ds)
        S = {n: load_scores(rdir, gen, ds, n) for n in RUNGS}
        S = {n: v for n, v in S.items() if v}
        if not y or not S:
            print(f"{gen}/{ds}: missing data"); continue
        m = min([len(y)] + [len(v) for v in S.values()])
        y = y[:m]; S = {n: v[:m] for n, v in S.items()}
        keep = [i for i in range(m) if y[i] in (0, 1)]
        y = np.array([y[i] for i in keep])
        S = {n: np.array([v[i] for i in keep], float) for n, v in S.items()}
        npos, nneg = int(y.sum()), int((1 - y).sum())
        print(f"=== {gen}/{ds}  n={len(y)} pos={npos} neg={nneg} ===")
        # bootstrap indices shared across rungs (paired), vectorized
        idx = RNG.integers(0, len(y), size=(B, len(y)))
        Yb = y[idx]
        boot = {}
        for n, s in S.items():
            vals = auc_rows(Yb, s[idx])
            boot[n] = vals
            # sanity: vectorized AUROC must agree with sklearn on the observed sample
            assert abs(auc_rows(y[None, :], s[None, :])[0] - auroc(y, s)) < 1e-9
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            print(f"  n{n}: AUROC={auroc(y,s):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  width={hi-lo:.3f}")
        # paired delta: best verify (6,7) - best baseline (1,3)
        if all(k in boot for k in RUNGS):
            dv = np.maximum(boot[6], boot[7]) - np.maximum(boot[1], boot[3])
            obs = max(auroc(y, S[6]), auroc(y, S[7])) - max(auroc(y, S[1]), auroc(y, S[3]))
            lo, hi = np.nanpercentile(dv, [2.5, 97.5])
            p = 2 * min(np.nanmean(dv <= 0), np.nanmean(dv >= 0))
            sig = "SIGNIFICANT" if lo > 0 else ("significant (negative)" if hi < 0 else "not significant")
            print(f"  DELTA(verify - baseline) = {obs:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}  -> {sig}")
        print()


if __name__ == "__main__":
    main()
