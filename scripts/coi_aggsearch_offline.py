#!/usr/bin/env python
"""OFFLINE aggregation search for CoI-Verbalized -- no GPU, no featdump job.

Reads the two artifacts already on disk:
  (1) correctness labels: the Stage-B sidecar  stageA_<ds>_<gen>_*_seed<seed>.labels.json
      (key 'correct_llm_judge', one entry per item_id, in cache order), and
  (2) CoI features: the n=3 raw log  rawlog_<gen>_<ds>_n3_seed<seed>.jsonl
      (one record per item, in the SAME cache order; each record carries the logged phi 'tv'
      and the full chain JSON 'raw' from which per-claim grounding/risk/contradiction come).

For each (generator, dataset) cell it compares the confidence->correct AUROC of the fixed-prior
phi (v0) against closed-form aggregations AND a cross-validated logistic regression (v1) fit over
the CoI features -- i.e. "is there a mathematical aggregation that better separates correct from
incorrect than the hand-set phi?". CPU-only; run in the ttlm venv.

    ~/JasonLucas/envs/ttlm/bin/python scripts/coi_aggsearch_offline.py
"""
import glob, json, os, statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

WORK = os.path.expanduser("~/JasonLucas")
RC = f"{WORK}/outputs/results_coi"
CACHE = {"trivia_qa": f"{WORK}/outputs/cache_full", "medlfqa": f"{WORK}/outputs/cache_full_health"}
CELLS = [("llama-3.1-8b", "trivia_qa"), ("llama-3.1-8b", "medlfqa"),
         ("qwen3-8b", "trivia_qa"), ("qwen3-8b", "medlfqa")]
SEED = 1

# grounding prior used by phi (matches coi_verbalized BASE); for feature extraction only
BASE = {"contextual": 1.00, "parametric": 0.85, "inferred": 0.65}
LOGIT_FEATS = ["mean_risk", "min_risk", "max_risk", "std_risk", "frac_contextual",
               "frac_parametric", "frac_inferred", "has_contradiction", "answers_query",
               "n_claims", "prod_1mrisk", "mean_base", "min_base_x_1mrisk"]


def load_labels(gen, ds):
    hits = glob.glob(f"{CACHE[ds]}/stageA_{ds}_{gen}_*_seed{SEED}.labels.json")
    if not hits:
        return None
    labs = json.load(open(hits[0]))["labels"]
    # cache order == integer item_id order 0..n-1
    keys = sorted(labs, key=lambda k: int(k))
    return [int(labs[k]["correct_llm_judge"]) for k in keys]


def _primary(grounding):
    """grounding may be pipe-joined e.g. 'contextual|parametric' -> pick highest-prior token."""
    toks = [t.strip() for t in str(grounding).split("|") if t.strip() in BASE]
    return max(toks, key=lambda t: BASE[t]) if toks else "inferred"


def feats_from_raw(rec):
    """Extract per-item aggregate features from one raw-log record."""
    phi_v0 = rec.get("tv")
    risks, grounds, bases, b1mr = [], [], [], []
    contra, answers_query = 0, 1
    try:
        obj = json.loads(rec["raw"]) if isinstance(rec.get("raw"), str) else (rec.get("raw") or {})
    except Exception:
        obj = {}
    crit = (obj.get("chain_2_evidence_critique") or {}).get("claims") or []
    for c in crit:
        r = c.get("risk")
        r = float(r) if isinstance(r, (int, float)) else 0.5
        g = _primary(c.get("grounding", "inferred"))
        risks.append(r); grounds.append(g)
        bases.append(BASE[g]); b1mr.append(BASE[g] * (1 - r))
    cons = obj.get("chain_3_consistency") or {}
    contra = 1 if (cons.get("contradictions") or []) else 0
    aq = cons.get("answers_query")
    answers_query = 1 if (aq is True or aq is None) else 0
    if not risks:  # parse produced no claims -> fall back to logged phi, neutral features
        risks = [0.5]; grounds = ["inferred"]; bases = [BASE["inferred"]]; b1mr = [BASE["inferred"] * 0.5]
    n = len(risks)
    return {
        "phi_v0": phi_v0,
        "mean_risk": statistics.fmean(risks),
        "min_risk": min(risks),
        "max_risk": max(risks),
        "std_risk": statistics.pstdev(risks) if n > 1 else 0.0,
        "frac_contextual": sum(g == "contextual" for g in grounds) / n,
        "frac_parametric": sum(g == "parametric" for g in grounds) / n,
        "frac_inferred": sum(g == "inferred" for g in grounds) / n,
        "has_contradiction": contra,
        "answers_query": answers_query,
        "n_claims": n,
        "prod_1mrisk": float(np.prod([1 - r for r in risks])),
        "mean_base": statistics.fmean(bases),
        "mean_base_x_1mrisk": statistics.fmean(b1mr),     # phi-like mean pool
        "min_base_x_1mrisk": min(b1mr),                   # phi-like min pool
        "noisy_or": 1 - float(np.prod(risks)),            # 1 - prod(risk)
    }


def auroc_pos(y, conf):
    y = np.asarray(y); conf = np.asarray(conf, float); ok = np.isfinite(conf)
    if ok.sum() < 4 or len(set(y[ok].tolist())) < 2:
        return float("nan")
    return roc_auc_score(y[ok], conf[ok])


def cv_logistic(X, y, seed=0):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    n = min(5, int(y.sum()), int((1 - y).sum()))
    if n < 2:
        return float("nan")
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


def main():
    hdr = (f"{'cell':<24}{'n/pos':>9}{'v0_phi':>8}{'1-mean':>8}{'1-max':>8}{'prod':>8}"
           f"{'nOR':>7}{'mean_bxr':>9}{'min_bxr':>9}{'logit_CV':>10}   best")
    print(hdr); print("-" * len(hdr))
    out = {}
    for gen, ds in CELLS:
        y = load_labels(gen, ds)
        rl = f"{RC}/rawlog_{gen}_{ds}_n3_seed{SEED}.jsonl"
        if y is None or not os.path.exists(rl):
            print(f"{gen+'/'+ds:<24}  MISSING (labels={y is not None}, rawlog={os.path.exists(rl)})")
            continue
        recs = [json.loads(l) for l in open(rl) if l.strip()]
        m = min(len(y), len(recs)); y = y[:m]; recs = recs[:m]
        # drop judge-abstain items (correct_llm_judge == -1) -> keep only binary 0/1
        keep = [i for i in range(m) if y[i] in (0, 1)]
        y = [y[i] for i in keep]; recs = [recs[i] for i in keep]; m = len(y)
        F = [feats_from_raw(r) for r in recs]
        col = lambda k: np.array([f[k] for f in F], float)
        cands = {
            "v0_phi": col("phi_v0"),
            "1-mean": 1 - col("mean_risk"),
            "1-max": 1 - col("max_risk"),
            "prod": col("prod_1mrisk"),
            "nOR": 1 - col("noisy_or"),               # = prod(risk); low risk -> high conf
            "mean_bxr": col("mean_base_x_1mrisk"),
            "min_bxr": col("min_base_x_1mrisk"),
        }
        A = {k: auroc_pos(y, v) for k, v in cands.items()}
        X = np.column_stack([col(f) for f in LOGIT_FEATS])
        A["logit_CV"] = cv_logistic(X, np.asarray(y))
        best = max(A, key=lambda k: (A[k] if np.isfinite(A[k]) else -1))
        npos = int(sum(y))
        print(f"{gen+'/'+ds:<24}{f'{m}/{npos}':>9}{A['v0_phi']:>8.3f}{A['1-mean']:>8.3f}"
              f"{A['1-max']:>8.3f}{A['prod']:>8.3f}{A['nOR']:>7.3f}{A['mean_bxr']:>9.3f}"
              f"{A['min_bxr']:>9.3f}{A['logit_CV']:>10.3f}   {best} ({A[best]:.3f})")
        out[f"{gen}/{ds}"] = {"n": m, "pos": npos, "aurocs": {k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in A.items()}, "best": best}
    dst = f"{RC}/aggsearch_offline_seed{SEED}.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
