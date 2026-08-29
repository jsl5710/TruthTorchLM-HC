#!/usr/bin/env python
"""Aggregation search: given the dumped CoI features (scripts/coi_featdump.py), find the math that best
correlates confidence with correctness. Compares the fixed-prior phi (v0) to closed-form aggregations
AND a cross-validated logistic regression (the MD's v1) over the CoI features. Reports AUROC per cell."""
import glob, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

WORK=os.path.expanduser("~/JasonLucas")
LOGIT_FEATS=["mean_risk","min_risk","max_risk","std_risk","frac_contextual","frac_parametric",
             "frac_inferred","has_contradiction","answers_query","n_claims","prod_1mrisk"]

def auroc(y, conf):
    y=np.asarray(y); conf=np.asarray(conf,float); ok=np.isfinite(conf)
    if ok.sum()<4 or len(set(y[ok]))<2: return float("nan")
    return roc_auc_score(1-y[ok], conf[ok]*-1 + 0)  # conf high => correct; predict incorrect=1-y vs -conf
def auroc_pos(y, conf):  # AUROC of confidence predicting CORRECT (higher conf -> correct)
    y=np.asarray(y); conf=np.asarray(conf,float); ok=np.isfinite(conf)
    if ok.sum()<4 or len(set(y[ok]))<2: return float("nan")
    return roc_auc_score(y[ok], conf[ok])

def cv_logistic(X, y, seed=0):
    y=np.asarray(y); 
    if len(set(y))<2: return float("nan")
    n=min(5,int(y.sum()),int((1-y).sum()))
    if n<2: return float("nan")
    oof=np.full(len(y),np.nan); skf=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
    for tr,te in skf.split(X,y):
        sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=2000,C=1.0).fit(sc.transform(X[tr]),y[tr])
        oof[te]=lr.predict_proba(sc.transform(X[te]))[:,1]   # P(correct)
    return roc_auc_score(y, oof)

CELLS=[("llama-3.1-8b","trivia_qa"),("llama-3.1-8b","medlfqa"),("qwen3-8b","trivia_qa"),("qwen3-8b","medlfqa")]
print(f"{'cell':<26}{'v0 phi':>8}{'1-mean':>8}{'1-max':>8}{'prod':>8}{'logistic-CV':>13}   best")
for gen,ds in CELLS:
    p=f"{WORK}/outputs/results_coi/features_{gen}_{ds}_seed1.json"
    if not os.path.exists(p): print(f"{gen}/{ds}: no features yet"); continue
    D=json.load(open(p)); y=np.array([d["correct"] for d in D])
    def col(k): return np.array([d[k] for d in D],float)
    cands={"v0 phi":col("phi_v0"), "1-mean":1-col("mean_risk"), "1-max":1-col("max_risk"),
           "prod":col("prod_1mrisk")}
    aurocs={k:auroc_pos(y,v) for k,v in cands.items()}
    X=np.column_stack([col(f) for f in LOGIT_FEATS]); aurocs["logistic-CV"]=cv_logistic(X,y)
    best=max(aurocs,key=lambda k: aurocs[k] if not np.isnan(aurocs[k]) else -1)
    print(f"{gen+'/'+ds:<26}{aurocs['v0 phi']:>8.3f}{aurocs['1-mean']:>8.3f}{aurocs['1-max']:>8.3f}"
          f"{aurocs['prod']:>8.3f}{aurocs['logistic-CV']:>13.3f}   {best} ({aurocs[best]:.3f})")
