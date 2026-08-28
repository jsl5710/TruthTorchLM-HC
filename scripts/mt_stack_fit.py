#!/usr/bin/env python
"""5-signal score-level STACKER on one best-student cell (leave-one-dataset-out).

Consumes the per-item dumps written by scripts/mt_stack_dump.slurm:
    items_ours_seed{S}.json     -> ours_u                 (Ours EDL-AU)
    items_dald_seed{S}.json     -> dald_au, dald_eu       (LogTokU on the DALD proxy)
    items_disaad_seed{S}.json   -> disaad_au, disaad_eu   (LogTokU on the DisAAD proxy)

Merges by (dataset, item_id), then for each dataset trains a logistic-regression stacker on the
OTHER six datasets' items (features = the 5 read-outs, standardized) and predicts P(incorrect) on
the held-out dataset -> per-dataset AUROC. Reports the LODO stacker vs. every single read-out, the
hard domain-router, and the per-dataset oracle ceiling (mean of the per-dataset best single signal).

    python scripts/mt_stack_fit.py --dump-dir ~/JasonLucas/outputs/mt_stack/qwen3-32b_qwen3-4b --seed 0
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURES = ["ours_u", "dald_au", "dald_eu", "disaad_au", "disaad_eu"]
# Domain router (documented): Ours owns Medical + Adversarial; DALD-au owns the rest.
OURS_DOMAINS = {"bioasq", "medqa", "medlfqa", "truthful_qa"}
ROUTER = {  # dataset -> feature used by the hard router
    "bioasq": "ours_u", "medqa": "ours_u", "medlfqa": "ours_u", "truthful_qa": "ours_u",
    "trivia_qa": "dald_au", "gsm8k": "dald_au", "wikipedia_factual": "dald_au",
}


def _key(r):
    return (r["dataset"], str(r["item_id"]))


def load_merged(dump_dir, seed):
    def _load(name):
        p = os.path.join(dump_dir, f"items_{name}_seed{seed}.json")
        if not os.path.exists(p):
            hits = glob.glob(os.path.join(dump_dir, f"items_{name}_seed*.json"))
            if not hits:
                raise SystemExit(f"missing dump: {p}")
            p = hits[0]
        return json.load(open(p))

    merged = {}
    for name in ("ours", "dald", "disaad"):
        for r in _load(name):
            k = _key(r)
            m = merged.setdefault(k, {"dataset": r["dataset"], "item_id": r["item_id"],
                                      "correct": r["correct"]})
            for f in FEATURES:
                if f in r and r[f] is not None:
                    m[f] = float(r[f])
    # keep only items that have all 5 features and a 0/1 label
    rows = [m for m in merged.values()
            if all(f in m for f in FEATURES) and m["correct"] in (0, 1)]
    return rows


def orient(rows):
    """Fixed global sign per feature so higher = more likely INCORRECT (for the single/oracle/router
    baselines). The LR stacker learns its own signs, so orientation only affects the baselines."""
    inc = np.array([1 - r["correct"] for r in rows])
    signs = {}
    for f in FEATURES:
        x = np.array([r[f] for r in rows])
        try:
            a = roc_auc_score(inc, x)
        except ValueError:
            a = 0.5
        signs[f] = 1.0 if a >= 0.5 else -1.0
    return signs


def per_dataset_auroc(rows, values):
    """AUROC of `values` (higher=incorrect) vs incorrectness, per dataset; skip degenerate."""
    out = {}
    by_ds = {}
    for r, v in zip(rows, values):
        by_ds.setdefault(r["dataset"], []).append((1 - r["correct"], v))
    for ds, pairs in by_ds.items():
        y = np.array([p[0] for p in pairs]); v = np.array([p[1] for p in pairs])
        if len(set(y.tolist())) < 2:
            out[ds] = None
        else:
            out[ds] = float(roc_auc_score(y, v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="optional JSON report path")
    ap.add_argument("--C", type=float, default=1.0, help="LR inverse-regularization")
    args = ap.parse_args()

    rows = load_merged(args.dump_dir, args.seed)
    datasets = sorted({r["dataset"] for r in rows})
    print(f"[stack] {len(rows)} items over {len(datasets)} datasets: {datasets}")

    signs = orient(rows)
    X = np.array([[r[f] for f in FEATURES] for rows_ in [rows] for r in rows_])
    Xo = X * np.array([signs[f] for f in FEATURES])          # oriented (higher=incorrect)
    inc = np.array([1 - r["correct"] for r in rows])
    ds_of = np.array([r["dataset"] for r in rows])

    # --- single read-outs (oriented), per dataset ---
    single = {f: per_dataset_auroc(rows, Xo[:, i]) for i, f in enumerate(FEATURES)}

    # --- oracle ceiling: per dataset, best single of the 5 ---
    oracle = {}
    for ds in datasets:
        vals = [single[f][ds] for f in FEATURES if single[f][ds] is not None]
        oracle[ds] = max(vals) if vals else None

    # --- hard domain router ---
    router = {}
    for ds in datasets:
        f = ROUTER.get(ds, "dald_au")
        router[ds] = single[f][ds]

    # --- LODO logistic-regression stacker ---
    stack = {}
    for ds in datasets:
        tr = ds_of != ds; te = ds_of == ds
        if len(set(inc[te].tolist())) < 2 or tr.sum() < 10:
            stack[ds] = None; continue
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, C=args.C)
        lr.fit(sc.transform(X[tr]), inc[tr])          # target = incorrect
        p = lr.predict_proba(sc.transform(X[te]))[:, 1]
        stack[ds] = float(roc_auc_score(inc[te], p))

    def _mean(d, drop=()):
        vals = [v for k, v in d.items() if v is not None and k not in drop]
        return float(np.mean(vals)) if vals else float("nan")

    # ---- report ----
    order = [d for d in ["trivia_qa", "bioasq", "medqa", "medlfqa", "gsm8k",
                         "truthful_qa", "wikipedia_factual"] if d in datasets]
    hdr = f"{'dataset':<18} " + " ".join(f"{f.replace('_',''):>9}" for f in FEATURES) + \
          f" {'router':>7} {'oracle':>7} {'STACK':>7}"
    print("\n" + hdr); print("-" * len(hdr))
    for ds in order:
        dom = "M/A" if ds in OURS_DOMAINS else "rest"
        cells = " ".join(f"{(single[f][ds] if single[f][ds] is not None else float('nan')):9.3f}"
                         for f in FEATURES)
        print(f"{ds+' ['+dom+']':<18} {cells} "
              f"{(router[ds] or float('nan')):7.3f} {(oracle[ds] or float('nan')):7.3f} "
              f"{(stack[ds] or float('nan')):7.3f}")
    print("-" * len(hdr))
    means = {f: _mean(single[f]) for f in FEATURES}
    best_single = max(means, key=means.get)
    row = " ".join(f"{means[f]:9.3f}" for f in FEATURES)
    print(f"{'MEAN (7 ds)':<18} {row} {_mean(router):7.3f} {_mean(oracle):7.3f} {_mean(stack):7.3f}")
    # Qwen: bioasq is degenerate/near-perfectly-invertible -> also report dropping it.
    if "bioasq" in datasets:
        d = ("bioasq",)
        rowx = " ".join(f"{_mean(single[f], d):9.3f}" for f in FEATURES)
        print(f"{'MEAN (-bioasq)':<18} {rowx} {_mean(router, d):7.3f} "
              f"{_mean(oracle, d):7.3f} {_mean(stack, d):7.3f}")

    print(f"\nbest single read-out : {best_single} = {means[best_single]:.3f}")
    print(f"hard domain-router   : {_mean(router):.3f}")
    print(f"LODO stacker (5-sig) : {_mean(stack):.3f}   (oracle ceiling {_mean(oracle):.3f})")
    lift = _mean(stack) - _mean(router)
    print(f"stacker - router     : {lift:+.3f}")

    if args.out:
        json.dump({"dump_dir": args.dump_dir, "seed": args.seed, "n_items": len(rows),
                   "datasets": datasets, "signs": signs,
                   "single": single, "router": router, "oracle": oracle, "stack": stack,
                   "means": {**means, "router": _mean(router), "oracle": _mean(oracle),
                             "stack": _mean(stack)}},
                  open(args.out, "w"), indent=2)
        print(f"[stack] wrote {args.out}")


if __name__ == "__main__":
    main()
