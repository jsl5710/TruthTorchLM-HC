#!/usr/bin/env python
"""Master CoI results table: AUROC for rungs n=1/3/6/7 across ALL datasets x both 8B targets,
consolidating every CoI results dir. Datasets grouped by (format, domain) to expose where the
verify rungs win. Reads results_coi_extend (7 new cached datasets), results_coi_verify
(trivia/medlfqa n1/6/7), results_coi_longfact (LongFact), results_coi (n3 baselines)."""
import json, os

W = os.path.expanduser("~/JasonLucas/outputs")
# search priority per dataset (first dir that has the rung wins)
DIRS = ["results_coi_extend", "results_coi_verify", "results_coi_longfact",
        "results_coi_bigtargets", "results_coi_bigtargets_health", "results_coi_gen",
        "results_coi_closed", "results_coi"]
GENS = [("llama-3.1-8b", "Llama-8B"), ("qwen3-8b", "Qwen-8B"),
        ("qwen3-32b", "Qwen-32B"), ("llama3.3-70b", "Llama-70B"),
        ("jhu-gpt-4o", "GPT-4o"), ("jhu-claude-haiku-4.5", "Claude-H4.5")]
SEEDS = [1, 0]  # 8B cells are seed1; big-target/closed cells are seed0
# (dataset key, label, group)
DSETS = [
    ("trivia_qa", "TriviaQA", "general/short"),
    ("natural_qa", "NaturalQA", "general/short"),
    ("pop_qa", "PopQA", "general/short"),
    ("longfact", "LongFact", "general/long"),
    ("truthful_qa", "TruthfulQA", "adversarial"),
    ("medqa", "MedQA", "health/MCQ"),
    ("mmlu_med", "MMLU-Med", "health/MCQ"),
    ("bioasq", "BioASQ", "health/free"),
    ("kqa", "K-QA", "health/free"),
    ("medlfqa", "MedLFQA", "health/long"),
    ("gsm8k", "GSM8K", "math/multi-hop"),
    ("hotpot_qa", "HotpotQA", "general/multi-hop"),
]
RUNGS = [1, 3, 6, 7]


def get(gen, ds, n):
    for d in DIRS:
        for seed in SEEDS:
            p = f"{W}/{d}/coi_{gen}_seed{seed}.json"
            if not os.path.exists(p):
                continue
            for c in json.load(open(p)).get("cells", []):
                if c.get("dataset") == ds:
                    r = (c.get("rows") or {}).get(f"CoIVerbalized_n{n}")
                    if r and r.get("auroc") is not None:
                        return r["auroc"], c.get("n_items"), c.get("n_positive")
    return None, None, None


def main():
    for gen, gl in GENS:
        print(f"\n===== {gl} ({gen}) -- AUROC =====")
        hdr = f"{'dataset':<12}{'group':<15}{'n':>5}{'n1_verb':>9}{'n3_phi':>8}{'n6_YN':>8}{'n7_selfC':>9}   best"
        print(hdr); print("-" * len(hdr))
        for ds, dl, grp in DSETS:
            au = {n: get(gen, ds, n)[0] for n in RUNGS}
            _, nit, npos = get(gen, ds, 1)
            if all(v is None for v in au.values()):
                print(f"{dl:<12}{grp:<15}{'--':>5}  (no results yet)"); continue
            # degenerate cell: minority class < 15 -> AUROC unreliable, don't rank
            if nit and npos is not None and min(npos, nit - npos) < 15:
                cells = "".join((f"{au[n]:>9.3f}" if n == 1 else f"{au[n]:>8.3f}") if au[n] is not None
                                else (f"{'--':>9}" if n == 1 else f"{'--':>8}") for n in RUNGS)
                print(f"{dl:<12}{grp:<15}{nit:>5}{cells}   DEGEN (minority={min(npos, nit-npos)})"); continue
            best = max((n for n in au if au[n] is not None), key=lambda n: au[n])
            cells = "".join((f"{au[n]:>9.3f}" if n == 1 else f"{au[n]:>8.3f}") if au[n] is not None
                            else (f"{'--':>9}" if n == 1 else f"{'--':>8}") for n in RUNGS)
            nlabel = f"{nit}" if nit else "--"
            print(f"{dl:<12}{grp:<15}{nlabel:>5}{cells}   n{best} ({au[best]:.3f})")
    # verify-family win-rate summary
    print("\n=== where does best-verify (n6/n7) beat best-baseline (n1/n3)? ===")
    for gen, gl in GENS:
        wins = []
        for ds, dl, grp in DSETS:
            au = {n: get(gen, ds, n)[0] for n in RUNGS}
            _, nit, npos = get(gen, ds, 1)
            if any(au[n] is None for n in (1, 3, 6, 7)):
                continue
            if nit and npos is not None and min(npos, nit - npos) < 15:
                continue  # skip degenerate cells
            vb, bb = max(au[6], au[7]), max(au[1], au[3])
            wins.append((dl, vb - bb))
        won = [f"{d}(+{x:.3f})" for d, x in wins if x > 0]
        lost = [f"{d}({x:.3f})" for d, x in wins if x <= 0]
        print(f"  {gl}: WIN {len(won)}/{len(wins)} -> {', '.join(won) or 'none'}")
        if lost:
            print(f"       lose: {', '.join(lost)}")


if __name__ == "__main__":
    main()
