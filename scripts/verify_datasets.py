#!/usr/bin/env python
"""Verify one benchmark dataset loads and exposes the expected columns.

Called once per dataset (so a per-dataset `timeout` can cap a runaway download):
    python scripts/verify_datasets.py <dataset_name> [size_of_data]

Prints one pipe-delimited status line:
    OK|<name>|n=..|gt_ok=..|q=..|gt=..     -- loaded, columns present
    WARN|<name>|..                         -- loaded but empty / bad columns
    FAIL|<name>|<error>                    -- loader raised (e.g. datasets 5.0 drift)

The benchmark contract (hc_datasets.py) is that every item is
    {"context": str, "question": str, "ground_truths": list[str]}
so we check `question` and a non-empty list[str] `ground_truths`.
"""

import sys

sys.path.insert(0, "src")


def main():
    name = sys.argv[1]
    size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005

    from TruthTorchLM.utils.dataset_utils import get_dataset

    try:
        ds = get_dataset(name, size_of_data=size, seed=0)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        # A too-small fraction can make train_test_split fail on a tiny dataset; retry whole.
        if "train" in msg and "size" in msg:
            try:
                ds = get_dataset(name, size_of_data=1.0, seed=0)
            except Exception as exc2:  # noqa: BLE001
                print(f"FAIL|{name}|{type(exc2).__name__}: {str(exc2)[:180]}")
                return
        else:
            print(f"FAIL|{name}|{type(exc).__name__}: {msg[:180]}")
            return

    n = len(ds)
    if n == 0:
        print(f"WARN|{name}|n=0|empty dataset")
        return
    ex = ds[0]
    keys = sorted(ex.keys())
    gt = ex.get("ground_truths")
    q = str(ex.get("question", ""))[:55]
    gt_ok = isinstance(gt, list) and len(gt) > 0 and all(isinstance(x, str) for x in gt)
    tag = "OK" if ("question" in ex and gt_ok) else "WARN"
    print(f"{tag}|{name}|n={n}|keys={keys}|gt_ok={gt_ok}|q={q!r}|gt={str(gt)[:70]!r}")


if __name__ == "__main__":
    main()
