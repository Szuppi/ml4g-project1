"""Cross-product: the frozen 14-model ensemble x the sweep grid, scored on joint-D1.

For every non-empty subset of {cnn, catboost, ridge}, every ordering in {cal, tie, gate},
and every staged weight w, blend the sweep unit into the 14-model ensemble and score joint-D1
Spearman. The ensemble14_only row is the baseline; rows below it are flagged, nothing is
dropped. Output is sorted by joint_mean descending.

Only D1 is available for all 14 ensemble members (9 of the ensemble runners validate on a
single fixed split == D1), so this is a one-fold held-out comparison; D2 is reported where
the sweep models provide it, left blank for the 14-model ensemble.
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from combined import blendlib as B

OOF_SWEEP = ROOT / "combined" / "oof_sweep"
HIS14 = ROOT / "combined" / "oof_ensemble" / "ensemble14_joint_D1.csv"
ORDERINGS = ["cal", "tie", "gate"]
WEIGHTS = [5, 10, 15, 20, 30, 40]
MODELS = {"cnn": "cnn_all_assays-ref.csv",
          "catboost": "catboost_all_assays-ref.csv",
          "ridge": "ridge_all_assays-ref.csv"}


def subsets(items):
    for r in range(1, len(items) + 1):
        yield from itertools.combinations(items, r)


def main():
    ensemble14 = pd.read_csv(HIS14)                       # fold,cell_line,gene_name,chr,gex,prediction
    candidates = {name: pd.read_csv(OOF_SWEEP / f).query("fold == 'D1'").reset_index(drop=True)
            for name, f in MODELS.items()}

    base = B.summarize(ensemble14, "prediction")
    rows = [dict(combo="ensemble14_only", ordering="-", w=0,
                 X1=base["cells"][("D1", "X1")], X2=base["cells"][("D1", "X2")],
                 joint_worst=base["joint_worst"], joint_mean=base["joint_mean"])]
    baseline = base["joint_mean"]

    for combo in subsets(list(MODELS)):
        for ordering in ORDERINGS:
            col = f"pred_{ordering}"
            my_unit = B.equal_rank_blend([candidates[m] for m in combo], [col] * len(combo))
            for w in WEIGHTS:
                final = B.staged_blend(ensemble14, my_unit, w)
                s = B.summarize(final, "prediction")
                rows.append(dict(combo="+".join(combo), ordering=ordering, w=w,
                                 X1=s["cells"][("D1", "X1")], X2=s["cells"][("D1", "X2")],
                                 joint_worst=s["joint_worst"], joint_mean=s["joint_mean"]))

    df = pd.DataFrame(rows)
    df["delta_vs_ensemble14"] = (df["joint_mean"] - baseline).round(4)
    df["below_baseline"] = df["delta_vs_ensemble14"] < 0
    for c in ["X1", "X2", "joint_worst", "joint_mean"]:
        df[c] = df[c].round(4)
    df = df.sort_values("joint_mean", ascending=False).reset_index(drop=True)
    dest = ROOT / "combined" / "results" / "crossproduct.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(df)} rows)  ensemble-14 baseline joint_mean={baseline:.4f}\n")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
