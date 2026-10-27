"""Replay the exact 14-model ensemble recipe on the reproduced joint-D1 OOF -> the ensemble-14 unit.

The ensemble blend is defined for a single cell line's gene vector (X3). The joint-D1 OOF has
two cells, scored separately, so we apply the recipe per cell: assemble the 14 members' continuous
predictions for that cell's D1 genes, call the ensemble_11 blend() verbatim, then apply the
three ensemble_14 average-rank steps. Local eval is same-distribution as training, so the
X3-specific quantile/original/absolutehalf normalizations are identities here and we feed the
raw continuous predictions; the zero-floor cutoffs use the D1-train zero rate as source_zero,
the local analog of the fixed 0.538.

Output: combined/oof_ensemble/ensemble14_joint_D1.csv  [fold, cell_line, gene_name, chr, gex, prediction]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE = ROOT / "combined" / "work" / "ensemble"
PREP = ROOT / "combined" / "prepared_ensemble"
sys.path.insert(0, str(ROOT / "ensemble_11_rank_blend"))
from build_submission import blend  # the ensemble's verbatim ensemble_11 blend()

KEYS = ["cell_line", "gene_name", "chr"]
D1 = ["chr3", "chr11"]
RESERVED_PLUS_D1 = ["chr1", "chr3", "chr11", "chr12", "chr18"]

# member -> (validation csv, value columns). Heads carry two columns (active logit, positive).
FILES = {
    "cnn": ("cnn/results/validation/D1/swa/validation_predictions.csv", ["continuous_prediction"]),
    "tree": ("tree/results/validation/D1/upgrade/validation_predictions.csv", ["continuous_prediction"]),
    "weighted": ("weighted/results/weight1/development/validation.csv", ["continuous_prediction"]),
    "dilated": ("variants/results/dilated/development/validation.csv", ["continuous_prediction"]),
    "gain": ("variants/results/gain/development/validation.csv", ["continuous_prediction"]),
    "rich": ("rich/results/joint/validation.csv", ["continuous_prediction"]),
    "seed43": ("seed43/results/combo/joint/validation.csv", ["continuous_prediction"]),
    "seed44": ("seed44/results/combo/joint/validation.csv", ["continuous_prediction"]),
    "extra": ("extra/results/joint/validation.csv", ["continuous_prediction"]),
    "basic": ("basic/results/twohead_basic/joint/validation.csv",
              ["active_logit", "positive_rank_prediction"]),
    "combo": ("combo/results/twohead_combo/joint/validation.csv",
              ["active_logit", "positive_rank_prediction"]),
    "wide": ("wide/results/wide/joint/validation.csv", ["continuous_prediction"]),
    "wide_combo": ("wide_combo/results/wide_combo/joint/validation.csv", ["continuous_prediction"]),
    "tree_log": ("tree_log/results/joint/validation.csv", ["continuous_prediction"]),
}


def rank(v):
    return 2 * rankdata(np.asarray(v, float), method="average")


def load_member(name):
    path, cols = FILES[name]
    df = pd.read_csv(ENSEMBLE / path)
    keep = df.chr.isin(D1)
    return df.loc[keep, KEYS + ["gex"] + cols].reset_index(drop=True)


def main():
    members = {name: load_member(name) for name in FILES}
    # Common D1 genes per cell, in one aligned frame (merge on keys).
    base = members["cnn"][KEYS + ["gex"]].copy()
    for name, df in members.items():
        cols = FILES[name][1]
        ren = {c: f"{name}__{c}" for c in cols}
        base = base.merge(df[KEYS + cols].rename(columns=ren), on=KEYS, validate="one_to_one")
    # D1-train zero rate (both cells, chromosomes the D1 models trained on) = local source_zero.
    m = pd.read_csv(PREP / "metadata.csv")
    train = m.cell_line.isin(["X1", "X2"]) & ~m.chr.isin(RESERVED_PLUS_D1)
    source_zero = float((m.loc[train, "gex"] == 0).mean())
    recipe = json.loads((ROOT / "ensemble_11_rank_blend/recipe.json").read_text())
    steps = json.loads((ROOT / "ensemble_14_wide_context/recipe.json").read_text())["steps"]
    cbase = recipe["base"]

    out = []
    for cell in sorted(base.cell_line.unique()):
        g = base[base.cell_line == cell].reset_index(drop=True)
        col = lambda n, c="continuous_prediction": g[f"{n}__{c}"].to_numpy()
        components = {n: col(n) for n in ("cnn", "tree", "weighted", "dilated", "gain",
                                          "rich", "seed43", "seed44", "extra")}
        for n in ("seed43", "seed44", "extra", "basic", "combo"):
            components[f"{n}_zero"] = source_zero
        heads = {}
        for n in ("basic", "combo"):
            heads[f"{n}_production_{recipe['heads']['normalization']}"] = np.column_stack(
                [col(n, "active_logit"), col(n, "positive_rank_prediction")])
        score = blend(components, heads, recipe, source_zero)  # ensemble_11 (verbatim)
        # ensemble_14: wide_combo, tree_log(log target), wide
        targets = {"wide_combo": "rank", "tree_log": "log", "wide": "rank"}
        for step in steps:
            name = step["model"]
            raw = col(name)
            cutoff = step["cut"] if targets[name] == "log" else step["cut"] * source_zero
            member = 2 * rankdata(np.where(raw <= cutoff, 0, raw))
            w = step["weight"]
            score = (100 - w) * 2 * rankdata(score) + w * member
        frame = g[KEYS + ["gex"]].copy()
        frame["prediction"] = score
        out.append(frame)

    ensemble14 = pd.concat(out, ignore_index=True)
    ensemble14.insert(0, "fold", "D1")
    dest = ROOT / "combined" / "oof_ensemble" / "ensemble14_joint_D1.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ensemble14.to_csv(dest, index=False)
    print(f"wrote {dest}  rows={len(ensemble14)}  source_zero={source_zero:.4f}")
    cells = {}
    for cell in sorted(ensemble14.cell_line.unique()):
        gg = ensemble14[ensemble14.cell_line == cell]
        cells[cell] = float(spearmanr(gg.gex, gg.prediction).statistic)
        print(f"  ensemble-14 D1 {cell}: spearman = {cells[cell]:.4f}")
    print(f"  ensemble-14 D1 mean = {np.mean(list(cells.values())):.4f}")


if __name__ == "__main__":
    main()
