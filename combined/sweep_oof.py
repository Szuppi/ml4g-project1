"""Reconstruct the 3 sweep models' joint out-of-fold predictions from the cached sweep.

For each of cnn/catboost/ridge (all genes, assays-ref features) under the joint protocol
(train both cells minus held-out chromosomes, evaluate those chromosomes in both cells),
load the cached raw predictions and re-apply the five post-hoc orderings (raw/tie/cal/gate/soft)
with their validation-selected parameters. Writes one keyed OOF table per model:

    combined/oof_sweep/<kind>_all_assays-ref.csv
    columns: fold, cell_line, gene_name, chr, gex, raw, pred_raw, pred_tie, pred_cal, pred_gate, pred_soft

These are the exact rows the walkthrough's equal_rank_blend harness merges on
(fold, cell_line, gene_name, chr), so they drop straight into the cross-product with the 14-model ensemble.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import masks, load_inputs
from sweep.scoring import score_families

INPUTS = ROOT / "inputs"
SWEEP = ROOT / "train_interactive" / "outputs" / "sweep"
OUT = ROOT / "combined" / "oof_sweep"
PROTOCOL = "joint"
FOLDS_USED = ("D1", "D2")
CONFIGS = [("cnn", "all", "assays-ref"),
           ("catboost", "all", "assays-ref"),
           ("ridge", "all", "assays-ref")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, _ = load_inputs(INPUTS)
    for kind, mode, version in CONFIGS:
        rows = []
        for fold in FOLDS_USED:
            train, validation, evaluate = masks(m, PROTOCOL, fold)
            key = f"{kind}_{mode}_{version}_{PROTOCOL}_{fold}"
            z = np.load(SWEEP / "preds" / f"{key}.npz")
            reg = dict(train=train, validation=validation, evaluate=evaluate,
                       raw_val=z["raw_val"], raw=z["raw"])
            cz = np.load(SWEEP / "clf_preds" / f"{version}_{PROTOCOL}_{fold}.npz")
            clf = dict(p_val=cz["p_val"], p_eval=cz["p_eval"])
            families = score_families(m, reg, clf)
            part = m.loc[evaluate, ["cell_line", "gene_name", "chr", "gex"]].copy()
            part.insert(0, "fold", fold)
            part["raw"] = np.asarray(z["raw"], float)
            for recipe, (scored, _param) in families.items():
                part[f"pred_{recipe}"] = np.asarray(scored, float)
            rows.append(part)
        out = pd.concat(rows, ignore_index=True)
        dest = OUT / f"{kind}_{mode}_{version}.csv"
        out.to_csv(dest, index=False)
        print(f"wrote {dest.relative_to(ROOT)}  rows={len(out)}  cols={list(out.columns)}")


if __name__ == "__main__":
    main()
