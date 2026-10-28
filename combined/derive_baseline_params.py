"""Derive the two validation-selected parameters baked into build_x3_ridge.py.

The ridge-hurdle baseline is refit on all source rows, so it has no held-out fold to select
its gate threshold or L2 penalty on. Both are therefore fixed to the median of the two
cross-validation folds' selections. This script recomputes those fold selections and prints
the medians, confirming GATE_T = 0.5 and ALPHA = 3000.

Reads the cached sweep (train_interactive/outputs/sweep) and the original inputs; run from the
repository root.
"""
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import masks, load_inputs
from sweep.features import tabular_features
from sweep.models import ridge_fold
from sweep.scoring import score_families

INPUTS = ROOT / "inputs"
SWEEP = ROOT / "train_interactive" / "outputs" / "sweep"
FOLDS = ("D1", "D2")


def main():
    m, _ = load_inputs(INPUTS)
    X = tabular_features(np.load(INPUTS / "tree_features.npy"), INPUTS, "assays-ref")

    gates, alphas = {}, {}
    for fold in FOLDS:
        train, validation, evaluate = masks(m, "joint", fold)

        # gate threshold: selected by score_families on cached raw + classifier predictions
        z = np.load(SWEEP / "preds" / f"ridge_all_assays-ref_joint_{fold}.npz")
        c = np.load(SWEEP / "clf_preds" / f"assays-ref_joint_{fold}.npz")
        reg = dict(train=train, validation=validation, evaluate=evaluate,
                   raw_val=z["raw_val"], raw=z["raw"])
        families = score_families(m, reg, dict(p_val=c["p_val"], p_eval=c["p_eval"]))
        gates[fold] = families["gate"][1]

        # L2 penalty: selected inside ridge_fold on the source-cell validation rows
        alphas[fold] = ridge_fold(m, X, "joint", fold, active=False)["model"].named_steps["ridge"].alpha

    print(f"gate threshold  D1={gates['D1']}  D2={gates['D2']}  -> median {median(gates.values())}")
    print(f"ridge L2 alpha  D1={alphas['D1']}  D2={alphas['D2']}  -> median {median(alphas.values())}")


if __name__ == "__main__":
    main()
