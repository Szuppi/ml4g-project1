"""Build the standalone ridge-hurdle X3 baseline submission.

This is a two-part (hurdle) model in the spirit of Dong et al. (2012): a ridge regressor
R(X) ordered by a CatBoost activity classifier C(X) -- the sweep's ``gate`` post-hoc, which
was ridge's best joint-protocol ordering (ridge_all[assays-ref]|gate, joint_mean = 0.8291).
It is submitted on its own (no blend onto the 14-model ensemble) as a reportable baseline.

Both parts are refit on ALL X1+X2 rows (no chromosome holdout, matching the ensemble's
production regime and build_x3.py), then predict the 1,984 X3 genes. Production parameters
are the medians of the joint-fold selections, exactly as build_x3.py sets GATE_T:

    gate threshold t  = 0.5    (median of D1=0.45, D2=0.55)
    ridge L2 alpha    = 3000   (median of D1=5000, D2=1000)

The submission is 2 * rankdata(gated score), cast to int64, the ensemble's own single-member
convention (reproduction/build_submissions.py: ``member = 2 * rankdata(...)``).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sweep.features import tabular_features
from sweep.models import _ridge_pipeline, _CATBOOST
from sweep.scoring import hard_gate
from common import targets

INPUTS = ROOT / "combined" / "work" / "inputs_x3"
ENSEMBLE14_X3 = ROOT / "combined" / "work" / "ensemble_x3_sub" / "ensemble_14.csv"
OUT = ROOT / "combined" / "submissions"
VERSION = "assays-ref"
GATE_T = 0.5            # median of joint-fold gate selections (D1=0.45, D2=0.55)
ALPHA = 3000.0          # median of joint-fold ridge-penalty selections (D1=5000, D2=1000)


def ridge_raw(X, source, x3, y_source, alpha):
    """Refit the ridge pipeline (median impute + standardise + L2) on all source; predict X3."""
    model = _ridge_pipeline(alpha).fit(X[source], y_source)
    return model.predict(X[x3]).astype(np.float64)


def main():
    m = pd.read_csv(INPUTS / "metadata.csv")
    tree_x = np.load(INPUTS / "tree_features.npy")
    X = tabular_features(tree_x, INPUTS, VERSION)
    source = m.cell_line.isin(["X1", "X2"]).to_numpy()
    x3 = m.cell_line.eq("X3").to_numpy()
    y_source = targets(m, source)

    # activity classifier C(X): same config as the sweep, fit on all source
    clf = CatBoostClassifier(loss_function="Logloss", **_CATBOOST)
    clf.fit(X[source], (m.loc[source, "gex"].to_numpy() > 0).astype(int))
    clf_p = clf.predict_proba(X[x3])[:, 1]

    # alpha stability: the gated ranking is what the submission is; confirm it barely moves
    subs = {}
    for a in (1000.0, 3000.0, 5000.0):
        gate = hard_gate(clf_p, ridge_raw(X, source, x3, y_source, a), GATE_T)
        subs[a] = 2.0 * rankdata(gate, method="average")
    print("alpha-stability (spearman of the 2*rankdata submission vs alpha=3000):")
    for a in (1000.0, 5000.0):
        print(f"  alpha={a:>6.0f} vs 3000: rho = {spearmanr(subs[a], subs[3000.0]).statistic:.6f}")
    print(f"  classifier hurdle keeps {(clf_p >= GATE_T).sum()} of {len(clf_p)} genes (rest zeroed)")

    score = subs[ALPHA]

    # align to the ensemble's gene order and emit
    ensemble = pd.read_csv(ENSEMBLE14_X3)
    genes = ensemble.gene_name.tolist()
    candidates = pd.DataFrame({"gene_name": m.loc[x3, "gene_name"].to_numpy(), "score": score}).set_index("gene_name")
    assert set(genes) == set(candidates.index) and len(genes) == 1984, "X3 gene set mismatch"
    score = candidates.loc[genes, "score"].to_numpy()

    assert len(score) == 1984 and np.isfinite(score).all() and np.ptp(score) > 0
    OUT.mkdir(parents=True, exist_ok=True)
    name = "baseline-ridge-hurdle"
    df = pd.DataFrame({"gene_name": genes, "gex_predicted": score.astype(np.int64)})
    assert df.gene_name.tolist() == genes and not df.gene_name.duplicated().any()
    df.to_csv(OUT / f"{name}.csv", index=False)
    rho_ensemble = spearmanr(df.gex_predicted, ensemble.gex_predicted).statistic
    print(f"\nwrote {OUT / (name + '.csv')}  genes={len(df)}  "
          f"rank_range=[{df.gex_predicted.min()},{df.gex_predicted.max()}]  "
          f"unique={df.gex_predicted.nunique()}  spearman_vs_ensemble14={rho_ensemble:.4f}")


if __name__ == "__main__":
    main()
