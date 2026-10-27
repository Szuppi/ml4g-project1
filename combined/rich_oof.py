"""Reproduce the `rich` member's held-out (joint D1) OOF.

rich.py is production-only natively (fits on all X1/X2, predicts X3; no validation split),
so we refit its exact CatBoost config on the joint-D1 training rows and predict the held-out
chr3/chr11 rows. Config and target construction are copied verbatim from reproduction/training/
rich.py (iterations=1388, depth=5, lr=.05, l2=10, RMSE, seed=42, border_count=64, nan_mode=Min;
per-cell rank target). Writes validation.csv in the same format as the other members.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "combined" / "prepared_ensemble"
OUTDIR = ROOT / "combined" / "work" / "ensemble" / "rich" / "results" / "joint"
RESERVED_PLUS_D1 = ["chr1", "chr3", "chr11", "chr12", "chr18"]
D1 = ["chr3", "chr11"]


def rank_target(m, mask):
    """Per-cell rank target over the masked rows (the sweep's targets())."""
    q = m.loc[mask]
    out = np.empty(len(q), np.float32)
    for c in q.cell_line.unique():
        take = q.cell_line.eq(c).to_numpy()
        y = q.loc[take, "gex"].to_numpy()
        s = np.sort(y)
        out[take] = (np.searchsorted(s, y, "left") + np.searchsorted(s, y, "right")) / (2 * len(s))
    return out


def main():
    m = pd.read_csv(PREP / "metadata.csv")
    X = np.load(PREP / "rich.npy").astype(np.float32)
    assert len(m) == len(X) and not np.isinf(X).any()
    source = m.cell_line.isin(["X1", "X2"]).to_numpy()
    train = source & ~m.chr.isin(RESERVED_PLUS_D1).to_numpy()
    evaluate = source & m.chr.isin(D1).to_numpy()
    model = CatBoostRegressor(iterations=1388, depth=5, learning_rate=.05, l2_leaf_reg=10,
                              loss_function="RMSE", random_seed=42, thread_count=2,
                              verbose=200, allow_writing_files=False, border_count=64,
                              nan_mode="Min")
    model.fit(X[train], rank_target(m, train))
    p = model.predict(X[evaluate])
    assert np.isfinite(p).all()
    f = m.loc[evaluate, ["cell_line", "gene_name", "chr", "gex"]].copy()
    f["continuous_prediction"] = p
    OUTDIR.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUTDIR / "validation.csv", index=False)
    print(f"wrote {OUTDIR / 'validation.csv'}  rows={len(f)}")
    for c in ["X1", "X2"]:
        g = f[f.cell_line == c]
        from scipy.stats import spearmanr
        print(f"  {c}: spearman(gex, pred) = {spearmanr(g.gex, g.continuous_prediction).statistic:.4f}")


if __name__ == "__main__":
    main()
