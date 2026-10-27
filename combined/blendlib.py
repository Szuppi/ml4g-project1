"""Scoring and rank-blend primitives for combining the 14-model ensemble with the sweep models.

Everything operates per (fold, cell_line) group, because that is the unit the joint
protocol is scored on (one Spearman per cell per fold). Ranking is the blend currency:
members are average-tie ranked within each group before being combined, exactly as in
the walkthrough's equal_rank_blend and in the ensemble's staged recipe.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

KEYS = ["fold", "cell_line", "gene_name", "chr"]
GROUP = ["fold", "cell_line"]


def tie_rank(x):
    """2 * average-tie rank, the shared ranking map of the ensemble recipe and the walkthrough."""
    return 2.0 * rankdata(np.asarray(x, dtype=float), method="average")


def cell_scores(df, pred_col, gex_col="gex"):
    """Per (fold, cell_line) Spearman of pred vs gex."""
    out = {}
    for (fold, cell), g in df.groupby(GROUP):
        y, p = g[gex_col].to_numpy(), g[pred_col].to_numpy()
        out[(fold, cell)] = float("nan") if np.ptp(p) == 0 else float(spearmanr(y, p).statistic)
    return out


def summarize(df, pred_col, gex_col="gex"):
    """joint_mean (mean over fold x cell groups), joint_worst (min), and per-fold means."""
    s = cell_scores(df, pred_col, gex_col)
    vals = list(s.values())
    per_fold = {}
    for fold in sorted({f for f, _ in s}):
        per_fold[fold] = float(np.mean([v for (f, _c), v in s.items() if f == fold]))
    return dict(joint_mean=float(np.mean(vals)), joint_worst=float(np.min(vals)),
                per_fold=per_fold, cells=s)


def equal_rank_blend(frames, pred_cols):
    """Sum of 2*tie-rank across components, within each (fold, cell_line) group.

    frames: list of DataFrames each carrying KEYS + its prediction column.
    Returns KEYS + gex + 'prediction' (the first frame must carry gex)."""
    base = frames[0][KEYS + ["gex"]].copy().reset_index(drop=True)
    acc = np.zeros(len(base), dtype=float)
    for frame, col in zip(frames, pred_cols):
        merged = base.merge(frame[KEYS + [col]], on=KEYS, validate="one_to_one")
        if len(merged) != len(base):
            raise ValueError("component does not cover the same rows")
        for _, idx in merged.groupby(GROUP).groups.items():
            acc[idx] = acc[idx] + np.rint(tie_rank(merged.loc[idx, col])).astype(np.int64)
    base["prediction"] = acc
    return base


def staged_blend(base_df, member_df, w, base_col="prediction", member_col="prediction"):
    """final = (100-w)*rank(base) + w*rank(member), within each (fold, cell_line) group.

    w in [0, 100]. Mirrors the ensemble's build_submission staged step, with the running
    blend as the base and the added model as one member."""
    base = base_df[KEYS + ["gex"]].copy().reset_index(drop=True)
    merged = base.merge(base_df[KEYS + [base_col]].rename(columns={base_col: "_base"}), on=KEYS)
    merged = merged.merge(member_df[KEYS + [member_col]].rename(columns={member_col: "_member"}), on=KEYS)
    out = np.zeros(len(base), dtype=float)
    for _, idx in merged.groupby(GROUP).groups.items():
        out[idx] = (100 - w) * tie_rank(merged.loc[idx, "_base"]) + w * tie_rank(merged.loc[idx, "_member"])
    base["prediction"] = out
    return base
