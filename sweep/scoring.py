"""Post-hoc orderings that finish a regressor's ranking, selected on validation rows.

``p`` is the activity probability, ``r`` the raw regressor output. Every parameter is chosen
on the source-cell validation rows only. Reuses common.rho and common.calibrate.
"""
from __future__ import annotations
import numpy as np

from common import rho, calibrate

POSTHOCS = ("raw", "tie", "cal", "gate", "soft")


def softplus(x):
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64))


def hard_gate(p, r, t, eps=1e-6):
    """Zero below the probability threshold; a positive rank-preserving map above it."""
    return np.where(p < t, 0.0, eps + softplus(r))


def soft_combine(p, r, a):
    """Dong-style product family: a=0 is probability*rank, a=1 is probability alone."""
    return p * (a + (1.0 - a) * np.clip(r, 0.0, 1.0))


def bottom_tie(score, tau):
    """Tie scores at/below tau to zero; tau=None means no tying."""
    if tau is None:
        return np.asarray(score, dtype=np.float64)
    return np.where(score <= tau, 0.0, score)


HARD_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.05))
SOFT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TIE_GRID = (None,) + tuple(round(x, 2) for x in np.arange(0.0, 0.9501, 0.05))


def select(objective, grid):
    """Grid value maximising the objective; ties break toward the earliest value."""
    return max(grid, key=lambda g: (objective(g), -grid.index(g)))


def worst_cell_rho(m, mask, score):
    """Minimum per-cell Spearman on ``mask`` (one cell for transfer; matches calibrate)."""
    part = m.loc[mask]
    return min(rho(part.loc[part.cell_line.eq(c), "gex"],
                   score[part.cell_line.eq(c).to_numpy()])
               for c in part.cell_line.unique())


def score_families(m, reg, clf_pair):
    """The five orderings for one regressor fold: {recipe: (scored evaluate vector, parameter)}.

    ``raw``/``tie``/``gate``/``soft`` use the helpers above; ``cal`` is common.calibrate.
    Every parameter is selected on the source-cell validation rows.
    """
    val = reg["validation"]
    rv, re = np.asarray(reg["raw_val"], float), np.asarray(reg["raw"], float)
    out = {"raw": (re, None)}
    tau = select(lambda t: worst_cell_rho(m, val, bottom_tie(rv, t)), TIE_GRID)
    out["tie"] = (bottom_tie(re, tau), tau)
    alpha, zero = calibrate(m, reg["train"], val, rv)
    cutoff = alpha * zero
    out["cal"] = (np.where(re <= cutoff, 0.0, re), cutoff)
    if clf_pair is not None:
        pv, pe = clf_pair["p_val"], clf_pair["p_eval"]
        g = select(lambda t: worst_cell_rho(m, val, hard_gate(pv, rv, t)), HARD_GRID)
        out["gate"] = (hard_gate(pe, re, g), g)
        a = select(lambda t: worst_cell_rho(m, val, soft_combine(pv, rv, t)), SOFT_GRID)
        out["soft"] = (soft_combine(pe, re, a), a)
    return out
