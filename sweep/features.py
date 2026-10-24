"""Feature sets for the sweep, and the active-gene fitting masks.

``assays_only`` is the 1565-column assay matrix from prepare.py; ``assays-ref`` appends the
reference-sequence block from prepare_reference.py. Rank targets come from common.targets.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from common import targets

VERSIONS = ("assays_only", "assays-ref")


def load_seq(inputs: Path) -> tuple[np.ndarray, list[dict]]:
    inputs = Path(inputs)
    block = np.load(inputs / "seq_features.npy")
    schema = json.loads((inputs / "seq_schema.json").read_text())
    return block, schema


def tabular_features(tree_x: np.ndarray, inputs: Path, version: str) -> np.ndarray:
    """``assays_only`` -> the 1565-column assay matrix; ``assays-ref`` -> that plus the
    reference-sequence block."""
    if version == "assays_only":
        return tree_x
    if version != "assays-ref":
        raise ValueError(f"Unknown feature version: {version}")
    block, _ = load_seq(inputs)
    return np.concatenate([tree_x, block], axis=1)


def active_mask(m, fit: np.ndarray) -> np.ndarray:
    """Restrict a fitting mask to measured-active genes (gex > 0, from labels only)."""
    return fit & (m.gex.to_numpy() > 0)


def fit_targets(m, fit: np.ndarray, active: bool) -> tuple[np.ndarray, np.ndarray]:
    """(fitting mask, within-cell rank targets) for an all-gene or active-only fit."""
    mask = active_mask(m, fit) if active else fit
    return mask, targets(m, mask)
