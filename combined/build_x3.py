"""Build two X3 submissions by blending the sweep models onto the 14-model ensemble.

The sweep models are refit on ALL X1+X2 rows (no chromosome exclusion, matching the
ensemble's production regime) and predict the 1,984 X3 genes, using the sweep's exact
training (so the X3 outputs are consistent with the joint-D1 OOF the greedy sweep scored):

  blend-15-cnn            = 14-model ensemble + cnn.gate.w40
  blend-16-cnn-catboost   = 14-model ensemble + cnn.gate.w40 + catboost.cal.w5

The 14-model ensemble's X3 ranks are its verified ensemble_14 output (byte-identical to the
committed 0.85305 submission). Ordering cutoffs are the medians of the sweep's joint-fold
selections (cnn gate t=0.5, catboost cal cutoff=0.4672). The staged blend follows the
ensemble's own idiom: final = (100-w)*rank(running) + w*rank(member), rank = 2*average-tie-rank.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import rankdata
from catboost import CatBoostRegressor, CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn.model import CNN, _preprocess, _predict
from sweep.features import tabular_features
from sweep.models import _seq_tab, _CATBOOST
from sweep.scoring import hard_gate
from common import targets

INPUTS = ROOT / "combined" / "work" / "inputs_x3"
ENSEMBLE14_X3 = ROOT / "combined" / "work" / "ensemble_x3_sub" / "ensemble_14.csv"
OUT = ROOT / "combined" / "submissions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GATE_T = 0.5            # cnn gate threshold, median of joint-fold selections
CAL_CUT = 0.4672        # catboost cal cutoff, median of joint-fold selections
VERSION = "assays-ref"


def rank(v):
    return 2.0 * rankdata(np.asarray(v, float), method="average")


def seed_all():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)


def cnn_x3(m, data, source, x3):
    """Refit the assays-ref CNN on all source rows (sweep training verbatim); predict X3."""
    seed_all()
    ti, ei, _pre = _preprocess(data, source, x3, "rescnn")
    seq_tr, seq_dev = _seq_tab(INPUTS, source, x3)
    ti = [torch.cat([ti[0], torch.from_numpy(seq_tr)], 1), ti[1]]
    ei = [torch.cat([ei[0], torch.from_numpy(seq_dev)], 1), ei[1]]
    labels = torch.from_numpy(targets(m, source))
    cells = torch.from_numpy(m.loc[source, "cell_line"].map({"X1": 0, "X2": 1})
                            .to_numpy(dtype=np.int64, copy=True))
    model = CNN(ti[0].shape[1]).to(DEVICE)
    loader = DataLoader(TensorDataset(ti[0], ti[1], labels, cells), batch_size=256,
                        shuffle=True, generator=torch.Generator().manual_seed(42))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    average, count = None, 0
    for epoch in range(1, 9):
        model.train()
        for tab, profile, label, cell in loader:
            tab, profile = tab.to(DEVICE), profile.to(DEVICE)
            label, cell = label.to(DEVICE), cell.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(tab, profile)
            loss = nn.functional.mse_loss(pred, label)
            perm = torch.randperm(len(label)).to(DEVICE)
            diff = label - label[perm]
            valid = (cell == cell[perm]) & (diff != 0)
            if valid.any():
                loss = loss + 0.01 * nn.functional.softplus(
                    -torch.sign(diff[valid]) * (pred[valid] - pred[perm][valid]) / 0.1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if epoch >= 6:
            if average is None:
                average = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                for k, v in model.state_dict().items():
                    average[k].add_(v.detach())
            count += 1
    model.load_state_dict({k: v / count for k, v in average.items()})
    return _predict(model, ei, torch.device(DEVICE), 128).astype(np.float64)


def main():
    m = pd.read_csv(INPUTS / "metadata.csv")
    with np.load(INPUTS / "features.npz") as z:
        data = {k: z[k] for k in z.files}
    tree_x = np.load(INPUTS / "tree_features.npy")
    X = tabular_features(tree_x, INPUTS, VERSION)
    source = m.cell_line.isin(["X1", "X2"]).to_numpy()
    x3 = m.cell_line.eq("X3").to_numpy()
    y_source = targets(m, source)

    # --- sweep models on X3 (cnn_x3 returns the 1,984 X3 rows in metadata order) ---
    cnn_pred = cnn_x3(m, data, source, x3)
    clf = CatBoostClassifier(loss_function="Logloss", **_CATBOOST)
    clf.fit(X[source], (m.loc[source, "gex"].to_numpy() > 0).astype(int))
    clf_p = clf.predict_proba(X[x3])[:, 1]
    reg = CatBoostRegressor(loss_function="RMSE", **_CATBOOST)
    reg.fit(X[source], y_source)
    cat_pred = reg.predict(X[x3])

    cnn_gate = hard_gate(clf_p, cnn_pred, GATE_T)
    cat_cal = np.where(cat_pred <= CAL_CUT, 0.0, cat_pred)

    # align the sweep X3 predictions to the ensemble's gene order
    ensemble = pd.read_csv(ENSEMBLE14_X3)
    genes = ensemble.gene_name.tolist()
    preds = pd.DataFrame({"gene_name": m.loc[x3, "gene_name"].to_numpy(),
                          "cnn_gate": cnn_gate, "cat_cal": cat_cal}).set_index("gene_name")
    assert set(genes) == set(preds.index) and len(genes) == 1984, "X3 gene set mismatch"
    preds = preds.loc[genes]
    ensemble14 = ensemble.gex_predicted.to_numpy(dtype=float)

    def staged(running, member, w):
        return (100 - w) * rank(running) + w * rank(member)

    blend15 = staged(ensemble14, preds.cnn_gate.to_numpy(), 40)
    blend16 = staged(blend15, preds.cat_cal.to_numpy(), 5)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, score in [("blend-15-cnn", blend15), ("blend-16-cnn-catboost", blend16)]:
        assert len(score) == 1984 and np.isfinite(score).all() and np.ptp(score) > 0
        df = pd.DataFrame({"gene_name": genes, "gex_predicted": score.astype(np.int64)})
        assert df.gene_name.tolist() == genes and not df.gene_name.duplicated().any()
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"wrote {OUT / (name + '.csv')}  genes={len(df)}  "
              f"rank_range=[{df.gex_predicted.min()},{df.gex_predicted.max()}]")


if __name__ == "__main__":
    main()
