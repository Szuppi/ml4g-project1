"""The sweep's fit units: the activity classifier and the three regressor folds.

Each fold returns continuous validation/evaluate predictions (and the fitted model, so it can
later score X3). The CNN reuses cnn/model.py; the post-hoc orderings live in scoring.py.
"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, CatBoostRegressor

from common import masks, targets
from cnn.model import CNN, _preprocess, _predict
from sweep.features import active_mask, fit_targets, load_seq
from sweep.scoring import worst_cell_rho

_CATBOOST = dict(iterations=960, depth=5, learning_rate=0.05, l2_leaf_reg=10, random_seed=42,
                 thread_count=8, verbose=False, allow_writing_files=False, border_count=64,
                 nan_mode="Min")
PENALTIES = (0.1, 1, 10, 100.0, 1000.0, 5_000, 10_000, 50_000, 100_000)


# --------------------------------------------------------------------------- #
# Activity classifier C(X)
# --------------------------------------------------------------------------- #
def classifier(m, x, protocol, fold):
    """CatBoost activity classifier; probabilities on validation and evaluate."""
    train, validation, evaluate = masks(m, protocol, fold)
    z = (m.loc[train, "gex"].to_numpy() > 0).astype(int)
    model = CatBoostClassifier(loss_function="Logloss", **_CATBOOST)
    model.fit(x[train], z)
    proba = model.predict_proba(x)[:, 1]
    return dict(train=train, validation=validation, evaluate=evaluate,
                p_val=proba[validation], p_eval=proba[evaluate], model=model)


# --------------------------------------------------------------------------- #
# Regressors R(X)
# --------------------------------------------------------------------------- #
def catboost_fold(m, x, protocol, fold, active=False):
    """CatBoost regressor; continuous predictions on validation and evaluate."""
    train, validation, evaluate = masks(m, protocol, fold)
    fit = active_mask(m, train) if active else train
    model = CatBoostRegressor(loss_function="RMSE", **_CATBOOST)
    model.fit(x[fit], targets(m, fit))
    return dict(train=train, validation=validation, evaluate=evaluate,
                raw_val=model.predict(x[validation]), raw=model.predict(x[evaluate]),
                model=model)


def _ridge_pipeline(penalty):
    """Train-fitted median imputation + standardisation + ridge; no label leakage."""
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=penalty))


def ridge_fold(m, x, protocol, fold, active=False):
    """Ridge regressor; the L2 penalty is chosen on the source-cell validation rows."""
    train, validation, evaluate = masks(m, protocol, fold)
    fit, y = fit_targets(m, train, active)
    best = None
    for penalty in PENALTIES:
        model = _ridge_pipeline(penalty).fit(x[fit], y)
        score = worst_cell_rho(m, validation, model.predict(x[validation]))
        if best is None or score > best[0]:
            best = (score, model)
    model = best[1]
    return dict(train=train, validation=validation, evaluate=evaluate,
                raw_val=model.predict(x[validation]), raw=model.predict(x[evaluate]),
                model=model)


def _transform_seq(block, schema):
    """Nonnegative counts/distances get log1p; bounded fractions/indicators stay as is."""
    out = block.astype(np.float64).copy()
    for j, col in enumerate(schema):
        if col["kind"] in ("count", "distance"):
            out[:, j] = np.log1p(out[:, j])
    return out


def _seq_tab(inputs, train, dev):
    """Train-fitted standardisation + missingness for the sequence block, in train|dev order."""
    seq, schema = load_seq(inputs)
    order = np.r_[np.flatnonzero(train), np.flatnonzero(dev)]
    transformed = _transform_seq(seq[order], schema)
    missing = ~np.isfinite(transformed)
    filled = np.where(missing, 0.0, transformed)
    scaler = StandardScaler().fit(filled[:int(train.sum())])
    tab = np.concatenate([scaler.transform(filled), missing.astype(np.float32)], axis=1)
    return tab[:int(train.sum())].astype(np.float32), tab[int(train.sum()):].astype(np.float32)


def cnn_fold(m, data, inputs, protocol, fold, device, version="assays_only", active=False):
    """Fit one CNN regressor (``assays-ref`` appends the sequence features to the tabular branch;
    ``active`` fits on measured-active genes only). Spatial branch, 8-epoch schedule, ranking
    loss and epoch 6-8 weight average are identical to the committed CNN (cnn/train.py)."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train, validation, evaluate = masks(m, protocol, fold)
    dev = validation | evaluate
    ti, ei, pre = _preprocess(data, train, dev, "rescnn")
    if version == "assays-ref":
        seq_tr, seq_dev = _seq_tab(inputs, train, dev)
        ti = [torch.cat([ti[0], torch.from_numpy(seq_tr)], 1), ti[1]]
        ei = [torch.cat([ei[0], torch.from_numpy(seq_dev)], 1), ei[1]]

    fit = active_mask(m, train) if active else train
    keep = torch.from_numpy(m.gex.to_numpy()[np.flatnonzero(train)] > 0) if active \
        else torch.ones(int(train.sum()), dtype=torch.bool)
    labels = torch.from_numpy(targets(m, fit))
    cells = torch.from_numpy(
        m.loc[fit, "cell_line"].map({"X1": 0, "X2": 1}).to_numpy(dtype=np.int64, copy=True))
    model = CNN(ti[0].shape[1]).to(device)
    loader = DataLoader(TensorDataset(ti[0][keep], ti[1][keep], labels, cells),
                        batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    average, count = None, 0
    for epoch in range(1, 9):
        model.train()
        for tab, profile, label, cell in loader:
            tab, profile = tab.to(device), profile.to(device)
            label, cell = label.to(device), cell.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tab, profile)
            loss = nn.functional.mse_loss(prediction, label)
            perm = torch.randperm(len(label)).to(device)
            difference = label - label[perm]
            valid = (cell == cell[perm]) & (difference != 0)
            if valid.any():
                loss = loss + 0.01 * nn.functional.softplus(
                    -torch.sign(difference[valid])
                    * (prediction[valid] - prediction[perm][valid]) / 0.1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient")
            optimizer.step()
        if epoch >= 6:
            if average is None:
                average = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                for k, v in model.state_dict().items():
                    average[k].add_(v.detach())
            count += 1
    model.load_state_dict({k: v / count for k, v in average.items()})

    pred = _predict(model, ei, torch.device(device), 128).astype(np.float64)
    return dict(train=train, validation=validation, evaluate=evaluate,
                raw_val=pred[validation[dev]], raw=pred[evaluate[dev]], model=model, pre=pre)


def regressor(kind, m, matrices, data, inputs, protocol, fold, version, mode, device):
    """Dispatch to one of the three regressor folds for (version, kind, mode)."""
    active = mode == "active"
    if kind == "catboost":
        return catboost_fold(m, matrices[version], protocol, fold, active)
    if kind == "ridge":
        return ridge_fold(m, matrices[version], protocol, fold, active)
    if kind == "cnn":
        return cnn_fold(m, data, inputs, protocol, fold, device, version, active)
    raise ValueError(kind)
