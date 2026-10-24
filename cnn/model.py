"""Compact residual CNN and training-only normalization."""
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _preprocess(feature_data, train, dev, kind):
    selected = np.r_[np.flatnonzero(train), np.flatnonzero(dev)]
    train_count = int(train.sum())
    local_train, local_dev = slice(0, train_count), slice(train_count, None)
    raw = np.concatenate([np.asarray(feature_data[k][selected], dtype=np.float32)
                          for k in ("bw56", "bed7", "coverage56")], axis=1)
    missing = ~np.isfinite(raw)
    values = _signed_log(np.where(missing, 0, raw))
    scaler = StandardScaler().fit(values[local_train])
    tab = np.concatenate([scaler.transform(values), missing.astype(np.float32)], axis=1)
    arrays = {"tab_mean": scaler.mean_, "tab_scale": scaler.scale_}
    train_inputs = [torch.from_numpy(tab[local_train].astype(np.float32))]
    dev_inputs = [torch.from_numpy(tab[local_dev].astype(np.float32))]
    if kind != "mlp":
        signal = np.asarray(feature_data["profile"][selected], dtype=np.float32)
        coverage = np.asarray(feature_data["profile_coverage"][selected], dtype=np.float32)
        missing_signal, missing_coverage = ~np.isfinite(signal), ~np.isfinite(coverage)
        logged = _signed_log(np.where(missing_signal, 0, signal))
        # Each assay shares scale across spatial bins; positions remain distinct.
        profile_scaler = StandardScaler().fit(logged[local_train].transpose(0, 2, 1).reshape(-1, 7))
        normalized = ((logged - profile_scaler.mean_[None, :, None]) /
                      profile_scaler.scale_[None, :, None])
        profile = np.concatenate([normalized, np.where(missing_coverage, 0, coverage),
                                  missing_signal, missing_coverage], axis=1).astype(np.float32)
        train_inputs.append(torch.from_numpy(profile[local_train]))
        dev_inputs.append(torch.from_numpy(profile[local_dev]))
        arrays.update(profile_mean=profile_scaler.mean_, profile_scale=profile_scaler.scale_)
    return train_inputs, dev_inputs, arrays


class _Residual(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Conv1d(width, width, 5, padding=2))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


def _predict(model, inputs, device, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs[0]), batch_size):
            predictions.append(model(*(x[start:start + batch_size].to(device) for x in inputs))
                               .detach().cpu().numpy())
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite neural predictions; result rejected.")
    return result


class CNN(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        width, dropout = 64, 0.3
        self.profile = nn.Sequential(
            nn.Conv1d(28, width, 7, padding=3), nn.GELU(),
            _Residual(width, dropout), nn.MaxPool1d(2),
            _Residual(width, dropout), _Residual(width, dropout),
            # Input here is length 50 (100 -> MaxPool1d(2) -> 50), so AvgPool1d(5) is
            # numerically identical to AdaptiveAvgPool1d(10) but has a deterministic CUDA
            # backward (adaptive_avg_pool lacks one). Enables reproducible GPU training.
            nn.AvgPool1d(5), nn.Flatten())
        self.context = nn.Sequential(nn.Linear(tab_dim, 64), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(width * 10 + 64, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1), nn.Softplus())

    def forward(self, tab, profile):
        return self.head(torch.cat([self.context(tab), self.profile(profile)], dim=1)).squeeze(-1)
