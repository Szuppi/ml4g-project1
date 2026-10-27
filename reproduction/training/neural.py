"""A-only neural model development; no competition submissions or X3 exports.

run_neural expects aligned metadata/feature rows. All imputers and scalers are
fitted on the protocol's training rows. Official validation B is never accessed.
The checkpoint describes architecture, preprocessing, split and label transform.
"""
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_SPECS = [
    {"name": "mlp128", "kind": "mlp", "width": 128, "dropout": 0.2},
    {"name": "cnn32", "kind": "cnn", "width": 32, "dropout": 0.2},
    {"name": "rescnn64", "kind": "rescnn", "width": 64, "dropout": 0.3},
    {"name": "profile_mlp128", "kind": "profile_mlp", "width": 128, "dropout": 0.2},
]
DEV_CHROMOSOMES = ("chr3", "chr11", "chr17")


def _json(path, value):
    """Write valid JSON, representing unavailable metrics as null."""
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, Path):
            return str(x)
        return x
    Path(path).write_text(json.dumps(clean(value), indent=2) + "\n")


def _spearman(y, pred):
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(pred) == 0:
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def _metrics(meta, y, pred_log):
    pred = np.expm1(np.minimum(pred_log.astype(np.float64), 80.0))
    rows = []
    groups = [("pooled", "all", np.ones(len(y), dtype=bool))]
    for field in ("cell_line", "chr"):
        groups.extend((field, str(value), (meta[field] == value).to_numpy())
                      for value in sorted(meta[field].unique()))
    for field, value, mask in groups:
        rows.append({"group": field, "value": value, "n": int(mask.sum()),
                     "spearman": _spearman(y[mask], pred[mask]),
                     "rmse_log1p": float(np.sqrt(np.mean(
                         (np.log1p(y[mask]) - pred_log[mask]) ** 2)))})
    return rows, pred


def _selection(metrics):
    values = [row["spearman"] for row in metrics if row["group"] == "cell_line"]
    # Every held-out cell line must have a defined rank correlation.
    return min(values) if values and all(np.isfinite(values)) else -float("inf")


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


class _Regressor(nn.Module):
    def __init__(self, kind, tab_dim, width, dropout, profile_channels=28):
        super().__init__()
        self.kind = kind
        if kind in {"mlp", "profile_mlp"}:
            input_dim = tab_dim if kind == "mlp" else tab_dim + profile_channels * 100
            self.head = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(width // 2, 1), nn.Softplus())
        else:
            layers = [nn.Conv1d(profile_channels, width, 7, padding=3), nn.GELU()]
            if kind == "rescnn":
                layers += [_Residual(width, dropout), nn.MaxPool1d(2),
                           _Residual(width, dropout), _Residual(width, dropout)]
            else:
                layers += [nn.MaxPool1d(2), nn.Conv1d(width, width, 5, padding=2), nn.GELU()]
            layers += [nn.AdaptiveAvgPool1d(10), nn.Flatten()]
            self.profile = nn.Sequential(*layers)
            self.context = nn.Sequential(nn.Linear(tab_dim, 64), nn.GELU(), nn.Dropout(dropout))
            self.head = nn.Sequential(nn.Linear(width * 10 + 64, 128), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(128, 1), nn.Softplus())

    def forward(self, tab, profile=None):
        if self.kind == "mlp":
            return self.head(tab).squeeze(-1)
        if self.kind == "profile_mlp":
            return self.head(torch.cat([tab, profile.flatten(start_dim=1)], dim=1)).squeeze(-1)
        return self.head(torch.cat([self.context(tab), self.profile(profile)], dim=1)).squeeze(-1)


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


def _audit_inputs(metadata, feature_data, specs):
    required = {"cell_line", "split", "chr", "gex"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(metadata.columns))}")
    if not metadata["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Unrecognized split; expected train=A, val=B, test=X3.")
    a = metadata["split"].eq("train")
    expected_a = {f"chr{i}" for i in range(2, 23)} - {"chr14", "chr19"}
    if not metadata.loc[a, "chr"].isin(expected_a).all():
        raise ValueError("Training rows include a chromosome outside official A.")
    if not metadata.loc[a, "cell_line"].isin(["X1", "X2"]).all():
        raise ValueError("Training rows must belong only to X1/X2.")
    labels = pd.to_numeric(metadata.loc[a, "gex"], errors="coerce").to_numpy()
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("A labels must be finite nonnegative raw CAGE; verify scale before log1p.")
    shapes = {"bw56": (len(metadata), 56), "bed7": (len(metadata), 7),
              "coverage56": (len(metadata), 56)}
    if any(spec["kind"] != "mlp" for spec in specs):
        shapes.update(profile=(len(metadata), 7, 100), profile_coverage=(len(metadata), 7, 100))
    for key, shape in shapes.items():
        if key not in feature_data or np.shape(feature_data[key]) != shape:
            raise ValueError(f"{key} must have shape {shape}, aligned to metadata.")
        if np.isinf(np.asarray(feature_data[key])[a.to_numpy()]).any():
            raise ValueError(f"{key}: infinite A features are invalid, not missing values.")
    for key in ("coverage56", "profile_coverage"):
        if key in shapes:
            values = np.asarray(feature_data[key])[a.to_numpy()]
            finite = values[np.isfinite(values)]
            if (finite < 0).any() or (finite > 1.00001).any():
                raise ValueError(f"{key} must contain covered-base fractions in [0,1].")
    return a.to_numpy()


def run_neural(metadata, feature_data, output_dir, experiment_specs=None, *,
               seed=42, max_epochs=40, patience=8, batch_size=256,
               learning_rate=1e-3, weight_decay=1e-3, allow_cpu=False):
    """Run each specification on three A-only protocols, return metrics DataFrame.

    Pooled uses both X1/X2; transfer trains one cell line, stops on that source
    cell's chr3/11/17, then evaluates the other's chr3/11/17. The same three
    chromosomes are excluded from training in every protocol. allow_cpu is
    explicitly for functional/small CPU runs.
    Each spec may override max_epochs, patience, batch_size, learning_rate and
    weight_decay. B labels/features never influence preprocessing or selection.
    """
    if max_epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("Epochs, patience and batch size must be positive.")
    specs = copy.deepcopy(DEFAULT_SPECS if experiment_specs is None else experiment_specs)
    if not specs or len({s["name"] for s in specs}) != len(specs):
        raise ValueError("Provide nonempty specifications with unique names.")
    for spec in specs:
        if spec["kind"] not in {"mlp", "cnn", "rescnn", "profile_mlp"}:
            raise ValueError(f"Unknown architecture: {spec['kind']}")
        default = next(item for item in DEFAULT_SPECS if item["kind"] == spec["kind"])
        spec.setdefault("width", default["width"])
        spec.setdefault("dropout", default["dropout"])
        if not spec["name"] or Path(spec["name"]).name != spec["name"]:
            raise ValueError("Spec name must be a plain directory name.")
        if spec["width"] < 2 or not 0 <= spec["dropout"] < 1:
            raise ValueError("Width must be >=2 and dropout in [0,1).")
    meta = metadata.reset_index(drop=True).copy()
    a = _audit_inputs(meta, feature_data, specs)
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA GPU required. Use allow_cpu=True only for explicit CPU checks.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    out = Path(output_dir)
    if (out / "neural_metrics.json").exists():
        raise FileExistsError("Choose a new output directory; completed experiments are preserved.")
    out.mkdir(parents=True, exist_ok=True)
    _json(out / "predefined_design.json", {"specifications": specs, "seed": seed,
          "max_epochs": max_epochs, "patience": patience, "batch_size": batch_size,
          "learning_rate": learning_rate, "weight_decay": weight_decay,
          "dev_chromosomes": DEV_CHROMOSOMES, "official_B_reserved": True,
          "transfer_stopping": "source-cell development chromosomes only",
          "selection": "minimum cell-line Spearman on stopping rows",
          "label_transform": "log1p(raw nonnegative CAGE)"})
    held_chr = meta["chr"].isin(DEV_CHROMOSOMES).to_numpy()
    protocols = [("pooled", ("X1", "X2"), ("X1", "X2")),
                 ("X1_to_X2", ("X1",), ("X2",)), ("X2_to_X1", ("X2",), ("X1",))]
    all_metrics, experiment_audits = [], []
    for protocol, train_lines, dev_lines in protocols:
        train = a & ~held_chr & meta["cell_line"].isin(train_lines).to_numpy()
        dev = a & held_chr & meta["cell_line"].isin(dev_lines).to_numpy()
        stopping = a & held_chr & meta["cell_line"].isin(train_lines).to_numpy()
        evaluation = dev | stopping
        if not train.any() or not dev.any() or not stopping.any():
            raise ValueError(f"{protocol}: empty training/development rows.")
        train_idx, dev_idx = np.flatnonzero(train), np.flatnonzero(dev)
        assert not np.intersect1d(train_idx, dev_idx).size
        y_train = meta.loc[train, "gex"].to_numpy(dtype=np.float32)
        y_dev = meta.loc[dev, "gex"].to_numpy(dtype=np.float64)
        y_stop = meta.loc[stopping, "gex"].to_numpy(dtype=np.float64)
        for base_spec in specs:
            spec = {"max_epochs": max_epochs, "patience": patience, "batch_size": batch_size,
                    "learning_rate": learning_rate, "weight_decay": weight_decay, **base_spec}
            if spec["max_epochs"] < 1 or spec["patience"] < 1 or spec["batch_size"] < 1:
                raise ValueError("Spec epochs, patience and batch size must be positive.")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            started = time.monotonic()
            dest = out / spec["name"] / protocol
            if (dest / "audit.json").exists():
                raise FileExistsError(f"Completed run already exists: {dest}")
            dest.mkdir(parents=True, exist_ok=True)
            train_inputs, evaluation_inputs, scaler = _preprocess(feature_data, train, evaluation, spec["kind"])
            dev_inputs = [tensor[dev[evaluation]] for tensor in evaluation_inputs]
            stop_inputs = [tensor[stopping[evaluation]] for tensor in evaluation_inputs]
            np.savez_compressed(dest / "preprocessing.npz", **scaler)
            model = _Regressor(spec["kind"], train_inputs[0].shape[1], spec["width"],
                               spec["dropout"]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"],
                                           weight_decay=spec["weight_decay"])
            loader = DataLoader(TensorDataset(*train_inputs, torch.from_numpy(np.log1p(y_train))),
                                batch_size=spec["batch_size"], shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
            best_score, best_epoch, best_state, stale, history = -float("inf"), 0, None, 0, []
            for epoch in range(1, spec["max_epochs"] + 1):
                model.train()
                total_loss = 0.0
                for batch in loader:
                    *inputs, target = [x.to(device) for x in batch]
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(*inputs), target)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss in {spec['name']}/{protocol}.")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(target)
                prediction = _predict(model, stop_inputs, device, spec["batch_size"])
                metrics, _ = _metrics(meta.loc[stopping], y_stop, prediction)
                score = _selection(metrics)
                pooled = next(row for row in metrics if row["group"] == "pooled")
                history.append({"epoch": epoch, "train_mse_log1p": total_loss / len(y_train),
                                "stopping_min_cellline_spearman": score if np.isfinite(score) else float("nan"),
                                "stopping_pooled_spearman": pooled["spearman"],
                                "stopping_rmse_log1p": pooled["rmse_log1p"]})
                if best_state is None or score > best_score + 1e-6:
                    best_score, best_epoch = score, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= spec["patience"]:
                    break
            model.load_state_dict(best_state)
            prediction = _predict(model, dev_inputs, device, spec["batch_size"])
            metrics, raw_prediction = _metrics(meta.loc[dev], y_dev, prediction)
            for row in metrics:
                row.update(model=spec["name"], protocol=protocol, seed=seed, best_epoch=best_epoch)
            all_metrics.extend(metrics)
            result = meta.loc[dev].copy()
            result.insert(0, "source_row", dev_idx)
            result["prediction_log1p"] = prediction
            result["prediction_raw_cage"] = raw_prediction
            result.to_csv(dest / "dev_predictions.csv", index=False)
            pd.DataFrame(metrics).to_csv(dest / "metrics.csv", index=False)
            pd.DataFrame(history).to_csv(dest / "epochs.csv", index=False)
            _json(dest / "metrics.json", metrics)
            _json(dest / "epochs.json", history)
            audit = {"model": spec["name"], "protocol": protocol, "specification": spec,
                     "seed": seed, "device": str(device), "torch_version": str(torch.__version__),
                     "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                     "epochs_run": len(history), "best_epoch": best_epoch,
                     "early_stopped": len(history) < spec["max_epochs"],
                     "best_stopping_min_cellline_spearman": best_score,
                     "selection_valid": bool(np.isfinite(best_score)),
                     "runtime_seconds": time.monotonic() - started,
                     "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                     "parameter_count": sum(p.numel() for p in model.parameters()),
                     "train_rows": train_idx, "dev_rows": dev_idx,
                     "stopping_rows": np.flatnonzero(stopping),
                     "train_chromosomes": sorted(meta.loc[train, "chr"].unique()),
                     "dev_chromosomes": sorted(meta.loc[dev, "chr"].unique()),
                     "train_cell_lines": train_lines, "dev_cell_lines": dev_lines,
                     "stopping_cell_lines": train_lines,
                     "official_B_used": False, "X3_used": False,
                     "label_transform": "log1p(raw nonnegative CAGE); softplus output",
                     "preprocessing": "signed log1p; missing->0 plus indicators; training-only StandardScaler; profile coverage channels",
                     "row_identity_sha256": hashlib.sha256(meta[[column for column in
                             ("cell_line", "split", "gene_name", "chr") if column in meta]]
                                                           .to_csv(index=False).encode()).hexdigest()}
            torch.save({"state_dict": best_state, "specification": spec,
                        "tab_dimension": train_inputs[0].shape[1], "profile_channels": 28,
                        "seed": seed, "best_epoch": best_epoch,
                        "label_transform": audit["label_transform"]}, dest / "model.pt")
            _json(dest / "audit.json", audit)
            experiment_audits.append(audit)
            dev_score = _selection(metrics)
            print(f"{spec['name']} {protocol}: best epoch {best_epoch}, dev minimum Spearman "
                  f"{dev_score:.4f}, {audit['runtime_seconds']:.1f}s", flush=True)
            del model, optimizer, loader, train_inputs, dev_inputs, stop_inputs, evaluation_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    frame = pd.DataFrame(all_metrics)
    frame.to_csv(out / "neural_metrics.csv", index=False)
    _json(out / "neural_metrics.json", all_metrics)
    _json(out / "neural_audit.json", experiment_audits)
    return frame
