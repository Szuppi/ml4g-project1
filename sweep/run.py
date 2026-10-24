"""Drive the full recipe sweep: 3 regressors x {all, active} x {assays_only, assays-ref},
each scored by the five post-hoc orderings -> 60 configurations.

Each regressor fold is trained once; the orderings are cheap transforms of its raw output
(scoring.score_families). With ``save=True`` the models, raw local-test predictions and score
tables are cached under ``out_dir`` and a recipe is retrained only when its prediction file is
absent, so re-runs load instead of refit. Every per-configuration prediction is also returned
in memory (``registry``) for the notebook's plots and ensembles.

    python -m sweep.run --inputs inputs --out train_interactive/outputs/sweep --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from common import FOLDS, PROTOCOLS, load_inputs, masks, rho
from sweep.features import VERSIONS, tabular_features
from sweep.scoring import score_families
from sweep.models import classifier, regressor


def reg_key(kind, mode, version, protocol, fold):
    return f"{kind}_{mode}_{version}_{protocol}_{fold}"


def _save_model(kind, model, base, pre=None):
    if kind == "catboost":
        model.save_model(str(base) + ".cbm")
    elif kind == "ridge":
        joblib.dump(model, str(base) + ".joblib")
    elif kind == "cnn":
        torch.save(model.state_dict(), str(base) + ".pt")
        if pre is not None:
            np.savez_compressed(str(base) + "_pre.npz", **pre)


def _rows(m, evaluate, score, recipe, version, kind, mode, protocol, fold, param):
    """One per-cell Spearman row per evaluated cell line."""
    part = m.loc[evaluate]
    out = []
    for cell in sorted(part.cell_line.unique()):
        ix = part.cell_line.eq(cell).to_numpy()
        out.append(dict(recipe=recipe, version=version, regressor=kind, mode=mode,
                        protocol=protocol, fold=fold, cell_line=cell, n=int(ix.sum()),
                        param=param, spearman=rho(part.loc[ix, "gex"], score[ix])))
    return out


def classifier_metrics(m, clf, version, protocol, fold):
    """Target-cell diagnostics for the activity classifier (reported, not selected on)."""
    part = m.loc[clf["evaluate"]]
    out = []
    for cell in sorted(part.cell_line.unique()):
        ix = part.cell_line.eq(cell).to_numpy()
        z = (part.loc[ix, "gex"].to_numpy() > 0).astype(int)
        p = clf["p_eval"][ix]
        out.append(dict(version=version, protocol=protocol, fold=fold, cell_line=cell,
                        pos_fraction=float(z.mean()),
                        auroc=roc_auc_score(z, p) if 0 < z.mean() < 1 else np.nan,
                        avg_precision=average_precision_score(z, p) if z.any() else np.nan,
                        log_loss=log_loss(z, p, labels=[0, 1]),
                        brier=brier_score_loss(z, p)))
    return out


def summarize(frame):
    """Mean/worst transfer Spearman (both directions) and joint mean per recipe family."""
    frame = frame.copy()
    frame["family"] = frame.regressor + "_" + frame["mode"] + "[" + frame.version + "]|" + frame.recipe
    transfer = frame[frame.protocol != "joint"]
    joint = frame[frame.protocol == "joint"]
    rows = []
    for family, _ in frame.groupby("family"):
        gt = transfer[transfer.family == family]
        gj = joint[joint.family == family]
        rows.append(dict(family=family, transfer_mean=gt.spearman.mean(),
                         transfer_worst=gt.spearman.min() if len(gt) else np.nan,
                         joint_mean=gj.spearman.mean()))
    return pd.DataFrame(rows).sort_values("transfer_mean", ascending=False)


def run_sweep(inputs_dir, out_dir, device="cuda", save=False):
    """Train (or load) every recipe, score all five orderings, return scores + predictions."""
    inputs_dir, out_dir = Path(inputs_dir), Path(out_dir)
    for sub in ("preds", "models", "clf_preds"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    m, data = load_inputs(inputs_dir)
    tree_x = np.load(inputs_dir / "tree_features.npy")
    matrices = {v: tabular_features(tree_x, inputs_dir, v) for v in VERSIONS}
    recipes = [(v, k, mode) for v in VERSIONS
               for k in ("catboost", "ridge", "cnn") for mode in ("all", "active")]
    experiments, diagnostics = [], []
    registry = {"scored": {}, "raw": {}}
    t0 = time.time()
    for protocol in PROTOCOLS:
        for fold in FOLDS:
            train, validation, evaluate = masks(m, protocol, fold)
            clf = {}
            for v in VERSIONS:
                cpath = out_dir / "clf_preds" / f"{v}_{protocol}_{fold}.npz"
                if save and cpath.exists():
                    z = np.load(cpath)
                    clf[v] = dict(train=train, validation=validation, evaluate=evaluate,
                                  p_val=z["p_val"], p_eval=z["p_eval"])
                else:
                    c = classifier(m, matrices[v], protocol, fold)
                    if save:
                        np.savez_compressed(cpath, p_val=c["p_val"], p_eval=c["p_eval"])
                        c["model"].save_model(
                            str(out_dir / "models" / f"classifier_{v}_{protocol}_{fold}.cbm"))
                    c.pop("model", None)
                    clf[v] = c
                diagnostics += classifier_metrics(m, clf[v], v, protocol, fold)
            for version, kind, mode in recipes:
                ppath = out_dir / "preds" / f"{reg_key(kind, mode, version, protocol, fold)}.npz"
                if save and ppath.exists():
                    z = np.load(ppath)
                    reg = dict(train=train, validation=validation, evaluate=evaluate,
                               raw_val=z["raw_val"], raw=z["raw"])
                else:
                    reg = regressor(kind, m, matrices, data, inputs_dir,
                                    protocol, fold, version, mode, device)
                    if save:
                        np.savez_compressed(ppath, raw_val=np.asarray(reg["raw_val"]),
                                            raw=np.asarray(reg["raw"]))
                        _save_model(kind, reg["model"],
                                    out_dir / "models" / reg_key(kind, mode, version, protocol, fold),
                                    reg.get("pre"))
                for recipe, (scored, param) in score_families(m, reg, clf[version]).items():
                    experiments += _rows(m, evaluate, scored, recipe, version, kind,
                                         mode, protocol, fold, param)
                    registry["scored"][(f"{kind}_{mode}[{version}]|{recipe}",
                                        protocol, fold)] = np.asarray(scored, float)
                registry["raw"][(kind, mode, version, protocol, fold)] = np.asarray(reg["raw"], float)
                reg.pop("model", None)
                del reg
                if device == "cuda":
                    torch.cuda.empty_cache()
            print(json.dumps(dict(protocol=protocol, fold=fold,
                                  elapsed_min=round((time.time() - t0) / 60, 1))), flush=True)
    experiments = pd.DataFrame(experiments)
    summary = summarize(experiments)
    diagnostics = pd.DataFrame(diagnostics)
    if save:
        experiments.to_csv(out_dir / "experiments.csv", index=False)
        summary.to_csv(out_dir / "summary.csv", index=False)
        diagnostics.to_csv(out_dir / "classifier_metrics.csv", index=False)
    return dict(experiments=experiments, summary=summary,
                diagnostics=diagnostics, registry=registry)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_sweep(a.inputs, a.out, a.device, save=True)
