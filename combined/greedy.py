"""Greedy sequential insertion of the sweep models onto the 14-model ensemble (joint-D1).

Unlike cross_product.py (14-model ensemble + one fused unit at one weight), this adds the sweep
models one staged step at a time -- ensemble14 -> +model@w -> +model@w -> ... -- each (model,
ordering, weight) chosen with the previous additions already in place. This is exactly how the
ensemble's recipe adds its own members, and it's free (blends cached OOF, no retraining). Stops
when no addition improves joint-D1. Prints every round's trials and the committed path.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from combined import blendlib as B

OOF_SWEEP = ROOT / "combined" / "oof_sweep"
HIS14 = ROOT / "combined" / "oof_ensemble" / "ensemble14_joint_D1.csv"
ORDERINGS = ["cal", "tie", "gate"]
WEIGHTS = [5, 10, 15, 20, 30, 40]
MODELS = {"cnn": "cnn_all_assays-ref.csv", "catboost": "catboost_all_assays-ref.csv",
          "ridge": "ridge_all_assays-ref.csv"}


def jmean(frame):
    return B.summarize(frame, "prediction")["joint_mean"]


def main():
    ensemble14 = pd.read_csv(HIS14)
    candidates = {n: pd.read_csv(OOF_SWEEP / f).query("fold == 'D1'").reset_index(drop=True)
            for n, f in MODELS.items()}
    running = ensemble14.copy()
    current = jmean(running)
    print(f"start: ensemble-14 joint-D1 = {current:.4f}\n")
    available = set(MODELS)
    step = 0
    while available:
        step += 1
        trials = []
        for model in sorted(available):
            for ordering in ORDERINGS:
                member = candidates[model][B.KEYS + ["gex"]].copy()
                member["prediction"] = candidates[model][f"pred_{ordering}"].to_numpy()
                for w in WEIGHTS:
                    cand = B.staged_blend(running, member, w)
                    trials.append((jmean(cand), model, ordering, w))
        trials.sort(reverse=True)
        print(f"--- round {step}: top 5 candidate additions (of {len(trials)}) ---")
        for jm, model, ordering, w in trials[:5]:
            print(f"    +{model:8s} {ordering:4s} w={w:<2d} -> {jm:.4f}  (Δ {jm - current:+.4f})")
        best_jm, best_model, best_ord, best_w = trials[0]
        if best_jm <= current + 1e-9:
            print(f"    no addition improves {current:.4f}; stopping.\n")
            break
        member = candidates[best_model][B.KEYS + ["gex"]].copy()
        member["prediction"] = candidates[best_model][f"pred_{best_ord}"].to_numpy()
        running = B.staged_blend(running, member, best_w)
        print(f"  COMMIT +{best_model} {best_ord} w={best_w}: {current:.4f} -> {best_jm:.4f}\n")
        current = best_jm
        available.discard(best_model)
    print(f"greedy final joint-D1 = {current:.4f}  (ensemble-14 baseline 0.8549, "
          f"best single-unit cnn·gate·w40 = 0.8563)")


if __name__ == "__main__":
    main()
