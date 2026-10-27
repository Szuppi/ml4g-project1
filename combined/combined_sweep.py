"""Combination sweep: add the sweep models onto the frozen 14-model ensemble -> a ranked toplist.

Greedy forward selection: start from the frozen 14-model ensemble and add the sweep models one
staged step at a time (ensemble14 -> +model.ordering.w -> ...), each (model, ordering, weight)
chosen with the previous additions already in place -- exactly how the ensemble's own recipe
adds its members. Every trial at every round is recorded as a full combined recipe, so the
returned toplist is the ranked menu from which the two blended submissions were picked. Fast:
it only blends the cached joint-D1 OOF (no retraining).

load_oof(repo) -> (ensemble14 frame, {model: oof frame}); run_combined_sweep(...) -> toplist df.
Run as a script to also write combined/results/combined_toplist.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from combined import blendlib as B

ORDERINGS = ["cal", "tie", "gate"]
WEIGHTS = [5, 10, 15, 20, 30, 40]
MODELS = {"cnn": "cnn_all_assays-ref.csv", "catboost": "catboost_all_assays-ref.csv",
          "ridge": "ridge_all_assays-ref.csv"}


def load_oof(repo):
    repo = Path(repo)
    ensemble14 = pd.read_csv(repo / "combined" / "oof_ensemble" / "ensemble14_joint_D1.csv")
    candidates = {n: pd.read_csv(repo / "combined" / "oof_sweep" / f).query("fold == 'D1'").reset_index(drop=True)
            for n, f in MODELS.items()}
    return ensemble14, candidates


def _member(candidates, model, ordering):
    m = candidates[model][B.KEYS + ["gex"]].copy()
    m["prediction"] = candidates[model][f"pred_{ordering}"].to_numpy()
    return m


def run_combined_sweep(ensemble14, candidates, orderings=ORDERINGS, weights=WEIGHTS):
    """Greedy sweep; returns a toplist DataFrame of every combined recipe tried, ranked."""
    def summ(frame):
        s = B.summarize(frame, "prediction")
        return s["joint_mean"], s["joint_worst"], s["cells"][("D1", "X1")], s["cells"][("D1", "X2")]

    base_jm, base_worst, base_x1, base_x2 = summ(ensemble14)
    rows = [dict(recipe="ensemble-14", n_add=0, X1=base_x1, X2=base_x2,
                 joint_worst=base_worst, joint_mean=base_jm, on_greedy_path=True)]
    running, used, prefix = ensemble14.copy(), set(), []
    current = base_jm
    committed = {"ensemble-14"}
    while len(used) < len(candidates):
        trials = []
        for model in sorted(set(candidates) - used):
            for ordering in orderings:
                member = _member(candidates, model, ordering)
                for w in weights:
                    cand = B.staged_blend(running, member, w)
                    jm, worst, x1, x2 = summ(cand)
                    recipe = " + ".join(["ensemble-14"] + prefix + [f"{model}.{ordering}.w{w}"])
                    rows.append(dict(recipe=recipe, n_add=len(prefix) + 1, X1=x1, X2=x2,
                                     joint_worst=worst, joint_mean=jm, on_greedy_path=False))
                    trials.append((jm, model, ordering, w, cand))
        trials.sort(key=lambda t: t[0], reverse=True)
        best_jm, model, ordering, w, cand = trials[0]
        if best_jm <= current + 1e-9:
            break
        running, current = cand, best_jm
        used.add(model)
        prefix.append(f"{model}.{ordering}.w{w}")
        committed.add(" + ".join(["ensemble-14"] + prefix))

    top = pd.DataFrame(rows).drop_duplicates("recipe")
    top["on_greedy_path"] = top["recipe"].isin(committed)
    top["delta_vs_ensemble14"] = (top["joint_mean"] - base_jm).round(4)
    top["below_baseline"] = top["delta_vs_ensemble14"] < 0
    for c in ["X1", "X2", "joint_worst", "joint_mean"]:
        top[c] = top[c].round(4)
    return top.sort_values(["joint_mean", "n_add"], ascending=[False, True]).reset_index(drop=True)


def main():
    ensemble14, candidates = load_oof(ROOT)
    top = run_combined_sweep(ensemble14, candidates)
    dest = ROOT / "combined" / "results" / "combined_toplist.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(dest, index=False)
    base = float(top.loc[top.recipe == "ensemble-14", "joint_mean"].iloc[0])
    print(f"wrote {dest}  ({len(top)} recipes)  ensemble-14 baseline = {base:.4f}\n")
    cols = ["recipe", "n_add", "X1", "X2", "joint_worst", "joint_mean", "delta_vs_ensemble14", "on_greedy_path"]
    with pd.option_context("display.max_colwidth", None, "display.width", 160):
        print(top[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
