"""Replay the fixed eleven-model rank ensemble from saved model outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parent


def rank(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite model output")
    return 2 * rankdata(values, method="average")


def blend(components, heads, recipe, source_zero):
    c = recipe["base"]
    names = ["cnn", "tree", "weighted", "dilated", "gain"]
    members = [rank(np.where(components[n] <= a * source_zero, 0, components[n]))
               for n, a in zip(names, c["alphas"])]
    score = (100 - c["weights"][0]) * members[0] + c["weights"][0] * members[1]
    for member, weight in zip(members[2:], c["weights"][1:]):
        score = (100 - weight) * rank(score) + weight * member

    rich = rank(np.where(components["rich"] <= c["rich_alpha"] * source_zero,
                         0, components["rich"]))
    score = (100 - c["rich_weight"]) * rank(score) + c["rich_weight"] * rich
    seeds = sum(rank(np.where(components[n] <= c["seed_alpha"] * components[n + "_zero"],
                                  0, components[n])) for n in ("seed43", "seed44"))
    score = (100 - c["seed_weight"]) * rank(score) + c["seed_weight"] * rank(seeds)
    extra = rank(np.where(components["extra"] <= c["extra_alpha"] * components["extra_zero"],
                          0, components["extra"]))
    score = (100 - c["extra_weight"]) * rank(score) + c["extra_weight"] * extra

    h = recipe["heads"]
    member_ranks = []
    for name in h["models"]:
        value = heads[name + "_production_" + h["normalization"]]
        z = components[name + "_zero"]
        floor = min(0.99, h["floor_scale"] * z / (2 - z))
        active = expit(value[:, 0])
        output = np.where(active <= h["activity_cut"], 0,
                          active ** h["gamma"] * (floor + (1 - floor) * value[:, 1]))
        member_ranks.append(rank(output))
    head_score = rank(h["basic_weight"] * member_ranks[0] +
                      (1 - h["basic_weight"]) * member_ranks[1])
    return (100 - h["weight"]) * rank(score) + h["weight"] * head_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    args = parser.parse_args()
    source = ROOT / "components"
    recipe = json.loads((ROOT / "recipe.json").read_text())
    first = json.loads((source / "new_inference_manifest.json").read_text())
    heads_info = json.loads((source / "new_head_inference_manifest.json").read_text())
    if first["genes"] != heads_info["genes"]:
        raise ValueError("Model output gene order differs")
    genes = first["genes"]
    if len(genes) != 1984 or len(set(genes)) != len(genes):
        raise ValueError("Unexpected gene list")
    c = {}
    with np.load(source / "raw_components.npz") as data:
        for name in ("cnn", "tree", "weighted", "dilated", "gain"):
            c[name] = data[name]
    with np.load(source / "old_tree_view_raw.npz") as data:
        c["rich"] = data["rich_1.0"]
    with np.load(source / "new_raw_predictions.npz") as data:
        for name in ("seed43", "seed44"):
            c[name] = data[name + "_original"]
    with np.load(source / "new_tree_raw_predictions.npz") as data:
        c["extra"] = data["d6_absolute_0.25"]
    with np.load(source / "new_head_predictions.npz") as data:
        heads = {key: data[key] for key in data.files}
    for name in ("seed43", "seed44"):
        c[name + "_zero"] = first["models"][name + "_original"]["source_zero"]
    extra_info = json.loads((source / "new_tree_inference_manifest.json").read_text())
    c["extra_zero"] = extra_info["models"]["d6_absolute_0.25"]["source_zero"]
    for name in recipe["heads"]["models"]:
        c[name + "_zero"] = heads_info["models"][name + "_production"]["source_zero"]
    # 32,568 labelled X1/X2 rows; source-zero rate is fixed by the training labels.
    source_zero = float((ROOT / "source_zero.txt").read_text().strip())
    output = blend(c, heads, recipe, source_zero)
    if len(output) != len(genes) or not np.isfinite(output).all():
        raise ValueError("Invalid prediction vector")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene_name": genes, "gex_predicted": output.astype(np.int64)}).to_csv(args.output, index=False)
    print(f"Saved {args.output} ({len(genes)} genes)")


if __name__ == "__main__":
    main()
