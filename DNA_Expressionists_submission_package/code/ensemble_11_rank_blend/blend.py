"""Fixed rank combination for the eleven base models."""
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

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

