# Combining the sweep models with the 14-model ensemble

This directory tests whether the sweep models (`../sweep/`) improve the submitted 14-model
ensemble when blended onto it, and builds a standalone ridge-hurdle baseline for reference.
The 14-model ensemble is treated as a single frozen unit: the sweep models are added to it with
the ensemble's own staged rank-blend recipe. On the Kaggle public leaderboard none of the blends
exceeded the ensemble's 0.85305, so the ensemble remains the selected submission; this directory
documents the attempt and the baseline.

## Approach

The 14-model ensemble's checkpoints predict the hidden X3 cell and cannot be scored locally. To
compare against it under local validation, its members are retrained under the joint
cross-validation fold (both cell lines, chromosomes held out) and its blend recipe is replayed to
give an out-of-fold ensemble prediction. The sweep models are then added one staged rank step at
a time by greedy forward selection, each addition chosen with the previous ones in place. All
blending works on cached out-of-fold predictions, so the sweep reruns in seconds. The two best
blends and the baseline are then rebuilt on the full X1+X2 data and predicted for X3.

The comparison is **D1-only**: 9 of the 14 ensemble members validate on a single fixed split that
equals the joint-D1 fold, so only that fold is common to both sides. The local gain on D1 did not
carry to the public leaderboard, which is the main finding here.

## Submissions and scores

| submission | recipe | local joint-D1 | Kaggle public |
| --- | --- | --- | --- |
| `blend-15-cnn.csv` | 14-model ensemble + cnn.gate.w40 | 0.8563 | 0.84824 |
| `blend-16-cnn-catboost.csv` | + catboost.cal.w5 | 0.8567 | 0.84755 |
| `baseline-ridge-hurdle.csv` | standalone ridge + activity classifier | 0.8291¹ | 0.81639 |

Reference: the 14-model ensemble scores 0.8549 on the same local D1 fold and 0.85305 on Kaggle.
¹ The baseline is scored on both folds (joint mean); the blends are D1-only because the ensemble
reproduces only on D1. The baseline is described in [`README_baseline.md`](README_baseline.md).

## Files

- `sweep_oof.py`, `ensemble_blend.py`, `rich_oof.py`, `localize.py`: reproduce the out-of-fold
  predictions. `localize.py` adapts the ensemble training runners (`ensemble_runners/`) to run
  locally; `ensemble_blend.py` replays the ensemble recipe on the reproduced members.
- `blendlib.py`: scoring and rank-blend primitives.
- `cross_product.py`, `greedy.py`, `combined_sweep.py`: the combination search. `combined_sweep.py`
  produces the ranked `results/combined_toplist.csv` the walkthrough renders.
- `build_x3.py`, `build_x3_ridge.py`, `derive_baseline_params.py`: build the X3 submissions in
  `submissions/`.
- `oof_sweep/`, `oof_ensemble/`, `results/`: cached predictions and score tables.

## Reproduce

Run from the repository root, after the sweep has been run (`../sweep/`) and the reference-genome
sequence features prepared:

```bash
python combined/sweep_oof.py        # sweep-model out-of-fold predictions
python combined/ensemble_blend.py   # 14-model ensemble out-of-fold prediction
python combined/combined_sweep.py   # ranked combination toplist
python combined/build_x3.py         # blend-15 and blend-16 X3 submissions
python combined/build_x3_ridge.py   # ridge-hurdle baseline X3 submission
```

The working directories under `combined/work/` (staged inputs, retrained members, X3 features) are
gitignored and rebuilt by `localize.py` and the reproduction preparation scripts.
