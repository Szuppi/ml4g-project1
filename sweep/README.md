# sweep

The systematic model sweep over feature sets, regressors and post-hoc orderings. It builds on the
shared base (`common.py`, `prepare.py`, `cnn/model.py`); the only change to that base is the CNN's
final pooling layer, switched to a deterministic variant so GPU training is reproducible.

## Modules

- `features.py`: the two feature sets (`assays_only`, `assays-ref`) and the active-gene fitting
  masks.
- `scoring.py`: the five post-hoc orderings (`raw`, `tie`, `cal`, `gate`, `soft`), their
  validation-set parameter selection, and `score_families`.
- `models.py`: the activity classifier and the three regressor folds (`ridge_fold`,
  `catboost_fold`, `cnn_fold`), with a `regressor(...)` dispatch.
- `run.py`: `run_sweep(...)` trains every recipe (3 regressors x {all, active} x
  {assays_only, assays-ref}), scores all five orderings, and optionally caches the models,
  per-recipe local-test predictions and score tables.

## Run

    python -m sweep.run --inputs inputs --out train_interactive/outputs/sweep --device cuda

This writes `summary.csv`, `experiments.csv` and `classifier_metrics.csv` under `--out`, plus the
per-recipe models and predictions. The saved models and the `.npz` predictions are gitignored;
the score `.csv` files are kept.
