# Ordered CatBoost component

**Kaggle submission:** No separate submission of this CatBoost component. It is a member of the submitted 14-model ensemble.

This is the additional CatBoost member selected for the wide-context ensemble. It uses 1,738 rich chromatin features and 130 relative/geometry features, with a log1p-expression target. The model has 1,200 depth-6 trees, learning rate 0.04, L2 leaf regularization 20, Ordered boosting and seed 51.

`train.py` defines the X1→X2, X2→X1, joint and full-source fits from aligned prepared feature arrays (`metadata.csv`, `rich.npy`, `relative.npy`). Run training in the private Kaggle environment. `production.cbm` is the saved full-source checkpoint. `build_submission.py` replays the saved production component after the fixed source-to-query input matching and rank cutoff used by the ensemble; it writes a standalone component CSV. The original relative-feature CatBoost is in `../gradient_boosting/`.
