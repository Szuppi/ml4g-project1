# Chromatin-to-expression prediction

Code and saved predictions for the three submitted approaches: the ranking CNN, an eleven-model ensemble, and its fourteen-model extension. All use the seven chromatin assays and labelled X1/X2 cells.

## Repository layout

| Folder | Role |
| --- | --- |
| `cnn/` | First submitted ranking CNN and its training code |
| `gradient_boosting/` | Original relative-feature CatBoost component |
| `catboost_ordered/` | Ordered CatBoost with log-expression target, used in the larger ensemble |
| `ensemble_cnn_catboost/` | Earlier two-model CNN + CatBoost experiment; a separate ensemble, not one of the later submissions |
| `ensemble_multimodel/` | Submitted eleven-model ensemble: eight CNNs and three CatBoost models |
| `ensemble_multimodel_wide/` | Submitted fourteen-model extension: adds two wide-context CNNs and the Ordered CatBoost |

`prepare.py` and `common.py` hold the shared preparation and validation utilities for the original CNN and CatBoost. Each model or ensemble has its own short README.

## Rebuild the submitted files

With Python 3.11 and the packages in `requirements.txt`:

```bash
python ensemble_multimodel/build_submission.py
python ensemble_multimodel_wide/build_submission.py
```

The scripts check gene order, component lengths and finite values. They reproduce the bundled prediction CSVs exactly from saved constituent outputs. The full training and checkpoint inference of every later ensemble member was performed separately; these replay commands do not retrain all members from the course archives. The CNN validation workflow starts with `python prepare.py --data /path/to/extracted/course/data --output inputs` and `python cnn/train.py --inputs inputs`.

The ensemble weights and thresholds are fixed in each `recipe.json`. Source-cell transfer and joint validation used chromosome holdouts. The repository does not contain raw course data.
