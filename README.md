# Chromatin-to-expression prediction

Three submitted prediction recipes using seven chromatin assays and the labelled X1/X2 cells. The repository includes the submitted CSVs, fixed blend code, and chromosome-held-out source scores. The blend scripts regenerate the exact CSV files from saved constituent predictions; see the reproducibility scope below.

| Submitted approach | X1 → X2 | X2 → X1 | Joint X1/X2 |
| --- | ---: | ---: | ---: |
| Ranking CNN (`cnn/`) | 0.809342 | 0.830220 | 0.845718 |
| Eleven-model blend (`blend_11/`) | 0.832914 | 0.831150 | 0.854925 |
| Fourteen-model blend (`blend_14/`) | 0.831785 | 0.831030 | 0.854837 |

The CNN numbers average two chromosome folds (D1 and D2). The later blend numbers use the D1 chromosome holdout and are therefore **not directly comparable** with the CNN's two-fold averages. Joint scores average X1 and X2 cell scores. The per-cell values and sample counts are in each `source_scores.csv` or the CNN's `results/metrics.csv`. These were development checks used to choose models and weights, not an untouched confirmation set.

## Approaches

- **Ranking CNN:** a 64-channel residual network processes strand-oriented ±5 kb assay profiles and coarse features. It predicts within-cell expression rank with MSE plus a small pairwise ranking loss. `cnn/README.md` describes the training recipe.
- **Eleven-model blend:** eight CNNs and three CatBoost models. Five initial models combine calibrated ranks in fixed stages; a richer-feature tree, two seeded CNNs and an additional tree enter at small weights. Two activity/positive-rank CNNs form another group, combined at 30% with the base group. The exact cutoffs, weights and tie handling are in `blend_11/recipe.json` and `build_submission.py`.
- **Fourteen-model blend:** starts from the eleven-model prediction and adds a wide-context CNN, a wide-context source-weighted CNN and an Ordered CatBoost trained on log expression. Three fixed rank-blend steps are in `blend_14/recipe.json` and `build_submission.py`.

## Run

Use Python 3.11 with NumPy, pandas and SciPy from `requirements.txt`.

```bash
python blend_11/build_submission.py
python blend_14/build_submission.py
```

Both commands check gene order, component lengths and finite values. The bundled component arrays are model outputs; running the scripts reproduces `blend_11/submission.csv` and `blend_14/submission.csv` byte for byte. The original model fits and checkpoint inference were performed separately. This bundle does **not** retrain all blend members from the course archives. For the first approach, `python prepare.py --data /path/to/extracted/course/data --output inputs` followed by `python cnn/train.py --inputs inputs` runs the source validation workflow. The saved CNN submission is the original full-source prediction.

The transfer checks fit on one source cell and evaluate the other on held-out chromosomes; the joint checks fit both source cells. Source validation excludes chr1/3/11/12/18 from fitting and evaluates chr3/11. Fold-specific normalization and rank targets use training rows. All blend weights are fixed across the three protocols. No raw course data or model checkpoints are committed.
