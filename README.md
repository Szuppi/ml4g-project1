# Chromatin-to-expression prediction

Three compact approaches for gene-expression ranking from seven chromatin assays. Each approach has its own folder, X1/X2 validation predictions, performance table, and ready-made `submission.csv`.

All results complete.

| Approach | X1 → X2 | X2 → X1 | Joint X1/X2 |
| --- | ---: | ---: | ---: |
| Ranking CNN | 0.809342 | 0.830220 | 0.845718 |
| CatBoost | 0.803047 | 0.825734 | 0.841608 |
| Equal-rank CNN + CatBoost | 0.807420 | 0.829167 | 0.847839 |

Scores are mean per-cell Spearman correlations across two chromosome folds, seed 42. Joint scores average X1 and X2 separately; transfer scores evaluate only the target cell. Fold-level scores and sample counts are in each `results/metrics.csv`.

## Approaches

- `cnn/`: 64-channel residual CNN on ±5 kb chromatin profiles, plus coarse features. Eight epochs, MSE on within-cell empirical expression ranks plus 0.01 pairwise ranking loss; average weights from epochs 6–8. AdamW, learning rate 0.001, batch size 256, dropout 0.3.
- `gradient_boosting/`: CatBoost, 960 trees, depth 5, learning rate 0.05, L2 10. 1,565 features: coarse signal/coverage/peaks, spatial bins at ±5/20/100 kb, relative signal and gene geometry.
- `ensemble/`: equal average-tie ranks of the two component predictions, computed separately within each cell and fold. No additional fitting.

Shared preparation and evaluation helpers stay at the repository root to avoid duplicated code. No raw data or model checkpoints are committed.

## Reproduce X1/X2 results

Use Python 3.11 and install `requirements.txt` in an isolated environment. Obtain the course data separately and extract the archives, retaining assay directory names. Preparation reads only `X1_train`, `X1_val`, `X2_train`, and `X2_val` info/label TSVs, plus their BigWig and BED tracks.

```bash
python prepare.py --data /path/to/extracted/course/data --output inputs
python cnn/train.py --inputs inputs
python gradient_boosting/train.py --inputs inputs
python ensemble/blend.py
```

`prepare.py` writes aligned `metadata.csv`, `features.npz`, and `tree_features.npy`. Feature windows use zero-based half-open coordinates, TSS interval midpoint, and strand orientation. Uncovered signal stays NaN; coverage is stored separately. Wider-profile summaries use covered-base weights. The preparation code does not fit any transform.

D1 evaluates chr3/11; D2 evaluates chr17/21/22. Fitting excludes the evaluated chromosomes and chr1/12/18. The remaining chromosomes from both labelled source splits are available for fitting. Transfer fits only the named source cell. All normalization and rank targets use training rows only. Zero-score calibration maximizes the worst source-cell validation correlation on a fixed grid; target-cell expression never selects it in transfer. The checkpoint recipe is fixed before the transfer rerun.

These are development folds used during model exploration, not independent confirmation. Joint predictions are reused from the saved audited runs. Transfer predictions are fresh fits of these same recipes; they must not be substituted with results from other model variants.

## Files to use

`results/x1_to_x2.csv`, `results/x2_to_x1.csv`, and `results/joint.csv` contain gene keys, raw labels, and predictions on the evaluated chromosomes. `metrics.csv` contains their fold-level scores. CNN and tree CSVs also retain continuous predictions before source calibration. The ensemble retains summed doubled ranks, which have the same ordering as equal average ranks.

Each approach's `submission.csv` has exactly `gene_name,gex_predicted`, 1,984 unique genes, and finite scores in the required order. Saved production constituents were fitted on all 32,568 labelled X1/X2 rows. Production zero cutoffs use the source-derived median factors: CNN 0.95 and CatBoost 0.80. The ensemble combines their equal ranks. Submission files are distinct from chromosome-validation predictions and have not been submitted by this repository workflow. Scores are relative ordering values, not calibrated expression units or fold changes.

Regenerating the commands above reproduces the X1/X2 evaluation workflow; it does not overwrite the bundled production submission files. Checkpoints and training normalization are saved under `results/models/` locally and ignored by Git.

## Team access

This repository is private. The owner can invite teammates through **Settings → Collaborators → Add people**. Invited teammates need a GitHub account and must accept the invitation.
