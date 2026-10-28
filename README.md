# Chromatin-to-expression prediction

Code, saved checkpoints and predictions for the DNA Expressionists team (Thorge Hamprecht and Krisztián Szuppinger), first project of Machine Learning for Genomics. The task is to predict gene expression from the seven chromatin assays, training on the labelled cells X1 and X2 and predicting the held-out cell X3. Submissions are scored by Spearman rank correlation.

## Submission package

`DNA_Expressionists_submission_package/` is the self-contained deliverable for the selected Kaggle submission: the fourteen-model ensemble, public Spearman **0.85305**. It carries `submission.csv`, the one-page report, and a `code/` tree with feature preparation, the training programs, the fourteen saved checkpoints, checkpoint inference and the fixed blend recipes. It rebuilds `submission.csv` from the original course data using nothing outside that folder; see `DNA_Expressionists_submission_package/code/README.md` for the model list and the exact commands.

The remaining top-level directories are the full working project: the individual submitted approaches, their per-component prediction caches, and the local model-selection and combination experiments.

## Repository layout

| Folder | Role |
| --- | --- |
| `DNA_Expressionists_submission_package/` | Self-contained bundle for the selected fourteen-model submission: `submission.csv`, report and `code/` to rebuild it |
| `cnn/` | First submitted ranking CNN and its training code |
| `gradient_boosting/` | Original relative-feature CatBoost component |
| `catboost_ordered/` | Ordered CatBoost with log-expression target, used in the larger ensemble |
| `ensemble_2_cnn_catboost/` | Earlier two-model CNN + CatBoost experiment; a separate ensemble, not one of the later submissions |
| `ensemble_11_rank_blend/` | Submitted eleven-model ensemble: eight CNNs and three CatBoost models |
| `ensemble_14_wide_context/` | Submitted fourteen-model extension: adds two wide-context CNNs and the Ordered CatBoost |
| `reproduction/` | Raw-data-to-submission path for the fourteen-model ensemble, with all production checkpoints |
| `sweep/` | Systematic sweep over feature sets, regressors and post-hoc orderings |
| `train_interactive/` | Guided notebook running the local X1/X2 cross-validation end to end |
| `combined/` | Blending the sweep models onto the fourteen-model ensemble, and a ridge-hurdle baseline |

`prepare.py`, `common.py` and `cnn/model.py` hold the shared feature preparation, validation utilities and CNN definition; `prepare_reference.py` adds reference-genome sequence features from the hg38 chromosome FASTA files (downloaded separately from UCSC, `hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/`). The sweep and combination studies import these shared modules, so the production code is kept at the top level as well as copied, self-contained, into the submission package.

## Submitted approaches and Kaggle scores

Six predictions were submitted. Their public scores are:

| Submission | Folder | Public Spearman |
| --- | --- | --- |
| Ranking CNN | `cnn/` | 0.83805 |
| Eleven-model ensemble | `ensemble_11_rank_blend/` | 0.85102 |
| Fourteen-model ensemble (selected) | `ensemble_14_wide_context/` | **0.85305** |
| Sweep blend `blend-15-cnn` | `combined/` | 0.84824 |
| Sweep blend `blend-16-cnn-catboost` | `combined/` | 0.84755 |
| Ridge-hurdle baseline | `combined/` | 0.81639 |

The fourteen-model ensemble is the selected submission; none of the three `combined/` blends exceeded it on the public leaderboard.

## Rebuild the submitted files

The clean, self-contained path is the submission package (`DNA_Expressionists_submission_package/code/README.md`). Within the working repository, with Python 3.11 and the packages in `requirements.txt`:

```bash
python ensemble_11_rank_blend/build_submission.py
python ensemble_14_wide_context/build_submission.py
```

These check gene order, component lengths and finite values, and reproduce the bundled prediction CSVs exactly from the saved constituent outputs. For a fresh run from the extracted course archives, including all constituent predictions, follow `reproduction/README.md`. The CNN validation workflow starts with `python prepare.py --data /path/to/extracted/course/data --output inputs` and `python cnn/train.py --inputs inputs`. The repository does not contain raw course data.

## Local cross-validation and combination study

Three directories hold the model-selection work on cell lines X1 and X2.

- `sweep/` runs a systematic sweep over two feature sets, three regressors (ridge, CatBoost, ranking CNN) and five post-hoc orderings under chromosome-holdout cross-validation. `train_interactive/` drives this sweep from a guided notebook and reports the scores; see their READMEs.
- `combined/` blends the best sweep models onto the fourteen-model ensemble (treated as one frozen unit) and builds a standalone ridge-hurdle baseline. Its public scores are listed above; `combined/README.md` records the study.
