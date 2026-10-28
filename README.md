# Chromatin-to-expression prediction

Code and saved predictions for the three submitted approaches: the ranking CNN, an eleven-model ensemble, and its fourteen-model extension. All use the seven chromatin assays and labelled X1/X2 cells.

## Repository layout

| Folder | Role |
| --- | --- |
| `cnn/` | First submitted ranking CNN and its training code |
| `gradient_boosting/` | Original relative-feature CatBoost component |
| `catboost_ordered/` | Ordered CatBoost with log-expression target, used in the larger ensemble |
| `ensemble_2_cnn_catboost/` | Earlier two-model CNN + CatBoost experiment; a separate ensemble, not one of the later submissions |
| `ensemble_11_rank_blend/` | Submitted eleven-model ensemble: eight CNNs and three CatBoost models |
| `ensemble_14_wide_context/` | Submitted fourteen-model extension: adds two wide-context CNNs and the Ordered CatBoost |
| `sweep/` | Systematic sweep over feature sets, regressors and post-hoc orderings |
| `train_interactive/` | Guided notebook running the local X1/X2 cross-validation end to end |
| `combined/` | Blending the sweep models onto the fourteen-model ensemble, and a ridge-hurdle baseline |

The public Kaggle scores for the submitted files are **0.83805** (CNN), **0.85102** (11 models) and **0.85305** (14 models). The submission status and score are stated in each approach README. The other three folders are components or an earlier experiment, with no separate Kaggle submission.

`prepare.py` and `common.py` hold the original shared preparation and validation utilities; `prepare_reference.py` adds reference-genome sequence features, extracted from the hg38 chromosome FASTA files (downloaded separately from UCSC, `hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/`). [The reproduction guide](reproduction/README.md) gives the complete course-data-to-submission path, training runners and all 14 production checkpoints. Each model or ensemble has its own short README.

## Rebuild the submitted files

With Python 3.11 and the packages in `requirements.txt`:

```bash
python ensemble_11_rank_blend/build_submission.py
python ensemble_14_wide_context/build_submission.py
```

The scripts check gene order, component lengths and finite values. They reproduce the bundled prediction CSVs exactly from saved constituent outputs. For a fresh run from the extracted course archives, including all constituent predictions, follow `reproduction/README.md`. The CNN validation workflow starts with `python prepare.py --data /path/to/extracted/course/data --output inputs` and `python cnn/train.py --inputs inputs`.

The ensemble weights and thresholds are fixed in each `recipe.json`. Source-cell transfer and joint validation used chromosome holdouts. The repository does not contain raw course data.

## Local cross-validation and combination study

Three further directories hold the local model-selection work on cell lines X1 and X2.

- `sweep/` runs a systematic sweep over two feature sets, three regressors (ridge, CatBoost, ranking CNN) and five post-hoc orderings under chromosome-holdout cross-validation. `train_interactive/` drives this sweep from a guided notebook and reports the scores; see their READMEs.
- `combined/` blends the best sweep models onto the fourteen-model ensemble (treated as one frozen unit) and builds a standalone ridge-hurdle baseline. Its submissions and their public Kaggle scores are **0.84824** (`blend-15-cnn`), **0.84755** (`blend-16-cnn-catboost`) and **0.81639** (`baseline-ridge-hurdle`). None exceeds the fourteen-model ensemble's 0.85305, which stays the selected submission; `combined/README.md` records the study.

