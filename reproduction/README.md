# Reproduce the submitted ensembles

This directory contains the training programs, the 14 production checkpoints, feature preparation, checkpoint inference and the final blend recipes. The course data are **not** stored in Git. The submitted CSVs can be rebuilt from the bundled checkpoints without retraining; the training programs allow every constituent fit to be repeated.

## Inputs and commands

Extract the original course archives into one directory, for example `/path/to/course-data`. It must contain `X1_train_info.tsv`, `X1_train_y.tsv`, `X1_val_info.tsv`, `X1_val_y.tsv` and the matching X2 files, `X3_test_info.tsv`, plus seven BigWig and BED tracks per cell line. The preparation code searches recursively, so the course archive's subdirectories can be retained. Test expression labels are neither needed nor read.

Install `requirements.txt`, then run from the repository root:

```bash
python reproduction/prepare_all.py --data /path/to/course-data --output /path/to/prepared
python reproduction/predict_from_checkpoints.py --inputs /path/to/prepared --output /path/to/predictions
python reproduction/build_submissions.py --inputs /path/to/prepared --components /path/to/predictions/components.npz --output /path/to/submissions
```

The two results are `/path/to/submissions/ensemble_11.csv` and `ensemble_14.csv`. `prepare_all.py` writes `metadata.csv`, `features.npz`, `coarse.npy`, `bins.npy`, `relative.npy` and `rich.npy` in the prepared directory. The checkpoint files are in `checkpoints/`; their SHA-256 digests are in `checkpoint_hashes.json`. The fixed blend parameters are in the two ensemble `recipe.json` files. `build_submissions.py` checks the expected test-gene order and all 14 output lengths.

This complete course-data-to-CSV path was run on 27 October 2026 with the bundled production checkpoints. Both outputs matched the archived submitted files byte for byte: 11 members `8ab47a3136c09c703e1a99df1133a27c03247b3c38712bc74f27d125e0c7801a`; 14 members `d93e9b38071cd8cc1f32b48951fd736ef756034093415baf4fef58e891095230` (SHA-256). This verifies the saved-checkpoint reconstruction; separately retraining every member may depend on the pinned Kaggle runtime.

## Repeat the training

`training/` contains the original Kaggle production runners. They expect an uploaded **private Kaggle input dataset** with the prepared feature files, Python wheelhouse and manifests. Create its directory outside this repository with:

```bash
python reproduction/stage_kaggle_inputs.py --inputs /path/to/prepared --wheels /path/to/wheels --output /path/to/kaggle-input
```

The wheel directory must contain the pinned wheels required by the runners (`numpy 2.4.6`, `pandas 3.0.6`, `scipy 1.17.1`, `scikit-learn 1.9.1`, `torch 2.14.1+cpu`, `catboost 1.2.10`). Upload `/path/to/kaggle-input` as a Kaggle dataset, attach it to a CPU notebook, and run the relevant file in `training/`. The scripts save their production model under `/kaggle/working/results/` (the early `cnn.py` and `rich.py` use `/kaggle/working/predictions/`). Replace the matching checkpoint in `checkpoints/` with the newly trained production model and preprocessing file before rerunning inference. Seeds and hyperparameters are in each runner. Training the exact same weights requires the pinned runtime and deterministic execution; the bundled weights are the production artifacts used for the submitted blends.

| Checkpoint | Training program and production result |
| --- | --- |
| `cnn` | `training/cnn.py` → ranking CNN, `production/swa` |
| `tree` | `training/tree.py` → relative-feature CatBoost, `production` |
| `weighted` | `training/weighted.py` → `weight1/production` |
| `dilated`, `gain` | `training/variants.py` → matching variant's `production` |
| `rich` | `training/rich.py` → rich-feature CatBoost |
| `seed43`, `seed44` | `training/seed43.py`, `training/seed44.py` → `combo/production` |
| `extra` | `training/extra.py` → depth-six relative-feature CatBoost, `production` |
| `basic`, `combo` | `training/basic.py`, `training/combo.py` → matching two-head `production` |
| `wide`, `wide_combo` | `training/wide.py`, `training/wide_combo.py` → matching variant's `production` |
| `tree_log` | `training/tree_log.py` → Ordered CatBoost on log-expression, `production` |

The 11-member blend uses the first eleven rows through `basic` and `combo`; the 14-member blend adds `wide_combo`, `tree_log` and `wide`. `training/neural.py` is the exact shared model module used by the original CNN and rich-feature runners.
