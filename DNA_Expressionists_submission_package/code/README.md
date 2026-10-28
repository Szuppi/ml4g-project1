# Code and model reproduction

## Models

The 14-model ensemble contains ten CNNs and four CatBoost regressors. Each model is trained separately on labelled X1/X2 data. The first eleven outputs form a fixed weighted rank blend; the final three outputs are added in sequence. The blend does not fit another model to the eleven-model result or to X3 labels.

| Model | Type | Role | Training program |
| --- | --- | --- | --- |
| `cnn` | CNN | Promoter ranking model | `reproduction/training/cnn.py` |
| `tree` | CatBoost | Relative chromatin features | `reproduction/training/tree.py` |
| `weighted` | CNN | Source-weighted variant | `reproduction/training/weighted.py` |
| `dilated` | CNN | Dilated convolution variant | `reproduction/training/variants.py` |
| `gain` | CNN | Signal-gain augmentation variant | `reproduction/training/variants.py` |
| `rich` | CatBoost | Rich regional and peak features | `reproduction/training/rich.py` |
| `seed43`, `seed44` | CNN | Additional seeded models | `reproduction/training/seed43.py`, `reproduction/training/seed44.py` |
| `extra` | CatBoost | Depth-six relative-feature model | `reproduction/training/extra.py` |
| `basic`, `combo` | CNN | Activity and positive-rank heads | `reproduction/training/basic.py`, `reproduction/training/combo.py` |
| `wide_combo`, `wide` | CNN | Wider genomic context | `reproduction/training/wide_combo.py`, `reproduction/training/wide.py` |
| `tree_log` | CatBoost | Ordered model on log expression | `reproduction/training/tree_log.py` |

`ensemble_11_rank_blend/recipe.json` defines the first eleven members' fixed cutoffs and weights. `ensemble_14_wide_context/recipe.json` adds `wide_combo`, `tree_log`, and `wide` to that rank result. `ensemble_11_rank_blend/blend.py` implements the eleven-model combination; `reproduction/build_submissions.py` applies both recipes.

## Rebuild the selected submission

Install `code/requirements.txt`. Extract the original course archives under one data directory. It must contain the X1/X2 training and validation information and expression TSVs, `X3_test_info.tsv`, and the seven BED and BigWig tracks per cell line. The scripts search subdirectories. X3 expression labels are not used.

From the extracted ZIP directory, run:

```bash
python code/reproduction/prepare_all.py --data /path/to/course-data --output /path/to/prepared
python code/reproduction/predict_from_checkpoints.py --inputs /path/to/prepared --output /path/to/predictions
python code/reproduction/build_submissions.py --inputs /path/to/prepared --components /path/to/predictions/components.npz --output /path/to/submissions
```

The final file is `/path/to/submissions/submission.csv`. The script also writes `ensemble_11.csv` and `ensemble_14.csv`. The supplied `submission.csv` at the ZIP root has SHA-256 `d93e9b38071cd8cc1f32b48951fd736ef756034093415baf4fef58e891095230`. The 11-model output has SHA-256 `8ab47a3136c09c703e1a99df1133a27c03247b3c38712bc74f27d125e0c7801a`.

`prepare_all.py` writes `metadata.csv`, `features.npz`, `coarse.npy`, `bins.npy`, `relative.npy`, and `rich.npy`. `predict_from_checkpoints.py` loads the model files in `reproduction/checkpoints/` and writes all fourteen model outputs to `components.npz`. `reproduction/checkpoint_hashes.json` records the checkpoint digests.

## Train the component models

The programs in `reproduction/training/` contain the production seeds and hyperparameters. They expect a private Kaggle input dataset of prepared features, manifests, and a wheelhouse. Create it outside the ZIP with:

```bash
python code/reproduction/stage_kaggle_inputs.py --inputs /path/to/prepared --wheels /path/to/wheels --output /path/to/kaggle-input
```

The wheelhouse must contain the pinned packages used by the training programs: NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, scikit-learn 1.9.1, PyTorch 2.14.1+cpu, and CatBoost 1.2.10. Attach the staged dataset to a Kaggle CPU notebook and run the relevant training file. The output checkpoints can replace the corresponding files in `reproduction/checkpoints/`; CNNs also need their matching preprocessing `.npz` file. The included checkpoints are the weights used for the selected submission.
