# Guided walkthrough notebook

`walkthrough.py` (jupytext percent source of truth) and its paired `walkthrough.ipynb` run the
project's local X1/X2 cross-validation end to end: assay and reference features, a sweep over all
model recipes, selection of the best model of each type, and ensembles of those. The sweep itself
lives in `../sweep/`; the notebook drives it and reports the results.

## Environment

A conda env `ml4g_project1` (Python 3.11). It follows the pinned `../requirements.txt` with one
deviation: PyTorch is `2.11.0+cu128` (the pinned `2.14.1` has no CUDA-12.8 wheel, which the
RTX 5060 / sm_120 needs). The torch minor version does not change the model recipe.

```bash
conda create -n ml4g_project1 python=3.11 -y
conda activate ml4g_project1
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.4.6 pandas==3.0.6 scipy==1.17.1 scikit-learn==1.9.1 \
            catboost==1.2.10 pyBigWig==0.3.26
pip install ipykernel jupytext nbconvert matplotlib
python -m ipykernel install --user --name ml4g_project1 --display-name "Python (ml4g_project1)"
```

## Running it

Keep `.py` and `.ipynb` in sync with `jupytext --sync walkthrough.py`, then run the notebook top
to bottom with the `Python (ml4g_project1)` kernel, or headless:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 jupyter nbconvert --to notebook --execute --inplace \
  walkthrough.ipynb --ExecutePreprocessor.kernel_name=ml4g_project1 \
  --ExecutePreprocessor.timeout=10800
```

## Configuration

The setup cell exposes:

- `SELECTION_METRIC` in {`transfer_mean`, `transfer_worst`, `joint`}: which score picks the
  "best" model of each type (and sorts the tables).
- `TRAIN_SAVE_MODELS`: `False` loads a prior run's scores (and its cached predictions if present);
  `True` trains every recipe once and caches its model, local-test predictions and scores under
  `./outputs/sweep/`, retraining only what is missing.
- `FORCE_PREPARE`: rebuild the cached features in `../inputs/`.
- `DEVICE`: `cuda` if available, else `cpu` (CPU is slower but supported).

Feature extraction (`prepare.py`, ~20-40 min over ~16 GB of tracks) is cached to `../inputs/`.
Model predictions and scores are written under `./outputs/`; the `.npz` predictions and saved
models are gitignored, the score `.csv` files are kept.

## Scope

This is the local X1/X2 cross-validation. The X3 Kaggle `submission.csv` is a pre-computed
production artifact and is not regenerated here (`prepare.py` processes only X1/X2).
