# Ridge-hurdle baseline (X3)

A standalone single-model baseline for the X3 test cell: a two-part (hurdle) model in the style of Dong et al. (2012). A ridge regression predicts expression and a CatBoost classifier predicts activity; genes the classifier scores below the gate threshold are set to zero, and the rest keep their ridge order. This is the best-scoring post-hoc ordering for ridge on the joint cross-validation protocol (joint mean Spearman 0.8291); on its own it scores 0.81639 on the Kaggle public leaderboard. It is submitted on its own rather than blended into the 14-model ensemble.

## Inputs and commands

Two prepared directories must be present; neither is stored in Git.

`combined/work/inputs_x3/` holds the features for X1, X2 and X3. The baseline reads `metadata.csv`, `tree_features.npy`, `seq_features.npy` and `seq_schema.json`. `tree_features.npy` is the concatenation of the coarse, bins and relative assay arrays; the sequence block is built from the reference genome, including chr1 for the X3 genes. `combined/work/ensemble_x3_sub/ensemble_14.csv` holds the 14-model ensemble's X3 ranks and is read only to fix the test-gene order.

Run from the repository root:

```bash
python combined/build_x3_ridge.py
```

This writes `combined/submissions/baseline-ridge-hurdle.csv`: 1,984 genes, columns `gene_name,gex_predicted`, in the fixed test-gene order. The score is `2 * rankdata(gated prediction)` cast to integer, so only the ordering is meaningful.

## Parameters

Both model parts are refit on all X1+X2 rows, so neither has a held-out fold on which to select its gate threshold or L2 penalty. Each parameter is fixed to the median of the two cross-validation folds' selections:

| parameter | D1 | D2 | used |
| --- | --- | --- | --- |
| gate threshold | 0.45 | 0.55 | 0.5 |
| ridge L2 alpha | 5000 | 1000 | 3000 |

Recompute the fold selections with:

```bash
python combined/derive_baseline_params.py
```

## Verification

Built on 28 October 2026: the classifier gate keeps 827 of the 1,984 genes and zeroes the rest; the submission is Spearman 0.947 against the 14-model ensemble and moves by less than 0.0003 Spearman across ridge alpha in {1000, 3000, 5000}.
