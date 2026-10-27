# CatBoost

**Kaggle submission:** No separate submission of this CatBoost component.

**Approach and architecture.** A CatBoost regressor predicts expression ranks from 1,565 engineered chromatin features per gene. Gradient boosting adds successive decision trees to correct prediction errors. The fixed recipe uses 960 depth-5 trees, learning rate 0.05, L2 regularization 10 and seed 42. No neural network or gene-specific model is fitted.

**Feature extraction.** `prepare.py` uses the TSS midpoint and strand-oriented windows for all seven assays. The feature vector combines 175 coarse features (promoter/gene-body signal and coverage, BED peak maxima and wider context rings), 1,260 spatial features (signal means and coverage in 50 bins across ±5 kb and 20 bins each across ±20/100 kb), and 130 relative/geometry features. The latter describe each assay's local signal distribution, relative promoter/body signal, gene length and TSS position within the gene. Spatial means weight covered bases so missing regions do not act like measured zeros.

**Preprocessing and evaluation.** Trees use the extracted values directly, without neural-style standardization or log transformation of the complete feature table. Missing signal stays NaN and uses CatBoost's native missing-value handling; selected geometry features are log-transformed during extraction. Targets are empirical expression ranks computed separately within each training cell. A cutoff chosen on source validation maps low predictions to zero. X1 → X2 and X2 → X1 use only the named source cell for fitting and calibration; joint training uses both. Evaluation chromosomes are excluded from fitting.

`results/` contains predictions and fold-level Spearman scores; `submission.csv` is the saved production prediction file. See the [main README](../README.md) for preparation, commands and exact splits.
