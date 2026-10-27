# Equal-rank CNN + CatBoost

**Kaggle submission:** No. This earlier two-model ensemble was not the submitted 11- or 14-model blend.

**Approach and architecture.** This ensemble combines the [ranking CNN](../cnn/README.md) and [CatBoost model](../gradient_boosting/README.md) with equal weight. The CNN learns spatial patterns in promoter profiles; the trees learn nonlinear relationships between engineered spatial, relative-signal and gene-context features. Their complementary predictions are combined without fitting another model.

**Features and preprocessing.** The ensemble performs no additional genomic feature extraction. Both constituents use the same gene/TSS coordinates and seven chromatin assays, with strand-oriented windows and separate signal coverage. The CNN uses standardized, signed-log promoter profiles and coarse summaries with missingness indicators. CatBoost uses the larger 1,565-feature table directly, retaining NaN values. Each constituent computes its expression-rank targets and any fitted preprocessing from training rows only, then calibrates low scores using source validation. The linked approach READMEs describe these feature pipelines in more detail.

**Combination and evaluation.** `blend.py` joins the saved predictions by fold, cell line, gene and chromosome, and checks that gene sets and labels agree. Within each cell/fold, it ranks each constituent's calibrated predictions, assigning average ranks to ties, and combines the two rank vectors equally. Integer doubled ranks preserve ties without floating-point rounding artifacts; only the final ordering matters for Spearman. Weights remain fixed at 50/50 for both transfer directions and joint training. The bundled production submission similarly combines the two saved production prediction vectors by equal average ranks.

`results/` contains the ensemble predictions and fold-level Spearman scores; `submission.csv` is the saved production prediction file. See the [main README](../README.md) for reproduction commands, split definitions and the performance comparison.
