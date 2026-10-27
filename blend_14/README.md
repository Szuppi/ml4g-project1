# Fourteen-model blend

This recipe adds three model predictions to the eleven-model blend: two wide-context CNNs and an Ordered CatBoost on log expression. It applies the three fixed steps in `recipe.json` with average-tie ranks. Run `python blend_11/build_submission.py` followed by `python blend_14/build_submission.py` from the repo root. The resulting `submission.csv` matches the saved candidate exactly.

`source_scores.csv` gives the D1 chromosome-held-out X1→X2, X2→X1 and joint X1/X2 Spearman checks. The replay uses saved constituent outputs; it does not retrain the fourteen models from raw tracks.
