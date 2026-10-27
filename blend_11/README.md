# Eleven-model blend

The fixed recipe combines eight CNNs and three CatBoost models, using average-tie ranks, source-derived zero thresholds and staged weights. Run `python build_submission.py` from this folder or `python blend_11/build_submission.py` from the repo root. `components/` holds the saved constituent outputs and the minimal metadata needed for the replay. The script writes the bundled `submission.csv` exactly.

`source_scores.csv` gives the D1 chromosome-held-out X1→X2, X2→X1 and joint X1/X2 Spearman checks. These development scores informed selection. The bundled script replays predictions; it does not retrain the eleven constituent models from raw tracks.
