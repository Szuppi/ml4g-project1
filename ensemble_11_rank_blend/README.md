# Eleven-model ensemble

**Kaggle submission:** Yes — `blend-11-models.csv`, public score **0.85102**.

The submitted ensemble combines eight CNNs and three CatBoost models. It uses average-tie ranks, source-derived zero thresholds and fixed staged weights. The two activity/positive-rank CNNs form a separate group that contributes 30% to the final ranking.

Run `python ensemble_11_rank_blend/build_submission.py` from the repo root. `components/` contains the saved constituent predictions and minimal ordering/calibration metadata; `recipe.json` defines the exact combination. The resulting `submission.csv` matches the submitted candidate byte for byte. This replay does not retrain the constituent models from raw tracks.
