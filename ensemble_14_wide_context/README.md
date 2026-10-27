# Wide-context ensemble

**Kaggle submission:** Yes — `blend-14-models.csv`, public score **0.85305**.

This submitted fourteen-model extension starts with the eleven-model ensemble and adds a wide-context CNN, a source-weighted wide-context CNN and the Ordered CatBoost in `../catboost_ordered/`. Three fixed average-rank steps are specified in `recipe.json`.

Run `python ensemble_11_rank_blend/build_submission.py` and then `python ensemble_14_wide_context/build_submission.py` from the repo root. The resulting `submission.csv` matches the saved candidate byte for byte. Both scripts replay saved constituent outputs; they do not retrain all models from raw tracks.
