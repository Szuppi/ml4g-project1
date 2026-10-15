"""CatBoost on spatial bins, chromatin summaries and relative signal features."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from common import FOLDS, PROTOCOLS, load_inputs, masks, targets, calibrate, save_predictions, summarize


def run(inputs, output, protocols=PROTOCOLS):
    output = Path(output)
    m, _ = load_inputs(inputs)
    x = np.load(Path(inputs) / 'tree_features.npy')
    if x.shape != (len(m), 1565) or np.isinf(x).any():
        raise ValueError('Expected 1565 aligned tree features')
    for protocol in protocols:
        frames = []
        for fold in FOLDS:
            train, validation, evaluate = masks(m, protocol, fold)
            model = CatBoostRegressor(iterations=960, depth=5, learning_rate=.05,
                l2_leaf_reg=10, loss_function='RMSE', random_seed=42, thread_count=2,
                verbose=False, allow_writing_files=False, border_count=64, nan_mode='Min')
            model.fit(x[train], targets(m, train))
            alpha, zero = calibrate(m, train, validation, model.predict(x[validation]))
            raw = model.predict(x[evaluate])
            calibrated = np.where(raw <= alpha * zero, 0., raw)
            dest = output / 'models' / protocol / fold
            dest.mkdir(parents=True, exist_ok=True)
            model.save_model(str(dest / 'model.cbm'))
            (dest / 'recipe.json').write_text(json.dumps(dict(alpha=alpha, zero_fraction=zero,
                seed=42, iterations=960, train_rows=int(train.sum())), indent=2))
            frames.append(save_predictions(dest / 'validation.csv', m, evaluate, fold, raw, calibrated))
            print(json.dumps(dict(protocol=protocol, fold=fold, completed=True)), flush=True)
        pd.concat(frames).to_csv(output / (protocol + '.csv'), index=False)
        print(json.dumps(summarize(output)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    p.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=list(PROTOCOLS))
    a = p.parse_args()
    run(a.inputs, a.output, a.protocols)
