"""Fit the Ordered CatBoost source model from prepared feature arrays on Kaggle.

Expected metadata columns: cell_line, chr, gex. The rich and relative arrays
must align row-for-row with metadata and have 1738 and 130 columns respectively.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr

EXCLUDED = {'chr1', 'chr3', 'chr11', 'chr12', 'chr18'}
EVAL = {'chr3', 'chr11'}


def fit(metadata, rich, relative, output):
    m = pd.read_csv(metadata)
    r = np.load(rich)
    q = np.load(relative)
    if r.shape != (len(m), 1738) or q.shape != (len(m), 130):
        raise ValueError('Features must align with metadata: 1738 rich + 130 relative columns')
    source = m.cell_line.isin(['X1', 'X2']).to_numpy()
    if not np.isfinite(m.loc[source, 'gex'].to_numpy()).all():
        raise ValueError('Nonfinite source labels')
    x = np.concatenate([r, q], axis=1).astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    for protocol in ('x1_to_x2', 'x2_to_x1', 'joint', 'production'):
        cells = ['X1'] if protocol == 'x1_to_x2' else ['X2'] if protocol == 'x2_to_x1' else ['X1', 'X2']
        train = source & m.cell_line.isin(cells).to_numpy()
        if protocol != 'production':
            train &= ~m.chr.isin(EXCLUDED).to_numpy()
        model = CatBoostRegressor(iterations=1200, depth=6, learning_rate=0.04,
            l2_leaf_reg=20, loss_function='RMSE', boosting_type='Ordered',
            random_seed=51, thread_count=2, verbose=False,
            allow_writing_files=False, border_count=128, nan_mode='Min',
            bootstrap_type='Bernoulli', subsample=0.8)
        model.fit(x[train], np.log1p(m.loc[train, 'gex'].to_numpy()).astype(np.float32))
        dest = output / protocol
        dest.mkdir(exist_ok=True)
        model.save_model(str(dest / 'model.cbm'))
        if protocol != 'production':
            val = source & m.chr.isin(EVAL).to_numpy()
            pred = model.predict(x[val])
            frame = m.loc[val, ['cell_line', 'gene_name', 'chr', 'gex']].copy()
            frame['prediction'] = pred
            frame.to_csv(dest / 'validation.csv', index=False)
            print(protocol, {cell: spearmanr(group.gex, group.prediction).statistic
                             for cell, group in frame.groupby('cell_line')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--rich', type=Path, required=True)
    parser.add_argument('--relative', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('catboost_ordered/results'))
    args = parser.parse_args()
    fit(args.metadata, args.rich, args.relative, args.output)
