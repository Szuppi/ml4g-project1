"""Equal average-tie ranks of the CNN and CatBoost predictions, within each cell/fold."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from common import PROTOCOLS, summarize


def blend(left, right):
    keys = ['fold', 'cell_line', 'gene_name', 'chr']
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError('Duplicate prediction keys')
    merged = left.merge(right, on=keys, suffixes=('_cnn', '_tree'), validate='one_to_one', how='outer', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('Component gene sets differ')
    np.testing.assert_allclose(merged.gex_cnn, merged.gex_tree, rtol=0, atol=1e-10)
    if not np.isfinite(merged[['prediction_cnn', 'prediction_tree']]).all().all():
        raise ValueError('Nonfinite component predictions')
    result = merged[keys].copy()
    result['gex'] = merged.gex_cnn
    result['prediction'] = np.nan
    for _, indices in merged.groupby(['fold', 'cell_line']).groups.items():
        # Integer doubled ranks avoid introducing artificial ties through rounding.
        a = np.rint(2 * rankdata(merged.loc[indices, 'prediction_cnn'])).astype(np.int64)
        b = np.rint(2 * rankdata(merged.loc[indices, 'prediction_tree'])).astype(np.int64)
        result.loc[indices, 'prediction'] = a + b
    return result


def run(cnn, trees, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for protocol in PROTOCOLS:
        a, b = Path(cnn) / (protocol + '.csv'), Path(trees) / (protocol + '.csv')
        if a.exists() and b.exists():
            blend(pd.read_csv(a), pd.read_csv(b)).to_csv(output / (protocol + '.csv'), index=False)
    print(summarize(output))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cnn', type=Path, default=Path(__file__).resolve().parents[1] / 'cnn/results')
    p.add_argument('--trees', type=Path, default=Path(__file__).resolve().parents[1] / 'gradient_boosting/results')
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    a = p.parse_args()
    run(a.cnn, a.trees, a.output)
