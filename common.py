"""Shared input validation, chromosome splits, calibration and reporting."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FOLDS = {'D1': ['chr3', 'chr11'], 'D2': ['chr17', 'chr21', 'chr22']}
RESERVED = ['chr1', 'chr12', 'chr18']
PROTOCOLS = ('x1_to_x2', 'x2_to_x1', 'joint')
ALPHAS = (0., .55, .65, .75, .85, .9, .95, 1.)


def load_inputs(folder):
    folder = Path(folder)
    m = pd.read_csv(folder / 'metadata.csv')
    if not m.cell_line.isin(['X1', 'X2']).all():
        raise ValueError('Only X1/X2 rows are supported')
    if m.duplicated(['cell_line', 'gene_name']).any():
        raise ValueError('Duplicate gene/cell pair')
    if not np.isfinite(m.gex).all() or (m.gex < 0).any():
        raise ValueError('Expected finite, nonnegative expression labels')
    with np.load(folder / 'features.npz') as z:
        data = {k: z[k] for k in z.files}
    for key, shape in {'bw56': (len(m), 56), 'coverage56': (len(m), 56),
                       'bed7': (len(m), 7), 'profile': (len(m), 7, 100),
                       'profile_coverage': (len(m), 7, 100)}.items():
        if data[key].shape != shape or np.isinf(data[key]).any():
            raise ValueError(f'Invalid array: {key}')
    return m, data


def masks(m, protocol, fold):
    source = ['X1', 'X2'] if protocol == 'joint' else [
        'X1' if protocol == 'x1_to_x2' else 'X2']
    target = source if protocol == 'joint' else [
        'X2' if protocol == 'x1_to_x2' else 'X1']
    chromosome = m.chr.isin(FOLDS[fold]).to_numpy()
    train = (~m.chr.isin(RESERVED + FOLDS[fold]) & m.cell_line.isin(source)).to_numpy()
    validation = chromosome & m.cell_line.isin(source).to_numpy()
    evaluate = chromosome & m.cell_line.isin(target).to_numpy()
    assert not (train & (validation | evaluate)).any()
    assert not set(m.loc[train, 'chr']) & set(m.loc[validation | evaluate, 'chr'])
    return train, validation, evaluate


def targets(m, mask):
    part = m.loc[mask]
    out = np.empty(len(part), np.float32)
    for cell in part.cell_line.unique():
        ix = part.cell_line.eq(cell).to_numpy()
        y = part.loc[ix, 'gex'].to_numpy()
        ref = np.sort(y)
        out[ix] = (np.searchsorted(ref, y, 'left') + np.searchsorted(ref, y, 'right')) / (2 * len(ref))
    return out


def rho(y, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    return float(spearmanr(y, prediction).statistic) if np.ptp(prediction) and np.ptp(y) else 0.


def calibrate(m, train, validation, prediction):
    zero = float((m.loc[train, 'gex'] == 0).mean())
    part = m.loc[validation]
    def objective(alpha):
        calibrated = np.where(prediction <= alpha * zero, 0., prediction)
        return min(rho(part.loc[part.cell_line.eq(c), 'gex'],
                       calibrated[part.cell_line.eq(c).to_numpy()]) for c in part.cell_line.unique())
    return max(ALPHAS, key=objective), zero


def save_predictions(path, m, mask, fold, continuous, prediction):
    frame = m.loc[mask, ['cell_line', 'gene_name', 'chr', 'gex']].copy()
    frame.insert(0, 'fold', fold)
    frame['continuous_prediction'] = continuous
    frame['prediction'] = prediction
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def summarize(folder):
    folder = Path(folder)
    records = []
    for protocol in PROTOCOLS:
        path = folder / (protocol + '.csv')
        if not path.exists():
            continue
        data = pd.read_csv(path)
        for (fold, cell), g in data.groupby(['fold', 'cell_line']):
            records.append(dict(protocol=protocol, fold=fold, cell_line=cell,
                                n=len(g), spearman=rho(g.gex, g.prediction)))
    table = pd.DataFrame(records)
    table.to_csv(folder / 'metrics.csv', index=False)
    return table.groupby('protocol').spearman.mean().to_dict()
