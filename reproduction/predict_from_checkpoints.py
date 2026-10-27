"""Produce all fourteen constituent predictions from supplied features and saved checkpoints.

Prepared inputs: metadata.csv, features.npz, coarse.npy, bins.npy, relative.npy,
rich.npy. Only labelled X1/X2 rows determine source distribution transforms.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from scipy.stats import rankdata
from models import AdaptiveCNN, CNN

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / 'checkpoints'
QUANTILES = np.linspace(0, 1, 1001)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def match(target, source):
    out = target.copy()
    observed = np.isfinite(target) & (target > 0)
    source = source[np.isfinite(source) & (source > 0)]
    if len(source) < 5 or observed.sum() < 5:
        return out
    tq = np.quantile(target[observed], QUANTILES)
    sq = np.quantile(source, QUANTILES)
    unique, positions = np.unique(tq, return_index=True)
    if len(unique) > 1:
        out[observed] = np.interp(target[observed], unique, sq[positions])
    return out


def match_columns(target, source, count):
    result = target.copy()
    for j in range(count):
        result[:, j] = match(target[:, j], source[:, j])
    return result


def load_model(name):
    path = CHECKPOINTS / name / 'model.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if name == 'cnn':
        model = CNN(saved['tab_dim'])
    else:
        variant = ('seed43' if name == 'seed43' else 'seed44' if name == 'seed44'
                   else name)
        model = AdaptiveCNN(saved['tab_dim'], variant)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, np.load(CHECKPOINTS / name / 'preprocessing.npz')


def neural(name, features, bins, source, target, norm):
    model, pre = load_model(name)
    raw = np.concatenate([features[k][target] for k in ('bw56', 'bed7', 'coverage56')], 1)
    missing = ~np.isfinite(raw)
    tab = signed_log(np.where(missing, 0, raw))
    profile_raw = features['profile'][target]
    coverage = features['profile_coverage'][target]
    missing_signal = ~np.isfinite(profile_raw)
    missing_coverage = ~np.isfinite(coverage)
    profile = signed_log(np.where(missing_signal, 0, profile_raw))
    if norm == 'quantile':
        source_raw = np.concatenate([features[k][source] for k in ('bw56', 'bed7', 'coverage56')], 1)
        source_tab = signed_log(source_raw)
        tab = match_columns(tab, source_tab, tab.shape[1])
        source_profile = signed_log(features['profile'][source])
        for j in range(7):
            profile[:, j] = match(profile[:, j], source_profile[:, j])
    tab = np.concatenate([(tab - pre['tab_mean']) / pre['tab_scale'], missing], 1).astype(np.float32)
    profile = np.concatenate([(profile - pre['profile_mean'][None, :, None]) /
                              pre['profile_scale'][None, :, None],
                              np.where(missing_coverage, 0, coverage), missing_signal,
                              missing_coverage], 1).astype(np.float32)
    inputs = [torch.from_numpy(tab), torch.from_numpy(profile)]
    if name in ('wide', 'wide_combo'):
        raw_wide = bins[target]
        missing_wide = ~np.isfinite(raw_wide)
        wide = signed_log(np.where(missing_wide, 0, raw_wide))
        if norm == 'quantile':
            source_wide = signed_log(np.where(np.isfinite(bins[source]), bins[source], 0))
            wide = match_columns(wide, source_wide, wide.shape[1])
        wide = (wide - pre['wide_mean']) / pre['wide_scale']
        inputs.append(torch.from_numpy(np.concatenate([wide, missing_wide], 1).astype(np.float32)))
    values = []
    with torch.no_grad():
        for start in range(0, len(tab), 128):
            values.append(model(*(x[start:start+128] for x in inputs)).numpy())
    result = np.concatenate(values).astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite {name} prediction')
    return result


def tree(name, matrix, source, target, end=0, strength=0):
    model = CatBoostRegressor()
    model.load_model(str(CHECKPOINTS / name / 'model.cbm'))
    x = matrix[target].copy()
    if end:
        x[:, :end] = (1-strength)*x[:, :end] + strength*match_columns(x, matrix[source], end)[:, :end]
    result = model.predict(x)
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite {name} prediction')
    return result


def run(inputs, output):
    m = pd.read_csv(inputs / 'metadata.csv')
    source = m.cell_line.isin(['X1', 'X2']).to_numpy()
    target = ~source
    if not np.isfinite(m.loc[source, 'gex']).all() or not m.loc[target, 'gex'].isna().all():
        raise ValueError('Expected labelled source rows and unlabelled query rows')
    genes = m.loc[target, 'gene_name'].tolist()
    expected = json.loads((ROOT.parent / 'ensemble_11_rank_blend/components/new_inference_manifest.json').read_text())['genes']
    if genes != expected or len(set(genes)) != len(genes):
        raise ValueError('Query gene order differs from bundled submission')
    with np.load(inputs / 'features.npz') as z:
        features = {k: z[k] for k in z.files}
    bins = np.load(inputs / 'bins.npy')
    coarse = np.load(inputs / 'coarse.npy')
    relative = np.load(inputs / 'relative.npy')
    rich = np.load(inputs / 'rich.npy')
    n = len(m)
    if bins.shape != (n, 1260) or coarse.shape != (n, 175) or relative.shape != (n, 130) or rich.shape != (n, 1738):
        raise ValueError('Feature block shapes differ from original training')
    x_relative = np.concatenate([coarse, bins, relative], 1).astype(np.float32)
    x_rich = rich.astype(np.float32)
    x_log = np.concatenate([rich, relative], 1).astype(np.float32)
    a = {}
    a['cnn'] = neural('cnn', features, bins, source, target, 'quantile')
    a['tree'] = tree('tree', x_relative, source, target, 175, .5)
    a['weighted'] = neural('weighted', features, bins, source, target, 'quantile')
    a['dilated'] = neural('dilated', features, bins, source, target, 'quantile')
    a['gain'] = neural('gain', features, bins, source, target, 'original')
    a['rich'] = tree('rich', x_rich, source, target, 175, 1.)
    a['seed43'] = neural('seed43', features, bins, source, target, 'original')
    a['seed44'] = neural('seed44', features, bins, source, target, 'original')
    a['extra'] = tree('extra', x_relative, source, target, 1435, .25)
    a['basic'] = neural('basic', features, bins, source, target, 'quantile')
    a['combo'] = neural('combo', features, bins, source, target, 'quantile')
    a['wide_combo'] = neural('wide_combo', features, bins, source, target, 'quantile')
    a['tree_log'] = tree('tree_log', x_log, source, target, 1435, .5)
    a['wide'] = neural('wide', features, bins, source, target, 'original')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'components.npz', **a)
    print('Saved 14 constituent outputs from checkpoints')
    return genes, a


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reproduction/results'))
    args = parser.parse_args()
    run(args.inputs, args.output)
