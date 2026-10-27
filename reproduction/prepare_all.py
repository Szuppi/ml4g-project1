"""Prepare all model inputs from the extracted, original course files.

No expression label is read for the test cell line. This produces the feature
layout consumed by the archived production checkpoints and training scripts.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import (MARKS, bin_features, context_for_gene, discover, interval_stats,
                     load_peaks, peak_features, profile_stats, profile_window,
                     regions_for_gene, relative_features, track_file, unique)


def profile_features(signal, coverage, scale):
    n, channels, bins = signal.shape
    width = 2 if scale == 'local' else bins // 20
    groups = bins // width
    small = np.empty((n, channels * groups * 2), np.float32)
    shape = np.empty((n, channels * 9 + 3), np.float32)
    centers = np.linspace(-1, 1, bins, dtype=np.float32)
    for a in range(0, n, 128):
        b = min(n, a + 128)
        raw, cov = signal[a:b], coverage[a:b]
        valid = np.isfinite(raw)
        values = np.where(valid, raw, 0.)
        count = valid.sum(-1)
        mean = values.sum(-1) / np.maximum(count, 1)
        variance = (np.where(valid, (values - mean[..., None]) ** 2, 0.).sum(-1) /
                    np.maximum(count, 1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            maximum = np.nanmax(raw, axis=-1)
            q90 = np.nanquantile(raw, .9, axis=-1)
        mass = np.maximum(values, 0.) * cov
        total = mass.sum(-1)
        centroid = (mass * centers).sum(-1) / np.maximum(total, 1e-8)
        upstream = mass[..., :bins // 2].sum(-1)
        downstream = mass[..., bins // 2:].sum(-1)
        asymmetry = (downstream - upstream) / np.maximum(total, 1e-8)
        near = mass[..., bins * 2 // 5:bins * 3 // 5].sum(-1) / np.maximum(total, 1e-8)
        stats = np.stack([mean, np.sqrt(variance), maximum, q90, centroid, asymmetry,
                          near, cov.mean(-1), (values > mean[..., None]).mean(-1)], -1)
        stats[count == 0] = np.nan
        pairs = []
        for j, k in [(0, 3), (0, 1), (1, 3)]:
            joint_valid = valid[:, j] & valid[:, k]
            pairs.append((np.minimum(values[:, j], values[:, k]) * joint_valid).sum(-1) /
                         np.maximum(joint_valid.sum(-1), 1))
        shape[a:b] = np.column_stack([stats.reshape(b - a, -1), *pairs])
        v = values.reshape(b - a, channels, groups, width)
        finite = valid.reshape(b - a, channels, groups, width)
        weights = cov.reshape(b - a, channels, groups, width)
        denom = (weights * finite).sum(-1)
        means = (v * weights).sum(-1) / np.maximum(denom, 1e-8)
        means[denom == 0] = np.nan
        small[a:b] = np.concatenate([means.reshape(b - a, -1),
                                     weights.mean(-1).reshape(b - a, -1)], 1)
    return small, shape


def peak_geometry(peaks, chrom, tss, strand):
    peaks = peaks[peaks.chr.eq(chrom)].sort_values('start')
    starts, ends, strength = (peaks[k].to_numpy() for k in ('start', 'end', 'signal'))
    widths = ends - starts
    max_width = int(widths.max()) if len(widths) else 0
    result = np.empty((len(tss), 15), np.float32)
    for i, center in enumerate(tss):
        for j, radius in enumerate([1000, 5000, 20000]):
            lo = np.searchsorted(starts, center - radius - max_width, 'left')
            hi = np.searchsorted(starts, center + radius, 'left')
            active = ends[lo:hi] > center - radius
            aa, bb, ss = starts[lo:hi][active], ends[lo:hi][active], strength[lo:hi][active]
            if len(aa):
                midpoint = (aa + bb) / 2
                k = int(np.argmin(np.abs(midpoint - center)))
                distance = (midpoint[k] - center) * (-1 if strand[i] == '-' else 1)
                result[i, j * 5:(j + 1) * 5] = [len(aa), np.mean(bb - aa),
                    np.sum(np.minimum(bb, center + radius) - np.maximum(aa, center - radius)),
                    ss.sum(), distance]
            else:
                result[i, j * 5:(j + 1) * 5] = [0., 0., 0., 0., np.nan]
    return result


def prepare(root, output):
    output.mkdir(parents=True, exist_ok=True)
    files = discover(root, output)
    parts = []
    for cell in ['X1', 'X2']:
        for split in ['train', 'val']:
            info = pd.read_csv(unique(files, f'{cell}_{split}_info.tsv'), sep='\t')
            labels = pd.read_csv(unique(files, f'{cell}_{split}_y.tsv'), sep='\t')
            if set(info.gene_name) != set(labels.gene_name):
                raise ValueError('Info and label gene sets differ')
            part = info.merge(labels[['gene_name', 'gex']], on='gene_name', validate='one_to_one', sort=False)
            part.insert(0, 'split', split)
            part.insert(0, 'cell_line', cell)
            parts.append(part)
    query = pd.read_csv(unique(files, 'X3_test_info.tsv'), sep='\t')
    query['gex'] = np.nan
    query.insert(0, 'split', 'test')
    query.insert(0, 'cell_line', 'X3')
    parts.append(query)
    m = pd.concat(parts, ignore_index=True)
    if m.duplicated(['cell_line', 'gene_name']).any():
        raise ValueError('Duplicate gene key')
    n = len(m)
    d = {'bw56': np.full((n, 56), np.nan, np.float32),
         'coverage56': np.zeros((n, 56), np.float32),
         'bed7': np.zeros((n, 7), np.float32),
         'profile': np.full((n, 7, 100), np.nan, np.float32),
         'profile_coverage': np.zeros((n, 7, 100), np.float32)}
    context = np.empty((n, 28), np.float32)
    context_coverage = np.empty((n, 28), np.float32)
    fine = {r: (np.empty((n, 7, b), np.float32),
                np.empty((n, 7, b), np.float32))
            for r, b in [(20000, 400), (100000, 800)]}
    geometry = np.empty((n, 105), np.float32)
    for cell in ['X1', 'X2', 'X3']:
        indices = np.flatnonzero(m.cell_line.eq(cell))
        genes = m.iloc[indices]
        for j, mark in enumerate(MARKS):
            print(f'Preparing {cell}/{mark}', flush=True)
            peaks = load_peaks(track_file(files, mark, cell, 'bed'))
            d['bed7'][indices, j] = peak_features(peaks, genes)
            for chrom, group in genes.groupby('chr'):
                tss = ((group.TSS_start + group.TSS_end) // 2).to_numpy()
                geometry[group.index, j * 15:(j + 1) * 15] = peak_geometry(
                    peaks, chrom, tss, group.strand.to_numpy())
            with pyBigWig.open(str(track_file(files, mark, cell, 'bw'))) as bw:
                for i, row in zip(indices, genes.itertuples(index=False)):
                    pairs = np.asarray([interval_stats(bw, row.chr, a, b)
                                        for a, b in regions_for_gene(row)], np.float32)
                    d['bw56'][i, j*8:j*8+8], d['coverage56'][i, j*8:j*8+8] = pairs.T
                    d['profile'][i, j], d['profile_coverage'][i, j] = profile_stats(bw, row)
                    stats = np.asarray(context_for_gene(bw, row), np.float32)
                    context[i, j*4:j*4+4], context_coverage[i, j*4:j*4+4] = stats
                    for radius, bins in [(20000, 400), (100000, 800)]:
                        signal, coverage = profile_window(bw, row, radius, bins)
                        fine[radius][0][i, j] = signal
                        fine[radius][1][i, j] = coverage
    coarse = np.concatenate([d['bw56'], d['coverage56'], d['bed7'], context,
                             context_coverage], 1).astype(np.float32)
    b0, s0 = profile_features(d['profile'], d['profile_coverage'], 'local')
    b1, s1 = profile_features(*fine[20000], 'fine20')
    b2, s2 = profile_features(*fine[100000], 'fine100')
    bins = np.concatenate([b0, b1, b2], 1).astype(np.float32)
    rich = np.concatenate([coarse, np.concatenate([s0, s1, s2], 1), bins,
                           geometry], 1).astype(np.float32)
    assert coarse.shape == (n, 175) and bins.shape == (n, 1260)
    assert rich.shape == (n, 1738)
    m.to_csv(output / 'metadata.csv', index=False)
    np.savez_compressed(output / 'features.npz', **d)
    np.save(output / 'coarse.npy', coarse)
    np.save(output / 'bins.npy', bins)
    np.save(output / 'relative.npy', relative_features(m, d))
    np.save(output / 'rich.npy', rich)
    print(f'Saved {n} rows in {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.output)
