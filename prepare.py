"""Prepare X1/X2 features from the extracted course TSV, BigWig and BED files."""
from pathlib import Path
import argparse
import heapq
import numpy as np
import pandas as pd
import pyBigWig
MARKS = ('DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'H3K27me3', 'H3K9me3')
RINGS = ((-100000, -20000), (-20000, -5000), (5000, 20000), (20000, 100000))
def discover(input_root: Path, output_dir: Path) -> list[Path]:
    """Index extracted files, deliberately excluding outputs and ZIP archives."""
    excluded = output_dir.resolve()
    return sorted(p for p in input_root.rglob("*") if p.is_file()
                  and excluded not in p.resolve().parents
                  and p.suffix.lower() != ".zip")

def unique(files: list[Path], name: str) -> Path:
    hits = [p for p in files if p.name == name]
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one {name}; found {len(hits)}: {hits}")
    return hits[0]

def track_file(files: list[Path], mark: str, cell: str, kind: str) -> Path:
    suffixes = (".bw", ".bigwig") if kind == "bw" else (".bed", ".bed.gz", ".narrowpeak", ".narrowpeak.gz")
    hits = [p for p in files if mark.lower() in str(p).lower()
            and cell.lower() in p.name.lower() and p.name.lower().endswith(suffixes)]
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one {kind} track for {mark}/{cell}; found {len(hits)}: {hits}")
    return hits[0]

def regions_for_gene(row) -> list[tuple[int, int]]:
    tss = (int(row.TSS_start) + int(row.TSS_end)) // 2
    edges = [-5000, -2000, -500, 500, 2000, 5000]
    promoter = [(tss + a, tss + b) for a, b in zip(edges[:-1], edges[1:])]
    start, end = int(row.gene_start), int(row.gene_end)
    body_edges = [start + (end - start) * k // 3 for k in range(4)]
    body = list(zip(body_edges[:-1], body_edges[1:]))
    if row.strand == "-":
        promoter.reverse()
        body.reverse()
    return promoter + body

def interval_stats(bw, chrom: str, start: int, end: int) -> tuple[float, float]:
    """Exact covered-base mean and full requested interval coverage."""
    size = bw.chroms().get(chrom)
    a, b = max(0, start), min(end, size or 0)
    if size is None or end <= start or b <= a:
        return np.nan, 0.0
    mean = bw.stats(chrom, a, b, type="mean", exact=True)[0]
    coverage = bw.stats(chrom, a, b, type="coverage", exact=True)[0]
    return (np.nan if mean is None else float(mean),
            0.0 if coverage is None else float(coverage) * (b-a)/(end-start))

def profile_stats(bw, row) -> tuple[np.ndarray, np.ndarray]:
    tss = (int(row.TSS_start) + int(row.TSS_end)) // 2
    start, end = tss - 5000, tss + 5000
    chrom = row.chr
    size = bw.chroms().get(chrom)
    if size is not None and 0 <= start and end <= size:
        # One exact 10kb read avoids 200 repeated range queries per gene.
        # Dense values are bounded to this window, never the whole chromosome.
        raw = np.asarray(bw.values(chrom, start, end, numpy=True), dtype=np.float32).reshape(100, 100)
        finite = np.isfinite(raw)
        counts = finite.sum(axis=1)
        means = np.where(finite, raw, 0).sum(axis=1, dtype=np.float64) / np.maximum(counts, 1)
        means[counts == 0] = np.nan
        values = means.astype(np.float32)
        coverage = (counts / 100).astype(np.float32)
    else:
        pairs = [interval_stats(bw, chrom, a, a+100) for a in range(start, end, 100)]
        values, coverage = np.asarray(pairs, dtype=np.float32).T
    if row.strand == "-":
        values, coverage = values[::-1], coverage[::-1]
    return values, coverage

def load_peaks(path: Path) -> pd.DataFrame:
    try:
        peaks = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0,1,2,6], names=["chr", "start", "end", "signal"], compression="infer")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["chr", "start", "end", "signal"])
    if peaks.isna().any().any():
        raise ValueError(f"Missing BED fields: {path}")
    for col in ("start", "end", "signal"):
        peaks[col] = pd.to_numeric(peaks[col], errors="raise")
    if not np.isfinite(peaks[["start", "end", "signal"]]).all().all() or (peaks.start < 0).any() or (peaks.end <= peaks.start).any() or (peaks.signal < 0).any():
        raise ValueError(f"Invalid BED intervals/signals: {path}")
    if not np.equal(peaks[["start", "end"]], np.floor(peaks[["start", "end"]])).all().all():
        raise ValueError(f"Noninteger BED coordinates: {path}")
    return peaks

def peak_features(peaks: pd.DataFrame, genes: pd.DataFrame) -> np.ndarray:
    """Maximum overlapping signalValue using a monotone-window heap sweep.

    Each peak enters once. Expired peaks are lazily removed, retaining nested
    overlaps correctly. Empty peak files/chromosomes mean zero, not missing.
    """
    result = np.zeros(len(genes), dtype=np.float32)
    centers = ((genes.TSS_start.to_numpy() + genes.TSS_end.to_numpy()) // 2)
    for chrom in genes.chr.unique():
        idx = np.flatnonzero(genes.chr.to_numpy() == chrom)
        idx = idx[np.argsort(centers[idx], kind="stable")]
        chromosome_peaks = peaks[peaks.chr == chrom].sort_values("start", kind="stable")
        entries = list(chromosome_peaks[["start", "end", "signal"]].itertuples(index=False, name=None))
        heap: list[tuple[float, int]] = []
        cursor = 0
        for i in idx:
            start, end = int(centers[i])-5000, int(centers[i])+5000
            while cursor < len(entries) and entries[cursor][0] < end:
                _, peak_end, signal = entries[cursor]
                heapq.heappush(heap, (-float(signal), int(peak_end)))
                cursor += 1
            while heap and heap[0][1] <= start:
                heapq.heappop(heap)
            if heap:
                result[i] = -heap[0][0]
    return result
def rings_for_gene(row) -> list[tuple[int, int]]:
    tss = (int(row.TSS_start) + int(row.TSS_end)) // 2
    if row.strand == "+":
        return [(tss + a, tss + b) for a, b in RINGS]
    if row.strand == "-":
        return [(tss - b, tss - a) for a, b in RINGS]
    raise ValueError(f"Invalid strand: {row.strand!r}")

def context_for_gene(bw, row) -> tuple[np.ndarray, np.ndarray]:
    if row.chr not in bw.chroms():
        raise ValueError(f"Missing required chromosome: {row.chr}")
    stats = np.asarray([interval_stats(bw, row.chr, a, b) for a, b in rings_for_gene(row)], dtype=np.float32)
    return stats[:, 0], stats[:, 1]
def profile_window(bw, row, radius, bins=100):
    width = 2 * radius // bins
    if width * bins != 2 * radius:
        raise ValueError("Window must divide into equal integer bins")
    if row.strand not in {"+", "-"} or row.chr not in bw.chroms():
        raise ValueError("Invalid strand or missing chromosome")
    center = (int(row.TSS_start) + int(row.TSS_end)) // 2
    start, end = center - radius, center + radius
    if start >= 0 and end <= bw.chroms(row.chr):
        x = np.asarray(bw.values(row.chr, start, end, numpy=True)).reshape(bins, width)
        valid = np.isfinite(x)
        count = valid.sum(axis=1)
        total = np.where(valid, x, 0).sum(axis=1, dtype=np.float64)
        signal = np.divide(total, count, out=np.full(bins, np.nan), where=count > 0)
        coverage = count / width
    else:
        pairs = np.asarray([interval_stats(bw, row.chr, start+i*width, start+(i+1)*width)
                            for i in range(bins)])
        signal, coverage = pairs[:, 0], pairs[:, 1]
    if row.strand == "-":
        signal, coverage = signal[::-1], coverage[::-1]
    return signal.astype(np.float32), coverage.astype(np.float32)

def bin_features(signal, coverage, groups):
    n, channels, bins = signal.shape
    width = bins // groups
    finite = np.isfinite(signal)
    values = np.where(finite, signal, 0.).reshape(n, channels, groups, width)
    weights = coverage.reshape(n, channels, groups, width)
    denominator = (weights * finite.reshape(n, channels, groups, width)).sum(-1)
    means = (values * weights).sum(-1) / np.maximum(denominator, 1e-8)
    means[denominator == 0] = np.nan
    return np.concatenate([means.reshape(n, -1), weights.mean(-1).reshape(n, -1)], 1)


def relative_features(m, d):
    x = d['profile'].astype(np.float64)
    w = np.nan_to_num(d['profile_coverage'], nan=0.)
    mass = np.maximum(np.where(np.isfinite(x), x, 0.), 0.) * w
    local = mass.reshape(len(m), 7, 10, 10).sum(-1)
    local /= np.maximum(local.sum(-1, keepdims=True), 1e-8)
    bw = d['bw56'].reshape(len(m), 7, 8).astype(np.float64)
    finite = np.isfinite(bw)
    b = np.where(finite, bw, 0.)
    relative = b / np.maximum(np.abs(b).sum(-1, keepdims=True), 1e-8)
    relative[~finite] = np.nan
    length = (m.gene_end - m.gene_start).to_numpy(dtype=np.float64)
    tss = ((m.TSS_start + m.TSS_end) / 2).to_numpy()
    start, end = m.gene_start.to_numpy(), m.gene_end.to_numpy()
    forward = m.strand.eq('+').to_numpy()
    up = np.where(forward, tss - start, end - tss)
    down = np.where(forward, end - tss, tss - start)
    annotation = np.column_stack([np.log1p(length), np.sign(up) * np.log1p(np.abs(up)),
                                  np.sign(down) * np.log1p(np.abs(down)), up / length])
    return np.concatenate([local.reshape(len(m), -1), relative.reshape(len(m), -1), annotation], 1).astype(np.float32)


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
    m = pd.concat(parts, ignore_index=True)
    if m.duplicated(['cell_line', 'gene_name']).any() or not np.isfinite(m.gex).all() or (m.gex < 0).any():
        raise ValueError('Invalid labels or gene keys')
    if not m.strand.isin(['+', '-']).all() or (m.gene_end <= m.gene_start).any():
        raise ValueError('Invalid gene coordinates')
    n = len(m)
    d = {'bw56': np.full((n, 56), np.nan, np.float32), 'coverage56': np.zeros((n, 56), np.float32),
         'bed7': np.zeros((n, 7), np.float32), 'profile': np.full((n, 7, 100), np.nan, np.float32),
         'profile_coverage': np.zeros((n, 7, 100), np.float32)}
    context, context_coverage = np.empty((n, 28), np.float32), np.empty((n, 28), np.float32)
    # Only compact spatial summaries of the wider profiles are retained.
    wider = {radius: np.empty((n, 280), np.float32) for radius in [20000, 100000]}
    for cell in ['X1', 'X2']:
        indices = np.flatnonzero(m.cell_line.eq(cell))
        genes = m.iloc[indices]
        for j, mark in enumerate(MARKS):
            print(f'Preparing {cell}/{mark}', flush=True)
            d['bed7'][indices, j] = peak_features(load_peaks(track_file(files, mark, cell, 'bed')), genes)
            with pyBigWig.open(str(track_file(files, mark, cell, 'bw'))) as bw:
                for i, row in zip(indices, genes.itertuples(index=False)):
                    pairs = np.asarray([interval_stats(bw, row.chr, a, b) for a, b in regions_for_gene(row)], np.float32)
                    d['bw56'][i, j * 8:j * 8 + 8], d['coverage56'][i, j * 8:j * 8 + 8] = pairs.T
                    d['profile'][i, j], d['profile_coverage'][i, j] = profile_stats(bw, row)
                    context[i, j * 4:j * 4 + 4], context_coverage[i, j * 4:j * 4 + 4] = context_for_gene(bw, row)
                    for radius, bins in [(20000, 400), (100000, 800)]:
                        signal, coverage = profile_window(bw, row, radius, bins)
                        # Preserve the same per-assay means-then-coverage column ordering.
                        compact = bin_features(signal[None, None], coverage[None, None], 20)[0]
                        wider[radius][i, j * 20:j * 20 + 20] = compact[:20]
                        wider[radius][i, 140 + j * 20:140 + j * 20 + 20] = compact[20:]
    coarse = np.concatenate([d['bw56'], d['coverage56'], d['bed7'], context, context_coverage], 1)
    bins = np.concatenate([bin_features(d['profile'], d['profile_coverage'], 50), wider[20000], wider[100000]], 1)
    tree = np.concatenate([coarse, bins, relative_features(m, d)], 1).astype(np.float32)
    assert tree.shape == (n, 1565)
    m.to_csv(output / 'metadata.csv', index=False)
    np.savez_compressed(output / 'features.npz', **d)
    np.save(output / 'tree_features.npy', tree)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('inputs'))
    a = p.parse_args()
    prepare(a.data, a.output)
