"""Prepare reference-genome features (GC/CpG composition and CpG-island annotation).

These features describe the reference hg38 sequence around each gene's TSS. They are
cell-line independent and label independent, so the same values apply to X1, X2 and X3.
Output rows align one-to-one with ``inputs/metadata.csv`` (the same order used by
``prepare.py``), so models can concatenate ``seq_features.npy`` to the assay features.

Resources (hg38 autosomes and UCSC cpgIslandExt) are downloaded on demand into a
gitignored directory and pinned by a manifest. Nothing large is committed.

    python prepare_reference.py --download --reference data/reference
    python prepare_reference.py --inputs inputs --reference data/reference
    python prepare_reference.py --selftest
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38"
# Oriented sequence windows relative to the TSS, as [a, b) in transcription direction.
WINDOWS = {"core": (-200, 100), "proximal": (-500, 500), "broad": (-1500, 1500)}
# Byte codes for an uppercase DNA sequence read as ASCII.
A, C, G, T, N = 65, 67, 71, 84, 78


# --------------------------------------------------------------------------- #
# Resource download and manifest
# --------------------------------------------------------------------------- #
def chromosomes(meta: pd.DataFrame) -> list[str]:
    """Autosomes actually present in the metadata, natural-sorted."""
    return sorted(meta.chr.unique(), key=lambda c: int(c[3:]))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def download_resources(reference: Path, chroms: list[str]) -> dict:
    """Download CpG-island table and per-chromosome FASTA; write a pinned manifest."""
    reference = Path(reference)
    files = {"cpgIslandExt.txt.gz": f"{UCSC}/database/cpgIslandExt.txt.gz"}
    for chrom in chroms:
        files[f"{chrom}.fa.gz"] = f"{UCSC}/chromosomes/{chrom}.fa.gz"
    manifest = {"assembly": "hg38", "source": UCSC, "files": {}}
    for name, url in files.items():
        path = reference / name
        if not path.exists():
            _fetch(url, path)
        manifest["files"][name] = {"url": url, "bytes": path.stat().st_size,
                                   "sha256": _sha256(path)}
    (reference / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_chromosome(path: Path) -> np.ndarray:
    """Read one gzipped FASTA chromosome into an uppercase ASCII byte array."""
    with gzip.open(path, "rt") as handle:
        handle.readline()  # drop the ">chrN" header
        sequence = handle.read().replace("\n", "").upper()
    return np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def tss_anchor(row) -> int:
    return (int(row.TSS_start) + int(row.TSS_end)) // 2


def oriented_window(tss: int, a: int, b: int, strand: str) -> tuple[int, int]:
    """Genomic [start, end) for the oriented interval [a, b); mirror on the minus strand."""
    return (tss + a, tss + b) if strand == "+" else (tss - b, tss - a)


# --------------------------------------------------------------------------- #
# GC / CpG composition (reverse-complement invariant, so no reverse complement needed)
# --------------------------------------------------------------------------- #
def gc_cpg(segment: np.ndarray, requested: int) -> dict:
    """GC fraction, CpG count/density, CpG observed/expected and unambiguous fraction."""
    unambiguous = (segment == A) | (segment == C) | (segment == G) | (segment == T)
    length = int(unambiguous.sum())
    pairs = int((unambiguous[:-1] & unambiguous[1:]).sum()) if segment.size > 1 else 0
    n_c, n_g = int((segment == C).sum()), int((segment == G).sum())
    n_cg = int(((segment[:-1] == C) & (segment[1:] == G)).sum()) if segment.size > 1 else 0
    gc = (n_c + n_g) / length if length else np.nan
    density = n_cg / pairs if pairs else np.nan
    expected = (n_c / length) * (n_g / length) if length and n_c and n_g else np.nan
    return {"gc_frac": gc, "cpg_count": float(n_cg), "cpg_density": density,
            "cpg_oe": density / expected if pairs and expected else np.nan,
            "unambig_frac": length / requested if requested else np.nan}


# --------------------------------------------------------------------------- #
# CpG islands (merged intervals per chromosome)
# --------------------------------------------------------------------------- #
def load_islands(path: Path, chroms: list[str]) -> dict:
    """Merged, sorted CpG-island intervals per chromosome from the UCSC table."""
    table = pd.read_csv(path, sep="\t", header=None, usecols=[1, 2, 3],
                        names=["chr", "start", "end"], compression="gzip")
    islands = {}
    for chrom in chroms:
        spans = table[table.chr == chrom].sort_values("start")
        merged: list[list[int]] = []
        for start, end in zip(spans.start, spans.end):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], int(end))
            else:
                merged.append([int(start), int(end)])
        islands[chrom] = np.array(merged, dtype=np.int64) if merged else np.empty((0, 2), np.int64)
    return islands


def covered_fraction(spans: np.ndarray, start: int, end: int) -> float:
    """Fraction of [start, end) covered by the union of (already merged) island spans."""
    if end <= start or not spans.size:
        return 0.0
    overlap = np.minimum(spans[:, 1], end) - np.maximum(spans[:, 0], start)
    return float(np.clip(overlap, 0, None).sum()) / (end - start)


def nearest_distance(spans: np.ndarray, point: int) -> float:
    """Distance in bp from a point to the nearest island; 0 inside, NaN if none exist."""
    if not spans.size:
        return np.nan
    left = point - spans[:, 1]            # >0 when the point is to the right of a span
    right = spans[:, 0] - point           # >0 when the point is to the left of a span
    return float(np.clip(np.maximum(left, right), 0, None).min())


def island_features(tss: int, strand: str, spans: np.ndarray, present: bool) -> dict:
    out = {"tss_in_island": float(covered_fraction(spans, tss, tss + 1) > 0),
           "island_dist": nearest_distance(spans, tss) if present else np.nan,
           "island_on_chr": float(present)}
    for name, (a, b) in WINDOWS.items():
        start, end = oriented_window(tss, a, b, strand)
        out[f"island_frac_{name}"] = covered_fraction(spans, start, end)
    return out


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #
def feature_schema() -> list[dict]:
    """Ordered (name, kind) list; kind drives model-specific preprocessing downstream."""
    schema = []
    for name in WINDOWS:
        schema += [{"name": f"gc_frac_{name}", "kind": "fraction"},
                   {"name": f"cpg_count_{name}", "kind": "count"},
                   {"name": f"cpg_density_{name}", "kind": "fraction"},
                   {"name": f"cpg_oe_{name}", "kind": "ratio"},
                   {"name": f"unambig_frac_{name}", "kind": "fraction"}]
    schema += [{"name": "tss_in_island", "kind": "indicator"},
               {"name": "island_dist", "kind": "distance"},
               {"name": "island_on_chr", "kind": "indicator"}]
    for name in WINDOWS:
        schema.append({"name": f"island_frac_{name}", "kind": "fraction"})
    return schema


def build_features(meta: pd.DataFrame, reference: Path) -> np.ndarray:
    chroms = chromosomes(meta)
    names = [col["name"] for col in feature_schema()]
    islands = load_islands(reference / "cpgIslandExt.txt.gz", chroms)
    out = np.full((len(meta), len(names)), np.nan, dtype=np.float32)
    for chrom in chroms:
        sequence = read_chromosome(reference / f"{chrom}.fa.gz")
        spans = islands[chrom]
        present = bool(spans.size)
        for i in np.flatnonzero(meta.chr.to_numpy() == chrom):
            row = meta.iloc[i]
            tss = tss_anchor(row)
            values = {}
            for name, (a, b) in WINDOWS.items():
                start, end = oriented_window(tss, a, b, row.strand)
                segment = sequence[max(0, start):min(end, sequence.size)]
                values.update({f"{k}_{name}": v for k, v in gc_cpg(segment, end - start).items()})
            values.update(island_features(tss, row.strand, spans, present))
            out[i] = [values[name] for name in names]
        print(f"  {chrom}: {int((meta.chr == chrom).sum())} rows", flush=True)
    return out


def prepare_reference(inputs: Path, reference: Path) -> None:
    inputs, reference = Path(inputs), Path(reference)
    meta = pd.read_csv(inputs / "metadata.csv")
    download_resources(reference, chromosomes(meta))
    features = build_features(meta, reference)
    assert features.shape == (len(meta), len(feature_schema()))
    if np.isinf(features).any():
        raise ValueError("Nonfinite (inf) sequence features")
    np.save(inputs / "seq_features.npy", features)
    (inputs / "seq_schema.json").write_text(json.dumps(feature_schema(), indent=2))
    print(f"wrote {inputs / 'seq_features.npy'} {features.shape}", flush=True)


# --------------------------------------------------------------------------- #
# Self-test on hand-checked synthetic sequences
# --------------------------------------------------------------------------- #
def selftest() -> None:
    def seq(s: str) -> np.ndarray:
        return np.frombuffer(s.encode(), np.uint8)
    # Oriented windows mirror on the minus strand.
    assert oriented_window(1000, -200, 100, "+") == (800, 1100)
    assert oriented_window(1000, -200, 100, "-") == (900, 1200)
    # GC/CpG on "ACGTCG": C,G counts=2,2 of 6; adjacent CG at positions 1-2 and 4-5 -> 2.
    stats = gc_cpg(seq("ACGTCG"), 6)
    assert stats["cpg_count"] == 2 and abs(stats["gc_frac"] - 4 / 6) < 1e-9
    assert abs(stats["unambig_frac"] - 1.0) < 1e-9
    # Observed/expected: density=2/5; expected=(2/6)*(2/6); ratio = 0.4 / (1/9).
    assert abs(stats["cpg_oe"] - (2 / 5) / ((2 / 6) * (2 / 6))) < 1e-9
    # Undefined O/E when there is no C or no G: returns NaN, not a misleading 0.
    assert np.isnan(gc_cpg(seq("AAAA"), 4)["cpg_oe"])
    # Ambiguous bases are excluded from the unambiguous length.
    assert gc_cpg(seq("ACGTN"), 5)["unambig_frac"] == 4 / 5
    # Island geometry: merge, coverage, nearest distance (0 inside, NaN if none).
    spans = np.array([[100, 200], [150, 300]], np.int64)  # overlap -> merged [100, 300)
    merged = np.array([[100, 300]], np.int64)
    assert covered_fraction(merged, 100, 200) == 1.0
    assert covered_fraction(merged, 50, 150) == 0.5
    assert nearest_distance(merged, 250) == 0.0 and nearest_distance(merged, 400) == 100.0
    assert np.isnan(nearest_distance(np.empty((0, 2), np.int64), 250))
    # Overlapping spans double-count without merging, which is why load_islands merges first.
    assert covered_fraction(spans, 100, 300) > 1.0
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, default=Path("inputs"))
    p.add_argument("--reference", type=Path, default=Path("data/reference"))
    p.add_argument("--download", action="store_true", help="only download/pin resources")
    p.add_argument("--selftest", action="store_true", help="run correctness checks and exit")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.download:
        meta = pd.read_csv(a.inputs / "metadata.csv")
        download_resources(a.reference, chromosomes(meta))
    else:
        prepare_reference(a.inputs, a.reference)
