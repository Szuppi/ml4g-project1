"""Replay the Ordered CatBoost component used by the fourteen-model blend."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent


def build(output):
    info = json.loads((ROOT.parent / 'ensemble_14_wide_context/components/inference_manifest.json').read_text())
    genes = info['genes']
    values = np.load(ROOT / 'production_absolutehalf.npy')
    if len(values) != len(genes) or len(set(genes)) != len(genes) or not np.isfinite(values).all():
        raise ValueError('Invalid model outputs or gene keys')
    # The blend uses an absolute cutoff of 1.0 on this component.
    score = (2 * rankdata(np.where(values <= 1.0, 0, values))).astype(np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'gene_name': genes, 'gex_predicted': score}).to_csv(output, index=False)
    print(f'Saved {output} ({len(genes)} genes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'submission.csv')
    build(parser.parse_args().output)
