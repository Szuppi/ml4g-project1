"""Build the fixed fourteen-model rank ensemble from the eleven-model output."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent


def build(base_path, output_path):
    base = pd.read_csv(base_path)
    if list(base.columns) != ['gene_name', 'gex_predicted'] or base.gene_name.duplicated().any():
        raise ValueError('Invalid base submission')
    info = json.loads((ROOT / 'components/inference_manifest.json').read_text())
    if base.gene_name.tolist() != info['genes']:
        raise ValueError('Gene order differs')
    steps = json.loads((ROOT / 'recipe.json').read_text())['steps']
    with np.load(ROOT / 'components/new_predictions.npz') as data:
        score = base.gex_predicted.to_numpy(dtype=float)
        for step in steps:
            name = step['model'] + '_production'
            raw = data[name + '_' + step['normalization']]
            if len(raw) != len(score) or not np.isfinite(raw).all():
                raise ValueError('Invalid component output')
            cutoff = step['cut'] if info['models'][name]['target'] == 'log' else step['cut'] * info['models'][name]['source_zero']
            member = 2 * rankdata(np.where(raw <= cutoff, 0, raw))
            weight = step['weight']
            score = (100 - weight) * 2 * rankdata(score) + weight * member
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'gene_name': base.gene_name, 'gex_predicted': score.astype(np.int64)}).to_csv(output_path, index=False)
    print(f'Saved {output_path} ({len(score)} genes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT.parent / 'blend_11/submission.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'submission.csv')
    args = parser.parse_args()
    build(args.base, args.output)
