"""Build the 11- and 14-member submitted CSVs from fresh checkpoint predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ensemble_11_rank_blend'))
from blend import blend


def write_csv(genes, values, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'gene_name': genes, 'gex_predicted': values.astype(np.int64)}).to_csv(path, index=False)
    print(f'{path}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}')


def build(inputs, components, output):
    m = pd.read_csv(inputs / 'metadata.csv')
    source = m.cell_line.isin(['X1', 'X2'])
    genes = m.loc[~source, 'gene_name'].tolist()
    first = json.loads((ROOT / 'ensemble_11_rank_blend/components/new_inference_manifest.json').read_text())
    heads_info = json.loads((ROOT / 'ensemble_11_rank_blend/components/new_head_inference_manifest.json').read_text())
    extra_info = json.loads((ROOT / 'ensemble_11_rank_blend/components/new_tree_inference_manifest.json').read_text())
    last = json.loads((ROOT / 'ensemble_14_wide_context/components/inference_manifest.json').read_text())
    if genes != first['genes'] or genes != heads_info['genes'] or genes != last['genes']:
        raise ValueError('Test gene order differs from the selected submissions')
    with np.load(components) as z:
        a = {key: z[key] for key in z.files}
    if len(a) != 14 or any(len(x) != len(genes) or not np.isfinite(x).all() for x in a.values()):
        raise ValueError('Expected 14 complete constituent outputs')
    for name in ('seed43', 'seed44'):
        a[name + '_zero'] = first['models'][name + '_original']['source_zero']
    a['extra_zero'] = extra_info['models']['d6_absolute_0.25']['source_zero']
    for name in ('basic', 'combo'):
        a[name + '_zero'] = heads_info['models'][name + '_production']['source_zero']
    recipe = json.loads((ROOT / 'ensemble_11_rank_blend/recipe.json').read_text())
    heads = {name + '_production_quantile': a[name] for name in ('basic', 'combo')}
    zero = float((m.loc[source, 'gex'] == 0).mean())
    eleven = blend(a, heads, recipe, zero)
    write_csv(genes, eleven, output / 'ensemble_11.csv')
    score = eleven.astype(np.int64).astype(float)
    steps = json.loads((ROOT / 'ensemble_14_wide_context/recipe.json').read_text())['steps']
    for step in steps:
        name = step['model']
        raw = a[name]
        info = last['models'][name + '_production']
        cutoff = step['cut'] if info['target'] == 'log' else step['cut'] * info['source_zero']
        member = 2 * rankdata(np.where(raw <= cutoff, 0, raw))
        score = (100 - step['weight']) * 2 * rankdata(score) + step['weight'] * member
    write_csv(genes, score, output / 'ensemble_14.csv')
    write_csv(genes, score, output / 'submission.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.inputs, args.components, args.output)
