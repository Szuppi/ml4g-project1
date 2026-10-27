"""Stage prepared features and wheelhouse as one Kaggle input dataset (outside Git)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(directory, name):
    files = sorted(x for x in directory.iterdir() if x.is_file() and x.name != name)
    (directory / name).write_text(json.dumps({p.name: digest(p) for p in files}, indent=2) + '\n')


def stage(inputs, wheel_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / 'prediction'
    training = output / 'training'
    upgrade = output / 'upgrade'
    rich = output / 'rich'
    wheels = output / 'wheels'
    for folder in (prediction, training, upgrade, rich, wheels):
        folder.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(inputs / 'metadata.csv')
    labelled = m.cell_line.isin(['X1', 'X2'])
    if not m.loc[labelled, 'gex'].notna().all() or not m.loc[~labelled, 'gex'].isna().all():
        raise ValueError('Invalid labelled and test rows')
    m.loc[labelled].to_csv(training / 'metadata.csv', index=False)
    shutil.copy2(ROOT / 'cnn/model.py', training / 'model.py')
    for name in ('features.npz', 'metadata.csv', 'coarse.npy', 'bins.npy'):
        shutil.copy2(inputs / name, prediction / name)
    with np.load(inputs / 'features.npz') as features:
        np.savez_compressed(training / 'features.npz',
                            **{name: features[name][labelled.to_numpy()] for name in features.files})
    shutil.copy2(inputs / 'relative.npy', upgrade / 'relative.npy')
    shutil.copy2(inputs / 'rich.npy', rich / 'rich.npy')
    with zipfile.ZipFile(prediction / 'model-code.zip', 'w') as z:
        z.write(ROOT / 'reproduction/training/neural.py', 'neural.py')
    for file in wheel_dir.iterdir():
        if file.suffix == '.whl':
            shutil.copy2(file, wheels / file.name)
    if not any(wheels.glob('*.whl')):
        raise ValueError('Wheel directory is empty')
    for folder, name in ((prediction, 'prediction-input-manifest.json'),
                         (training, 'manifest.json'),
                         (upgrade, 'upgrade-input-manifest.json'),
                         (rich, 'rich-input-manifest.json')):
        manifest(folder, name)
    manifest(wheels, 'wheel-manifest.json')
    # The training scripts expect this specific key in the wheel manifest.
    p = wheels / 'wheel-manifest.json'
    p.write_text(json.dumps({'sha256': json.loads(p.read_text())}, indent=2) + '\n')
    print(f'Upload the staged directory as a private Kaggle dataset: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--wheels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage(args.inputs, args.wheels, args.output)
