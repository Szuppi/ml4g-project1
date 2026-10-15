"""Fit the fixed eight-epoch ranking CNN on the requested X1/X2 protocols."""
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from common import FOLDS, PROTOCOLS, load_inputs, masks, targets, calibrate, save_predictions, summarize
from cnn.model import CNN, _preprocess, _predict


def run(inputs, output, protocols=PROTOCOLS):
    output = Path(output)
    m, data = load_inputs(inputs)
    torch.set_num_threads(2)
    for protocol in protocols:
        frames = []
        for fold in FOLDS:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            train, validation, evaluate = masks(m, protocol, fold)
            # Transform source validation and target evaluation together; fit only on training.
            selected = validation | evaluate
            ti, ei, pre = _preprocess(data, train, selected, 'rescnn')
            model = CNN(ti[0].shape[1])
            cell = torch.from_numpy(m.loc[train, 'cell_line'].map({'X1': 0, 'X2': 1}).to_numpy(dtype=np.int64, copy=True))
            loader = DataLoader(TensorDataset(*ti, torch.from_numpy(targets(m, train)), cell),
                                batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(42))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
            average, count = None, 0
            for epoch in range(1, 9):
                model.train()
                for tab, profile, label, cells in loader:
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(tab, profile)
                    loss = nn.functional.mse_loss(prediction, label)
                    perm = torch.randperm(len(label))
                    difference = label - label[perm]
                    valid = (cells == cells[perm]) & (difference != 0)
                    if valid.any():
                        loss = loss + .01 * nn.functional.softplus(
                            -torch.sign(difference[valid]) * (prediction[valid] - prediction[perm][valid]) / .1).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    if not torch.isfinite(norm):
                        raise RuntimeError('Nonfinite gradient')
                    optimizer.step()
                if epoch >= 6:
                    if average is None:
                        average = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    else:
                        for k, v in model.state_dict().items():
                            average[k].add_(v.detach())
                    count += 1
                print(json.dumps(dict(protocol=protocol, fold=fold, epoch=epoch)), flush=True)
            model.load_state_dict({k: v / count for k, v in average.items()})
            p = _predict(model, ei, torch.device('cpu'), 128).astype(np.float64)
            alpha, zero = calibrate(m, train, validation, p[validation[selected]])
            raw = p[evaluate[selected]]
            calibrated = np.where(raw <= alpha * zero, 0., raw)
            dest = output / 'models' / protocol / fold
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            np.savez_compressed(dest / 'preprocessing.npz', **pre)
            (dest / 'recipe.json').write_text(json.dumps(dict(alpha=alpha, zero_fraction=zero, seed=42,
                epochs=8, checkpoint='mean_epochs_6_8', train_rows=int(train.sum())), indent=2))
            frames.append(save_predictions(dest / 'validation.csv', m, evaluate, fold, raw, calibrated))
        pd.concat(frames).to_csv(output / (protocol + '.csv'), index=False)
        print(json.dumps(summarize(output)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    p.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=list(PROTOCOLS))
    a = p.parse_args()
    run(a.inputs, a.output, a.protocols)
