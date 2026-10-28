"""Exact production CNN architectures used by the selected ensemble."""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cnn'))
from model import CNN  # noqa: E402


class AdaptiveCNN(CNN):
    def __init__(self, tab_dim, variant):
        super().__init__(tab_dim)
        self.variant = variant
        if variant in ('dilated', 'seed43', 'seed44', 'combo', 'wide_combo'):
            for i, d in ((2, 1), (4, 2), (5, 4)):
                self.profile[i].layers[3] = nn.Conv1d(64, 64, 5, padding=2*d, dilation=d)
        if variant in ('basic', 'combo'):
            self.head[-2] = nn.Linear(128, 2)
            self.head[-1] = nn.Identity()
        if variant in ('wide', 'wide_combo'):
            self.wide = nn.Sequential(nn.Linear(2520, 128), nn.GELU(), nn.Dropout(.3),
                                      nn.Linear(128, 64), nn.GELU())
            self.head[0] = nn.Linear(768, 128)

    def forward(self, tab, profile, wide=None):
        if self.variant in ('basic', 'combo'):
            raw = self.head(torch.cat([self.context(tab), self.profile(profile)], dim=1))
            return torch.stack([raw[:, 0], torch.sigmoid(raw[:, 1])], dim=1)
        if self.variant in ('wide', 'wide_combo'):
            return self.head(torch.cat([self.context(tab), self.profile(profile),
                                        self.wide(wide)], dim=1)).squeeze(-1)
        return super().forward(tab, profile)
