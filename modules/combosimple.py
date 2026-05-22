"""Simple equal-weight (or weighted) alpha combination."""
import numpy as np

from modules.alphabase import AlphaBase
from lib import fast


class ComboSimple(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def generate(self, idx: int) -> None:
        alphas = np.array([child.module.alpha for child in self.children], dtype=np.float32)
        weights = np.array([child.module.cfg.get('weight', 1.0) for child in self.children], dtype=np.float32)[:, np.newaxis]

        weighted = alphas * weights
        w_sum = np.where(np.isfinite(alphas), weights, 0.0).sum(axis=0)
        w_sum[w_sum == 0] = np.nan
        self.alpha[:] = np.nansum(weighted, axis=0) / w_sum


def create(cfg: dict) -> ComboSimple:
    return ComboSimple(cfg)
