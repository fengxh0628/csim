"""Cross-sectional z-score normalization."""
import numpy as np

from modules.opbase import OpBase


class OpZscore(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        v = np.isfinite(alpha)
        if v.sum() < 2:
            return
        mean = np.nanmean(alpha[v])
        std = np.nanstd(alpha[v])
        if std > 0:
            alpha[v] = (alpha[v] - mean) / std
        else:
            alpha[v] = 0.0


def create(cfg: dict) -> OpZscore:
    return OpZscore(cfg)
