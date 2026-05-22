"""Cross-sectional momentum alpha: N-period return up to current bar."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration


class AlphaMomentum(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.lookback_bars = parse_duration(cfg.get('lookback', '5d'))
        self.iclose = self.dr.getdata('itvl.close')

    def generate(self, idx: int) -> None:
        prev_idx = idx - self.lookback_bars
        if prev_idx < 0:
            return
        cur = self.iclose[idx, :]
        prev = self.iclose[prev_idx, :]
        valid = (cur > 0) & (prev > 0) & np.isfinite(cur) & np.isfinite(prev)
        self.alpha[valid] = cur[valid] / prev[valid] - 1.0


def create(cfg: dict) -> AlphaMomentum:
    return AlphaMomentum(cfg)
