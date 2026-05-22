"""Cross-sectional reversal alpha: negative of N-period return."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration


class AlphaReversal(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.lookback_bars = parse_duration(cfg.get('lookback', '1d'))
        self.iclose = self.dr.getdata('itvl.close')

    def generate(self, idx: int) -> None:
        prev_idx = idx - self.lookback_bars
        if prev_idx < 0:
            return
        cur = self.iclose[idx, :]
        prev = self.iclose[prev_idx, :]
        valid = (cur > 0) & (prev > 0) & np.isfinite(cur) & np.isfinite(prev)
        self.alpha[valid] = -(cur[valid] / prev[valid] - 1.0)


def create(cfg: dict) -> AlphaReversal:
    return AlphaReversal(cfg)
