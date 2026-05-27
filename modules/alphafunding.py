"""Funding rate alpha: high funding = short, low funding = long.

Cross-sectional signal: z-score of recent average funding rate.
High funding means crowded longs paying shorts → expect mean reversion.
"""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase


class AlphaFunding(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.lookback = int(cfg.get('lookback', 3))  # days
        self.funding = self.dr.getdata('funding.rate')

    def generate(self, idx: int) -> None:
        didx = idx - self.delay
        di = univbase.itvl_to_didx(didx)
        if di < self.lookback:
            return
        # Average funding rate over lookback days
        avg_fr = np.nanmean(self.funding[di - self.lookback:di, :], axis=0)
        valid = np.isfinite(avg_fr)
        # Negative: high funding → short (expect longs to get squeezed)
        self.alpha[valid] = -avg_fr[valid]


def create(cfg: dict) -> AlphaFunding:
    return AlphaFunding(cfg)
