"""Autocorrelation-based alpha."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration


class AlphaAutocorr(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.lookback_bars = parse_duration(cfg.get('lookback', '20d'))
        self.iclose = self.dr.getdata('itvl.close')

    def generate(self, idx: int) -> None:
        didx = idx - self.delay
        bpd = univbase.bars_per_day
        if didx < self.lookback_bars + bpd:
            return

        # Daily returns over lookback
        close_daily = self.iclose[didx - self.lookback_bars:didx + 1:bpd, :]
        rets = close_daily[1:, :] / close_daily[:-1, :] - 1.0

        for si in range(univbase.n_symbols):
            r = rets[:, si]
            v = np.isfinite(r)
            if v.sum() < 10:
                continue
            rv = r[v]
            r1 = rv[:-1]
            r2 = rv[1:]
            m1, m2 = r1.mean(), r2.mean()
            d1, d2 = r1 - m1, r2 - m2
            denom = np.sqrt(np.dot(d1, d1) * np.dot(d2, d2))
            if denom > 0:
                autocorr = np.dot(d1, d2) / denom
            else:
                continue
            self.alpha[si] = autocorr * rv[-1]


def create(cfg: dict) -> AlphaAutocorr:
    return AlphaAutocorr(cfg)
