"""Volatility-based alpha signals."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration


class AlphaVolatility(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.signal_type = cfg.get('signal', 'vol_mean_revert')
        self.short_bars = parse_duration(cfg.get('short_window', '5d'))
        self.long_bars = parse_duration(cfg.get('long_window', '30d'))
        self.iclose = self.dr.getdata('itvl.close')

    def generate(self, idx: int) -> None:
        if idx < self.long_bars:
            return

        bpd = univbase.bars_per_day

        # Sample daily close for vol computation
        close_long = self.iclose[idx - self.long_bars:idx + 1:bpd, :]
        if close_long.shape[0] < 5:
            return
        rets_long = close_long[1:, :] / close_long[:-1, :] - 1.0

        short_days = self.short_bars // bpd
        short_rets = rets_long[-short_days:, :]
        short_vol = np.nanstd(short_rets, axis=0)
        long_vol = np.nanstd(rets_long, axis=0)

        valid = (long_vol > 0) & np.isfinite(short_vol) & np.isfinite(long_vol)

        if self.signal_type == 'vol_mean_revert':
            self.alpha[valid] = -(short_vol[valid] / long_vol[valid] - 1.0)

        elif self.signal_type == 'vol_breakout':
            mom = close_long[-1, :] / close_long[-short_days, :] - 1.0
            vol_ratio = short_vol / long_vol
            breakout = np.where(vol_ratio < 0.8, mom, 0.0)
            self.alpha[valid] = breakout[valid]


def create(cfg: dict) -> AlphaVolatility:
    return AlphaVolatility(cfg)
