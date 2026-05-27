"""Volume-based alpha signals."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration


class AlphaVolume(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.lookback_bars = parse_duration(cfg.get('lookback', '5d'))
        self.window_bars = parse_duration(cfg.get('window', '1d'))
        self.signal_type = cfg.get('signal', 'taker_ratio')
        self.ivol = self.dr.getdata('volume')
        self.itaker = self.dr.getdata('taker_buy_volume')

    def generate(self, idx: int) -> None:
        didx = idx - self.delay
        if didx < self.lookback_bars + self.window_bars:
            return

        if self.signal_type == 'volume_change':
            valid = self.get_valid(idx)
            cur_vol = np.nansum(self.ivol[didx - self.window_bars:didx, :], axis=0)
            avg_vol = np.nansum(self.ivol[didx - self.lookback_bars:didx - self.window_bars, :], axis=0)
            n_periods = (self.lookback_bars - self.window_bars) / self.window_bars
            if n_periods > 0:
                avg_vol /= n_periods
            valid &= (avg_vol > 0) & (cur_vol > 0)
            self.alpha[valid] = cur_vol[valid] / avg_vol[valid] - 1.0

        elif self.signal_type == 'taker_ratio':
            valid = self.get_valid(idx)
            tb = np.nansum(self.itaker[didx - self.window_bars:didx, :], axis=0)
            tv = np.nansum(self.ivol[didx - self.window_bars:didx, :], axis=0)
            valid &= (tv > 0) & np.isfinite(tb)
            self.alpha[valid] = tb[valid] / tv[valid] - 0.5


def create(cfg: dict) -> AlphaVolume:
    return AlphaVolume(cfg)
