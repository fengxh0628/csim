"""Intraday microstructure alpha signals."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration


class AlphaIntraday(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.signal_type = cfg.get('signal', 'close_loc')
        self.window_bars = parse_duration(cfg.get('window', '1d'))
        self.iclose = self.dr.getdata('itvl.close')
        self.ivol = self.dr.getdata('itvl.volume')
        self.ihigh = self.dr.getdata('itvl.high')
        self.ilow = self.dr.getdata('itvl.low')

    def generate(self, idx: int) -> None:
        if idx < self.window_bars:
            return

        sl = slice(idx - self.window_bars, idx)
        half = self.window_bars // 2

        if self.signal_type == 'intraday_vol_ratio':
            vol = self.ivol[sl, :]
            first_half = np.nansum(vol[:half, :], axis=0)
            second_half = np.nansum(vol[half:, :], axis=0)
            valid = (second_half > 0) & (first_half > 0)
            self.alpha[valid] = -(first_half[valid] / second_half[valid] - 1.0)

        elif self.signal_type == 'close_loc':
            high = self.ihigh[sl, :]
            low = self.ilow[sl, :]
            close = self.iclose[sl, :]

            day_high = np.nanmax(high, axis=0)
            day_low = np.nanmin(low, axis=0)
            valid_c = np.isfinite(close)
            has_close = valid_c.any(axis=0)
            last_idx_arr = close.shape[0] - 1 - np.argmax(valid_c[::-1, :], axis=0)
            last_close = close[last_idx_arr, np.arange(univbase.n_symbols)]

            rng = day_high - day_low
            valid = (rng > 0) & has_close
            self.alpha[valid] = (last_close[valid] - day_low[valid]) / rng[valid] - 0.5

        elif self.signal_type == 'amihud':
            close = self.iclose[sl, :]
            vol = self.ivol[sl, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                rets = np.diff(close, axis=0) / close[:-1, :]
            abs_rets = np.nansum(np.abs(rets), axis=0)
            total_vol = np.nansum(vol, axis=0)
            valid = (total_vol > 0) & np.isfinite(abs_rets)
            self.alpha[valid] = -(abs_rets[valid] / total_vol[valid])


def create(cfg: dict) -> AlphaIntraday:
    return AlphaIntraday(cfg)
