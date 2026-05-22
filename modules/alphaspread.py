"""Spread/range-based alpha signals."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration


class AlphaSpread(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.signal_type = cfg.get('signal', 'range_compression')
        self.lookback_days = int(cfg.get('lookback', 5))
        self.iclose = self.dr.getdata('itvl.close')
        self.ihigh = self.dr.getdata('itvl.high')
        self.ilow = self.dr.getdata('itvl.low')

    def generate(self, idx: int) -> None:
        bpd = univbase.bars_per_day
        lookback_bars = self.lookback_days * bpd
        if idx < lookback_bars + bpd:
            return

        if self.signal_type == 'range_compression':
            today_high = np.nanmax(self.ihigh[idx - bpd:idx, :], axis=0)
            today_low = np.nanmin(self.ilow[idx - bpd:idx, :], axis=0)
            today_range = today_high - today_low

            avg_range = np.zeros(univbase.n_symbols, dtype=np.float32)
            for d in range(1, self.lookback_days + 1):
                sl = slice(idx - (d + 1) * bpd, idx - d * bpd)
                day_h = np.nanmax(self.ihigh[sl, :], axis=0)
                day_l = np.nanmin(self.ilow[sl, :], axis=0)
                avg_range += (day_h - day_l)
            avg_range /= self.lookback_days

            valid = (avg_range > 0) & np.isfinite(today_range)
            compression = np.full(univbase.n_symbols, np.nan, dtype=np.float32)
            compression[valid] = 1.0 - today_range[valid] / avg_range[valid]

            cur = self.iclose[idx, :]
            prev = self.iclose[idx - bpd, :]
            direction = np.where((cur > 0) & (prev > 0), cur / prev - 1.0, 0.0)

            v = np.isfinite(compression) & (compression > 0)
            self.alpha[v] = compression[v] * direction[v]

        elif self.signal_type == 'high_low_mom':
            scores = np.zeros(univbase.n_symbols, dtype=np.float32)
            count = np.zeros(univbase.n_symbols, dtype=np.float32)
            for d in range(self.lookback_days):
                sl = slice(idx - (d + 1) * bpd, idx - d * bpd)
                day_h = np.nanmax(self.ihigh[sl, :], axis=0)
                day_l = np.nanmin(self.ilow[sl, :], axis=0)
                day_close = self.iclose[idx - d * bpd - 1, :]
                rng = day_h - day_l
                v = (rng > 0) & np.isfinite(day_close)
                scores[v] += (day_close[v] - day_l[v]) / rng[v] - 0.5
                count[v] += 1
            valid = count >= 2
            self.alpha[valid] = scores[valid] / count[valid]


def create(cfg: dict) -> AlphaSpread:
    return AlphaSpread(cfg)
