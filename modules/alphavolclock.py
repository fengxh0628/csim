"""Volume clock alpha: time-of-day volume pattern anomalies."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration


class AlphaVolClock(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.lookback_days = int(cfg.get('lookback', 7))
        self.n_buckets = int(cfg.get('n_buckets', 6))
        self.ivol = self.dr.getdata('itvl.volume')
        self.iclose = self.dr.getdata('itvl.close')

    def generate(self, idx: int) -> None:
        didx = idx - self.delay
        bpd = univbase.bars_per_day
        total_bars = (self.lookback_days + 1) * bpd
        if didx < total_bars:
            return

        bucket_size = bpd // self.n_buckets

        # Today's volume distribution
        today_vol = np.zeros((self.n_buckets, univbase.n_symbols), dtype=np.float32)
        for b in range(self.n_buckets):
            start = didx - bpd + b * bucket_size
            end = start + bucket_size
            today_vol[b, :] = np.nansum(self.ivol[start:end, :], axis=0)

        today_total = today_vol.sum(axis=0)
        valid = today_total > 0
        if not valid.any():
            return

        today_pct = np.zeros_like(today_vol)
        today_pct[:, valid] = today_vol[:, valid] / today_total[valid]

        # Historical average
        hist_vol = np.zeros((self.n_buckets, univbase.n_symbols), dtype=np.float32)
        for d in range(1, self.lookback_days + 1):
            for b in range(self.n_buckets):
                start = didx - (d + 1) * bpd + b * bucket_size
                end = start + bucket_size
                hist_vol[b, :] += np.nansum(self.ivol[start:end, :], axis=0)

        hist_total = hist_vol.sum(axis=0)
        v = valid & (hist_total > 0)
        hist_pct = np.zeros_like(hist_vol)
        hist_pct[:, v] = hist_vol[:, v] / hist_total[v]

        last_bucket_excess = today_pct[-1, :] - hist_pct[-1, :]

        cur = self.iclose[didx, :]
        prev = self.iclose[didx - bpd, :]
        direction = np.where((cur > 0) & (prev > 0), np.sign(cur / prev - 1.0), 0.0)

        self.alpha[v] = last_bucket_excess[v] * direction[v]


def create(cfg: dict) -> AlphaVolClock:
    return AlphaVolClock(cfg)
