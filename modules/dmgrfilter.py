"""Universe filter: produces a daily tradeable mask.

Filters symbols based on:
  - min_volume: minimum daily quote volume (USDT) over lookback period
  - min_days: minimum number of days with data (excludes newly listed)
  - top_n: only keep top N by volume (optional)

Registers:
  tradeable  (n_dates, n_symbols) bool — True if symbol is tradeable on that day
"""
import numpy as np

from modules.dmgrbase import DmgrBase
from core.universe import univbase
from core.sim_config import simcfg


class DmgrFilter(DmgrBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.min_volume = float(cfg.get('min_volume', 1e7))  # $10M daily
        self.min_days = int(cfg.get('min_days', 30))  # 30 days of history
        self.top_n = int(cfg.get('top_n', 50))
        self.lookback = int(cfg.get('lookback', 5))  # average over N days

    def dependencies(self) -> list[str]:
        return ['itvl.quote_volume']

    def initialize(self):
        super().initialize()
        # Tradeable mask: (n_intervals, n_symbols), forward-filled per day
        shape = (univbase.n_intervals, univbase.n_symbols)
        data = np.full(shape, False, dtype=bool)
        self.dr.setdata('tradeable', data)

    def load_data(self):
        iqvol = self.dr.getdata('itvl.quote_volume')
        tradeable = self.dr.getdata('tradeable')
        bpd = univbase.bars_per_day

        for di in range(univbase.n_dates):
            if di < self.min_days:
                continue

            # Compute daily quote volume for each day in lookback
            lb_start = max(0, di - self.lookback)
            daily_vols = np.zeros((di - lb_start, univbase.n_symbols), dtype=np.float32)
            for d in range(lb_start, di):
                sl = univbase.day_slice(d)
                daily_vols[d - lb_start, :] = np.nansum(iqvol[sl, :], axis=0)

            avg_vol = np.nanmean(daily_vols, axis=0)
            has_data = avg_vol > 0

            # Volume filter
            vol_ok = avg_vol >= self.min_volume

            # History filter: count days with any volume
            valid_days = np.sum(daily_vols > 0, axis=0)
            # Also check earlier days
            for d in range(0, lb_start):
                sl = univbase.day_slice(d)
                day_vol = np.nansum(iqvol[sl, :], axis=0)
                valid_days += (day_vol > 0).astype(int)
            history_ok = valid_days >= self.min_days

            mask = has_data & vol_ok & history_ok

            # Top N filter
            if self.top_n > 0 and mask.sum() > self.top_n:
                vol_masked = np.where(mask, avg_vol, 0)
                threshold = np.sort(vol_masked)[-self.top_n]
                mask = mask & (avg_vol >= threshold)

            # Forward-fill: same mask for all bars in this day
            sl = univbase.day_slice(di)
            tradeable[sl, :] = mask[np.newaxis, :]

        bpd = univbase.bars_per_day
        n_avg = np.mean(tradeable[self.min_days * bpd::bpd, :].sum(axis=1))
        print(f'[{self.id}] Tradeable: avg {n_avg:.0f} symbols/day')


def create(cfg: dict) -> DmgrFilter:
    mgr = DmgrFilter(cfg)
    mgr.initialize()
    return mgr
