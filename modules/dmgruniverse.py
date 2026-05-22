"""Dynamic universe filter: monthly liquidity-based mask.

Each month, selects top N symbols by total quote_volume over the past K months.
Produces a mask (n_intervals, n_symbols) that is constant within each month.

Config:
  - top_n: number of symbols to keep (default: 50)
  - lookback_months: lookback window in months (default: 12)

Registers:
  universe_mask  (n_intervals, n_symbols) bool — True if symbol is in universe
"""
import numpy as np
import os
import yaml

from modules.dmgrbase import DmgrBase
from core.universe import univbase


class DmgrUniverse(DmgrBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.top_n = int(cfg.get('top_n', 50))
        self.lookback_months = int(cfg.get('lookback_months', 12))

    def dependencies(self) -> list[str]:
        return ['quote_volume']

    def initialize(self):
        super().initialize()
        
        # Create persistent memmap for universe_mask
        shape = (univbase.n_intervals, univbase.n_symbols)
        meta_file = os.path.join(self.data_dir, '.meta')
        
        if not os.path.exists(meta_file):
            self._rebuilt = True
            mask_path = os.path.join(self.data_dir, 'universe_mask')
            data = np.memmap(mask_path, dtype=bool, mode='w+', shape=shape)
            data[:] = False
            data.flush()
            
            meta = {
                'fields': {'universe_mask': {'shape': list(shape)}},
                'datapath': self.cfg.get('datapath', ''),
                'update_idx': 0,
            }
            with open(meta_file, 'w') as f:
                yaml.dump(meta, f, default_flow_style=False)
        
        self.dr.setdata('universe_mask', self._open_mask())

    def _open_mask(self):
        mask_path = os.path.join(self.data_dir, 'universe_mask')
        return np.memmap(mask_path, dtype=bool, mode='r+', shape=(univbase.n_intervals, univbase.n_symbols))

    def load_data(self):
        iqvol = self.dr.getdata('quote_volume')
        mask = self.dr.getdata('universe_mask')
        bpd = univbase.bars_per_day

        # Group dates by month
        months = {}
        for di, d in enumerate(univbase.dates):
            ym = (d.year, d.month)
            if ym not in months:
                months[ym] = []
            months[ym].append(di)

        sorted_months = sorted(months.keys())

        for mi, ym in enumerate(sorted_months):
            # Get lookback window: past K months, excluding current month
            # Use months strictly before the current month to avoid look-ahead bias
            start_mi = max(0, mi - self.lookback_months)
            window_months = sorted_months[start_mi:mi]

            # Sum quote_volume over the window
            window_vol = np.zeros(univbase.n_symbols, dtype=np.float64)
            for w_ym in window_months:
                for di in months[w_ym]:
                    sl = univbase.day_slice(di)
                    window_vol += np.nansum(iqvol[sl, :], axis=0)

            # Select top N
            if window_vol.sum() == 0:
                continue
            ranked = np.argsort(-window_vol)
            top_symbols = set(ranked[:self.top_n])

            # Apply mask for all intervals in this month
            for di in months[ym]:
                sl = univbase.day_slice(di)
                for si in range(univbase.n_symbols):
                    mask[sl, si] = si in top_symbols

        mask.flush()
        n_avg = np.mean(mask[::bpd, :].sum(axis=1))
        print(f'[{self.id}] Universe: avg {n_avg:.0f} symbols/month')


def create(cfg: dict) -> DmgrUniverse:
    mgr = DmgrUniverse(cfg)
    mgr.initialize()
    return mgr
