"""Data manager for funding rate (premiumIndexKlines, 8h interval).

Stores on interval axis: (n_intervals, n_symbols).
Non-zero only at settlement bars (every 8h = every bars_per_day/3 bars).
All other bars are 0.

Path: {datapath}/futures/um/{monthly|daily}/premiumIndexKlines/{SYMBOL}/8h/{SYMBOL}-8h-{date}.csv
"""
import numpy as np
import pandas as pd
import os
from datetime import date

from modules.dmgrbase import DmgrBase
from core.universe import univbase
from core.sim_config import simcfg


# Settlement times in minutes from UTC 00:00
SETTLEMENT_MINUTES = [0, 480, 960]  # 00:00, 08:00, 16:00


class DmgrFunding(DmgrBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.datapath = cfg['datapath']
        self.data_type = cfg.get('data_type', 'premiumIndexKlines')
        self.interval = cfg.get('interval', '8h')

    def initialize(self):
        super().initialize()
        self.add_itvl_data(['funding.rate'])

    def load_data(self):
        if self.mode == 'r':
            return

        print(f'[{self.id}] Loading funding rates...')
        funding = self.dr.getdata('funding.rate')
        bpd = univbase.bars_per_day
        n_loaded = 0

        # Settlement bar offsets within each day
        settlement_offsets = [m // univbase.interval_minutes for m in SETTLEMENT_MINUTES]

        for si, symbol in enumerate(univbase.symbols):
            dfs = self._load_symbol_csvs(symbol)
            for d, df in dfs.items():
                if not univbase.has_date(d):
                    continue
                di = univbase._date_idx[d]
                day_start = di * bpd
                rates = df['close'].values
                # Place each rate at the corresponding settlement bar
                for i, rate in enumerate(rates[:len(settlement_offsets)]):
                    if np.isfinite(rate):
                        bar_idx = day_start + settlement_offsets[i]
                        if bar_idx < funding.shape[0]:
                            funding[bar_idx, si] = rate
                n_loaded += 1

        funding.flush()
        print(f'[{self.id}] Loaded {n_loaded} symbol-days')

    def _load_symbol_csvs(self, symbol: str) -> dict[date, pd.DataFrame]:
        result = {}
        for subdir in ['monthly', 'daily']:
            sym_dir = os.path.join(self.datapath, 'futures', 'um', subdir,
                                   self.data_type, symbol, self.interval)
            if not os.path.isdir(sym_dir):
                continue
            for fname in sorted(os.listdir(sym_dir)):
                if not fname.endswith('.csv'):
                    continue
                fpath = os.path.join(sym_dir, fname)
                try:
                    df = pd.read_csv(fpath)
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                    for d, day_df in df.groupby(df['open_time'].dt.date):
                        result[d] = day_df.reset_index(drop=True)
                except Exception:
                    continue
        return result


def create(cfg: dict) -> DmgrFunding:
    mgr = DmgrFunding(cfg)
    mgr.initialize()
    return mgr
