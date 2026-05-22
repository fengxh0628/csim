"""Data manager for funding rate (premiumIndexKlines, 5m interval).

Stores on interval axis: (n_intervals, n_symbols).
Every 5m bar has a funding rate value.

Path: {datapath}/futures/um/{monthly|daily}/premiumIndexKlines/{SYMBOL}/5m/{SYMBOL}-5m-{date}.csv
"""
import numpy as np
import pandas as pd
import os
from datetime import date

from modules.dmgrbase import DmgrBase
from core.universe import univbase
from core.sim_config import simcfg


class DmgrFunding(DmgrBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.datapath = cfg['datapath']
        self.data_type = cfg.get('data_type', 'premiumIndexKlines')
        self.interval = cfg.get('interval', '5m')

    def initialize(self):
        super().initialize()
        self.add_itvl_data(['funding.rate'])

    def load_data(self):
        if self.mode == 'r':
            return

        update_idx = self.get_update_idx()
        if self._rebuilt:
            update_idx = 0

        di_start = update_idx // univbase.bars_per_day
        if di_start >= univbase.n_dates - 1:
            print(f'[{self.id}] Already up to date')
            return

        print(f'[{self.id}] Loading funding rates from di={di_start}...')
        funding = self.dr.getdata('funding.rate')
        bpd = univbase.bars_per_day
        n_loaded = 0

        for si, symbol in enumerate(univbase.symbols):
            dfs = self._load_symbol_csvs(symbol, univbase.dates[di_start])
            for di in range(di_start, univbase.n_dates):
                d = univbase.dates[di]
                if d not in dfs:
                    continue
                df = dfs[d]
                rates = df['close'].values
                offset = di * bpd
                n_bars = min(len(rates), bpd)
                funding[offset:offset + n_bars, si] = rates[:n_bars]
                n_loaded += 1

            if (si + 1) % 10 == 0:
                funding.flush()
                print(f'  [{si+1}/{univbase.n_symbols}] {symbol} done')

        funding.flush()
        self.set_update_idx(univbase.n_intervals - 1)
        print(f'[{self.id}] Loaded {n_loaded} symbol-days')

    def _load_symbol_csvs(self, symbol: str, since: date = None) -> dict[date, pd.DataFrame]:
        result = {}
        for subdir in ['monthly', 'daily']:
            sym_dir = os.path.join(self.datapath, 'futures', 'um', subdir,
                                   self.data_type, symbol, self.interval)
            if not os.path.isdir(sym_dir):
                continue
            for fname in sorted(os.listdir(sym_dir)):
                if not fname.endswith('.csv'):
                    continue

                if since is not None and subdir == 'daily':
                    try:
                        parts = fname.replace('.csv', '').split('-')
                        file_date = date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
                        if file_date < since:
                            continue
                    except (ValueError, IndexError):
                        pass

                fpath = os.path.join(sym_dir, fname)
                try:
                    df = pd.read_csv(fpath)
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                    for d, day_df in df.groupby(df['open_time'].dt.date):
                        if univbase.has_date(d) and (since is None or d >= since):
                            result[d] = day_df.reset_index(drop=True)
                except Exception:
                    continue
        return result


def create(cfg: dict) -> DmgrFunding:
    mgr = DmgrFunding(cfg)
    mgr.initialize()
    return mgr
