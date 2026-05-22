"""Data manager for Binance USDT-M perpetual futures klines (memmap, flat time axis).

Data shape: (n_intervals, n_symbols) where n_intervals = n_dates * bars_per_day.

Raw CSV layout:
  {datapath}/futures/um/monthly/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM}.csv
  {datapath}/futures/um/daily/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM-DD}.csv

Registered fields:
  itvl.{open,high,low,close,volume,quote_volume,count,taker_buy_volume}  (n_intervals, n_symbols)

Incremental update: tracks update_didx in .meta. On next run, only loads dates >= update_didx
for ALL symbols.
"""
import numpy as np
import pandas as pd
import os
from datetime import date

from modules.dmgrbase import DmgrBase
from core.universe import univbase
from core.sim_config import simcfg


ITVL_FIELDS = ['itvl.open', 'itvl.high', 'itvl.low', 'itvl.close',
               'itvl.volume', 'itvl.quote_volume', 'itvl.count', 'itvl.taker_buy_volume']


class DmgrKlines(DmgrBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.datapath = cfg['datapath']
        self.interval = simcfg.constants.get('interval', '5m')

    def initialize(self):
        super().initialize()
        self.add_itvl_data(ITVL_FIELDS)

    def load_data(self):
        if self.mode == 'r':
            return

        update_idx = self.get_update_idx()

        if self._rebuilt:
            update_idx = 0

        # Convert idx to date index
        di_start = update_idx // univbase.bars_per_day

        if di_start >= univbase.n_dates - 1:
            print(f'[{self.id}] Already up to date')
            return

        print(f'[{self.id}] Loading klines ({self.interval}) from di={di_start} '
              f'({univbase.dates[di_start]})...')

        idata = {f: self.dr.getdata(f) for f in ITVL_FIELDS}
        bars_per_day = univbase.bars_per_day
        n_loaded = 0

        # Load ALL symbols from di_start onwards
        for si, symbol in enumerate(univbase.symbols):
            dfs = self._load_symbol_csvs(symbol, univbase.dates[di_start])

            for di in range(di_start, univbase.n_dates):
                d = univbase.dates[di]
                if d not in dfs:
                    continue
                df = dfs[d]
                n_bars = min(len(df), bars_per_day)

                offset = di * bars_per_day
                sl = slice(offset, offset + n_bars)

                idata['itvl.open'][sl, si] = df['open'].values[:n_bars]
                idata['itvl.high'][sl, si] = df['high'].values[:n_bars]
                idata['itvl.low'][sl, si] = df['low'].values[:n_bars]
                idata['itvl.close'][sl, si] = df['close'].values[:n_bars]
                idata['itvl.volume'][sl, si] = df['volume'].values[:n_bars]
                idata['itvl.quote_volume'][sl, si] = df['quote_volume'].values[:n_bars]
                idata['itvl.count'][sl, si] = df['count'].values[:n_bars]
                idata['itvl.taker_buy_volume'][sl, si] = df['taker_buy_volume'].values[:n_bars]
                n_loaded += 1

            if (si + 1) % 10 == 0:
                for arr in idata.values():
                    arr.flush()
                print(f'  [{si+1}/{univbase.n_symbols}] {symbol} done')

        for arr in idata.values():
            arr.flush()

        # Mark fully updated
        self.set_update_idx(univbase.n_intervals - 1)
        print(f'[{self.id}] Loaded {n_loaded} symbol-days')


    def _load_symbol_csvs(self, symbol: str, since: date = None) -> dict[date, pd.DataFrame]:
        """Load CSV files for a symbol. If since is set, only load files that may contain dates >= since."""
        result = {}
        for subdir in ['monthly', 'daily']:
            sym_dir = os.path.join(self.datapath, 'futures', 'um', subdir, 'klines', symbol, self.interval)
            if not os.path.isdir(sym_dir):
                continue
            for fname in sorted(os.listdir(sym_dir)):
                if not fname.endswith('.csv'):
                    continue

                # Skip files that are definitely before `since`
                if since is not None and subdir == 'monthly':
                    # Monthly file: SYMBOL-interval-YYYY-MM.csv
                    # Only skip if the month ends before `since`
                    try:
                        parts = fname.replace('.csv', '').split('-')
                        y, m = int(parts[-2]), int(parts[-1])
                        from datetime import timedelta
                        if m == 12:
                            month_end = date(y + 1, 1, 1) - timedelta(days=1)
                        else:
                            month_end = date(y, m + 1, 1) - timedelta(days=1)
                        if month_end < since:
                            continue
                    except (ValueError, IndexError):
                        pass
                elif since is not None and subdir == 'daily':
                    # Daily file: SYMBOL-interval-YYYY-MM-DD.csv
                    try:
                        parts = fname.replace('.csv', '').split('-')
                        file_date = date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
                        if file_date < since:
                            continue
                    except (ValueError, IndexError):
                        pass

                fpath = os.path.join(sym_dir, fname)
                df = self._read_csv(fpath)
                if df is None or df.empty:
                    continue
                for d, day_df in df.groupby(df['open_time'].dt.date):
                    if univbase.has_date(d) and (since is None or d >= since):
                        result[d] = day_df.reset_index(drop=True)
        return result

    def _read_csv(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'count', 'taker_buy_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            print(f'  WARNING: {path}: {e}')
            return None


def create(cfg: dict) -> DmgrKlines:
    mgr = DmgrKlines(cfg)
    mgr.initialize()
    return mgr
