"""Crypto universe: symbols, timestamps, and rebalance schedule.

Data is stored as (n_intervals, n_symbols) — a flat continuous time axis.
Rebalance points (daily) are tracked separately.

Persisted to datacache/__universe/:
  - instruments.mpk: list of symbol strings
  - dates.mpk: list of date ints (YYYYMMDD)
  - intervals.mpk: total number of intervals
"""
import os
from datetime import date, timedelta, datetime
import msgpack
import numpy as np

from core.sim_config import simcfg


INTERVAL_MINUTES = {
    '1m': 1,
    '3m': 3,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
}


class Universe:

    def __init__(self):
        self.symbols: list[str] = []
        self.dates: list[date] = []
        self.n_intervals: int = 0
        self.bars_per_day: int = 0
        self.interval_minutes: int = 0
        self._sym_idx: dict[str, int] = {}
        self._date_idx: dict[date, int] = {}
        # rebalance_idx[di] = interval index for the start of day di
        self.rebalance_idx: np.ndarray = None
        self.initialize()

    def initialize(self):
        interval = simcfg.constants.get('interval', '5m')
        self.interval_minutes = INTERVAL_MINUTES[interval]
        self.bars_per_day = 1440 // self.interval_minutes
        # Auto-append interval to datacache path
        datacache = os.path.join(simcfg.constants['datacache'], interval)
        simcfg.constants['datacache'] = datacache

        univ_path = os.path.join(datacache, '__universe')
        os.makedirs(univ_path, exist_ok=True)

        instruments_path = os.path.join(univ_path, 'instruments.mpk')
        dates_path = os.path.join(univ_path, 'dates.mpk')

        # Load existing
        if os.path.exists(instruments_path):
            with open(instruments_path, 'rb') as f:
                self.symbols = list(msgpack.unpackb(f.read(), raw=False))

        if os.path.exists(dates_path):
            with open(dates_path, 'rb') as f:
                dates_int = msgpack.unpackb(f.read(), raw=False)
            self.dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in dates_int]

        mode = simcfg.constants.get('mode', 'r')
        if mode != 'r':
            # Update symbols: list, file, or auto-discover from directory
            symbols_cfg = simcfg.universe.get('symbols', [])
            if isinstance(symbols_cfg, str):
                if os.path.isdir(symbols_cfg):
                    # Auto-discover: scan directory for symbol subdirectories
                    symbols_cfg = sorted([
                        d for d in os.listdir(symbols_cfg)
                        if os.path.isdir(os.path.join(symbols_cfg, d)) and d.endswith('USDT')
                    ])
                else:
                    # File with one symbol per line
                    with open(symbols_cfg, 'r') as f:
                        symbols_cfg = [line.strip() for line in f if line.strip()]

            new_symbols = [s for s in symbols_cfg if s not in self.symbols]
            if new_symbols:
                print(f'Updating universe: +{len(new_symbols)} symbols')
                self.symbols.extend(new_symbols)
                with open(instruments_path, 'wb') as f:
                    f.write(msgpack.packb(self.symbols))

            # Update dates
            start = datetime.strptime(str(simcfg.constants['startdate']), '%Y%m%d').date()
            end = datetime.strptime(str(simcfg.constants['enddate']), '%Y%m%d').date()
            all_dates = []
            d = start
            while d <= end:
                all_dates.append(d)
                d += timedelta(days=1)

            new_dates = [d for d in all_dates if d not in set(self.dates)]
            if new_dates:
                print(f'Updating universe: +{len(new_dates)} dates')
                self.dates.extend(new_dates)
                self.dates.sort()
                dates_int = [int(d.strftime('%Y%m%d')) for d in self.dates]
                with open(dates_path, 'wb') as f:
                    f.write(msgpack.packb(dates_int))

        self._sym_idx = {s: i for i, s in enumerate(self.symbols)}
        self._date_idx = {d: i for i, d in enumerate(self.dates)}
        self.n_intervals = len(self.dates) * self.bars_per_day

        # Build rebalance index: rebalance_idx[di] = first interval of day di
        self.rebalance_idx = np.arange(len(self.dates)) * self.bars_per_day

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    def sym_to_idx(self, symbol: str) -> int:
        return self._sym_idx[symbol]

    def date_to_idx(self, d: date) -> int:
        return self._date_idx[d]

    def has_date(self, d: date) -> bool:
        return d in self._date_idx

    def itvl_to_didx(self, itvl: int) -> int:
        """Which day does this interval belong to?"""
        return itvl // self.bars_per_day

    def itvl_to_date(self, itvl: int) -> date:
        """Convert interval index to calendar date."""
        return self.dates[self.itvl_to_didx(itvl)]

    def itvl_to_timestamp(self, itvl: int) -> str:
        """Convert interval index to YYYYMMDDHHMMSS string."""
        di = itvl // self.bars_per_day
        bar_in_day = itvl % self.bars_per_day
        minutes = bar_in_day * self.interval_minutes
        h, m = divmod(minutes, 60)
        return f'{self.dates[di].strftime("%Y%m%d")}{h:02d}{m:02d}00'

    def day_slice(self, di: int) -> slice:
        """Return slice of interval indices for day di."""
        start = di * self.bars_per_day
        return slice(start, start + self.bars_per_day)


univbase: Universe = None


def init_universe():
    global univbase
    if univbase is None:
        univbase = Universe()
    return univbase
