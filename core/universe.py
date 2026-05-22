"""Crypto universe: symbols, timestamps, and rebalance schedule.

Data is stored as (n_intervals, n_symbols) — a flat continuous time axis.
Rebalance points (daily) are tracked separately.

Persisted to datacache/__universe/:
  - instruments.mpk: list of symbol strings
  - dates.mpk: list of date ints (YYYYMMDD)
  - cache_range.mpk: { startdate: int, enddate: int }
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
        cache_range_path = os.path.join(univ_path, 'cache_range.mpk')

        # Load existing
        if os.path.exists(instruments_path):
            with open(instruments_path, 'rb') as f:
                self.symbols = list(msgpack.unpackb(f.read(), raw=False))

        if os.path.exists(dates_path):
            with open(dates_path, 'rb') as f:
                dates_int = msgpack.unpackb(f.read(), raw=False)
            self.dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in dates_int]

        # Load cached date range
        cached_start = None
        cached_end = None
        if os.path.exists(cache_range_path):
            with open(cache_range_path, 'rb') as f:
                cr = msgpack.unpackb(f.read(), raw=False)
            cached_start = datetime.strptime(str(cr['startdate']), '%Y%m%d').date()
            cached_end = datetime.strptime(str(cr['enddate']), '%Y%m%d').date()

        mode = simcfg.constants.get('mode', 'r')
        if mode != 'r':
            cache_startdate = datetime.strptime(str(simcfg.constants['cache_startdate']), '%Y%m%d').date()
            cache_enddate = datetime.strptime(str(simcfg.constants['cache_enddate']), '%Y%m%d').date()

            if cached_start is not None:
                if cache_startdate != cached_start:
                    raise ValueError(
                        f'cache_startdate mismatch: config={cache_startdate} '
                        f'cache={cached_start}. Must match.'
                    )
                if cache_enddate < cached_end:
                    raise ValueError(
                        f'cache_enddate too early: config={cache_enddate} '
                        f'cache={cached_end}. Must be >= cached enddate.'
                    )
                print(f'Cache date range verified: {cache_startdate} -> {cache_enddate} '
                      f'(expanding from {cached_end})')
            else:
                print(f'New cache: {cache_startdate} -> {cache_enddate}')

            # Update symbols: list, file, or auto-discover from directory
            symbols_cfg = simcfg.universe.get('symbols', [])
            if isinstance(symbols_cfg, str):
                if os.path.isdir(symbols_cfg):
                    symbols_cfg = sorted([
                        d for d in os.listdir(symbols_cfg)
                        if os.path.isdir(os.path.join(symbols_cfg, d)) and d.endswith('USDT')
                    ])
                else:
                    with open(symbols_cfg, 'r') as f:
                        symbols_cfg = [line.strip() for line in f if line.strip()]

            new_symbols = [s for s in symbols_cfg if s not in self.symbols]
            if new_symbols:
                print(f'Updating universe: +{len(new_symbols)} symbols')
                self.symbols.extend(new_symbols)
                with open(instruments_path, 'wb') as f:
                    f.write(msgpack.packb(self.symbols))

            # Update dates to cache_enddate
            start = cache_startdate
            end = cache_enddate
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

            # Save cache range
            with open(cache_range_path, 'wb') as f:
                f.write(msgpack.packb({
                    'startdate': int(cache_startdate.strftime('%Y%m%d')),
                    'enddate': int(cache_enddate.strftime('%Y%m%d')),
                }))

        self._sym_idx = {s: i for i, s in enumerate(self.symbols)}
        self._date_idx = {d: i for i, d in enumerate(self.dates)}
        self.n_intervals = len(self.dates) * self.bars_per_day

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
        return itvl // self.bars_per_day

    def itvl_to_date(self, itvl: int) -> date:
        return self.dates[self.itvl_to_didx(itvl)]

    def itvl_to_timestamp(self, itvl: int) -> str:
        di = itvl // self.bars_per_day
        bar_in_day = itvl % self.bars_per_day
        minutes = bar_in_day * self.interval_minutes
        h, m = divmod(minutes, 60)
        return f'{self.dates[di].strftime("%Y%m%d")}{h:02d}{m:02d}00'

    def day_slice(self, di: int) -> slice:
        start = di * self.bars_per_day
        return slice(start, start + self.bars_per_day)


univbase: Universe = None


def init_universe():
    global univbase
    if univbase is None:
        univbase = Universe()
    return univbase
