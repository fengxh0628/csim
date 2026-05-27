#!/usr/bin/env python3
"""Deep verify: check if mask truly matches top 50 by past 12-month quote_volume."""
import os
import sys
import numpy as np
import yaml
import msgpack
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    cache_dir = '/home/fengxh/csimcache/5m'
    univ_path = os.path.join(cache_dir, '__universe')
    universe_dir = os.path.join(cache_dir, 'universe')
    klines_dir = os.path.join(cache_dir, 'klines')

    # Load dates
    with open(os.path.join(univ_path, 'dates.mpk'), 'rb') as f:
        dates_int = msgpack.unpackb(f.read(), raw=False)
    dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in dates_int]
    bpd = 288

    # Load mask
    with open(os.path.join(universe_dir, '.meta'), 'r') as f:
        mask_meta = yaml.safe_load(f)
    mask_shape = tuple(mask_meta['fields']['universe_mask']['shape'])
    mask = np.memmap(os.path.join(universe_dir, 'universe_mask'), dtype=bool, mode='r', shape=mask_shape)

    # Load qvol
    with open(os.path.join(klines_dir, '.meta'), 'r') as f:
        klines_meta = yaml.safe_load(f)
    qvol_shape = tuple(klines_meta['fields']['quote_volume']['shape'])
    qvol = np.memmap(os.path.join(klines_dir, 'quote_volume'), dtype=np.float32, mode='r', shape=qvol_shape)

    # Load symbols
    with open(os.path.join(univ_path, 'instruments.mpk'), 'rb') as f:
        symbols = list(msgpack.unpackb(f.read(), raw=False))

    # Group dates by month
    months = {}
    for di, d in enumerate(dates):
        ym = (d.year, d.month)
        if ym not in months:
            months[ym] = []
        months[ym].append(di)
    sorted_months = sorted(months.keys())

    # Check specific dates
    check_dates = [
        datetime(2021, 5, 1).date(),
        datetime(2022, 2, 26).date(),
        datetime(2023, 5, 1).date(),
        datetime(2025, 1, 1).date(),
    ]

    for check_date in check_dates:
        try:
            di = next(i for i, d in enumerate(dates) if d == check_date)
        except StopIteration:
            print(f"Date {check_date} not found.")
            continue

        ym = (check_date.year, check_date.month)
        mi = sorted_months.index(ym)
        
        # Lookback 12 months
        start_mi = max(0, mi - 12 + 1)
        window_months = sorted_months[start_mi:mi + 1]
        
        # Calculate true top 50
        window_vol = np.zeros(len(symbols), dtype=np.float64)
        for w_ym in window_months:
            for w_di in months[w_ym]:
                window_vol += np.nansum(qvol[w_di * bpd:(w_di + 1) * bpd, :], axis=0)
        
        true_top50 = set(np.argsort(-window_vol)[:50])
        mask_top50 = set(np.where(mask[di * bpd, :])[0])
        
        # Check match
        match = true_top50 == mask_top50
        missing = true_top50 - mask_top50
        extra = mask_top50 - true_top50
        
        print(f"\n=== {check_date} (Month {ym}, Index {mi}) ===")
        print(f"Lookback months: {window_months[0]} to {window_months[-1]}")
        print(f"Mask matches true Top 50: {match}")
        if missing:
            print(f"  Missing from mask: {[symbols[s] for s in list(missing)[:5]]}")
        if extra:
            print(f"  Extra in mask: {[symbols[s] for s in list(extra)[:5]]}")
            
        # Check NaN in mask symbols
        mask_syms = [symbols[s] for s in mask_top50]
        nan_count = 0
        for s in mask_top50:
            if np.isnan(qvol[di * bpd, s]):
                nan_count += 1
        print(f"NaN count among mask symbols: {nan_count}/50")
        if nan_count > 0:
            nan_syms = [symbols[s] for s in mask_top50 if np.isnan(qvol[di * bpd, s])]
            print(f"  NaN symbols: {nan_syms[:5]}")

if __name__ == '__main__':
    main()
