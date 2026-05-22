#!/usr/bin/env python3
"""One-off script: scan all symbols' volume to find the superset universe.

Each month, selects the top N symbols by total quote_volume over the past K months.
Output: union of all monthly top N symbols.

Optimized: only reads quote_volume column, parses date from filename, uses multiprocessing.

Usage:
  python select_universe.py --datapath /path/to/data --interval 1m --top 50 --lookback 6 --output symbols.txt
"""
import argparse
import os
import re
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


BARS_PER_DAY = {'1m': 1440, '5m': 288, '15m': 96, '1h': 24}

# Parse year/month from filename: SYMBOL-interval-YYYY-MM.csv or SYMBOL-interval-YYYY-MM-DD.csv
MONTHLY_RE = re.compile(r'-(\d{4})-(\d{2})\.csv$')
DAILY_RE = re.compile(r'-(\d{4})-(\d{2})-(\d{2})\.csv$')


def scan_symbol(args_tuple):
    """Scan one symbol and return {symbol: {ym_tuple: total_quote_volume}}."""
    symbol, datapath, interval = args_tuple
    result = {}

    for subdir in ['monthly', 'daily']:
        sym_dir = os.path.join(datapath, 'futures', 'um', subdir, 'klines', symbol, interval)
        if not os.path.isdir(sym_dir):
            continue
        for fname in os.listdir(sym_dir):
            if not fname.endswith('.csv'):
                continue

            # Parse year/month from filename
            m = DAILY_RE.search(fname)
            if m:
                year, month = int(m.group(1)), int(m.group(2))
            else:
                m = MONTHLY_RE.search(fname)
                if not m:
                    continue
                year, month = int(m.group(1)), int(m.group(2))

            fpath = os.path.join(sym_dir, fname)
            try:
                # Only read quote_volume column
                df = pd.read_csv(fpath, usecols=['quote_volume'])
                total = df['quote_volume'].sum()
                ym = (year, month)
                result.setdefault(ym, 0.0)
                result[ym] += total
            except Exception:
                continue

    return symbol, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', required=True)
    parser.add_argument('--interval', default='1m')
    parser.add_argument('--top', type=int, default=50, help='Top N symbols per month')
    parser.add_argument('--lookback', type=int, default=6, help='Lookback months for volume calculation')
    parser.add_argument('--start', default='2021-01-01', help='Start date for selection (YYYY-MM-DD)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--output', default='symbols.txt')
    args = parser.parse_args()

    # Discover all symbols
    klines_dir = os.path.join(args.datapath, 'futures', 'um', 'monthly', 'klines')
    if not os.path.isdir(klines_dir):
        klines_dir = os.path.join(args.datapath, 'futures', 'um', 'daily', 'klines')
    all_symbols = sorted([d for d in os.listdir(klines_dir)
                          if os.path.isdir(os.path.join(klines_dir, d)) and d.endswith('USDT')])
    print(f'Found {len(all_symbols)} symbols')

    # Scan symbols in parallel
    # month_volume[(year, month)] = {symbol: quote_volume}
    month_volume = defaultdict(lambda: defaultdict(float))

    tasks = [(s, args.datapath, args.interval) for s in all_symbols]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scan_symbol, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Scanning symbols'):
            symbol, result = future.result()
            for ym, vol in result.items():
                month_volume[ym][symbol] += vol

    # Build sorted list of all months
    start_ym = pd.Timestamp(args.start).to_period('M')
    start_tuple = (start_ym.year, start_ym.month)
    all_months = sorted(ym for ym in month_volume.keys() if ym >= start_tuple)
    n_months = len(all_months)

    # For each month, compute lookback volume and rank
    ever_top = set()

    for mi, (year, month) in enumerate(tqdm(all_months, desc='Ranking months', total=n_months)):
        # Get lookback window: past `lookback` months including current
        start_mi = max(0, mi - args.lookback + 1)
        window = all_months[start_mi:mi + 1]

        # Sum volume over the window
        window_vol = defaultdict(float)
        for ym in window:
            for sym, vol in month_volume[ym].items():
                window_vol[sym] += vol

        ranked = sorted(window_vol.items(), key=lambda x: -x[1])
        for sym, _ in ranked[:args.top]:
            ever_top.add(sym)

    selected = sorted(ever_top)
    print(f'\nSelected {len(selected)} symbols (ever in top {args.top} by {args.lookback}-month rolling volume from {args.start} across {n_months} months)')
    print(f'Sample: {selected[:10]}')

    with open(args.output, 'w') as f:
        for sym in selected:
            f.write(sym + '\n')

    print(f'Written to {args.output}')


if __name__ == '__main__':
    main()
