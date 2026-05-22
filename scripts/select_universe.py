#!/usr/bin/env python3
"""One-off script: scan all symbols' daily volume to find the superset universe.

Selects symbols that have EVER been in the top N by daily quote_volume.
Output: a symbols.txt file to use in config.

Usage:
  python select_universe.py --datapath /path/to/data --interval 1m --top 80 --output symbols.txt
"""
import argparse
import os
import pandas as pd
from datetime import date
from collections import defaultdict
from tqdm import tqdm


BARS_PER_DAY = {'1m': 1440, '5m': 288, '15m': 96, '1h': 24}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', required=True)
    parser.add_argument('--interval', default='1m')
    parser.add_argument('--top', type=int, default=50, help='Include symbols ever in daily top N')
    parser.add_argument('--output', default='symbols.txt')
    args = parser.parse_args()

    bars_per_day = BARS_PER_DAY[args.interval]

    # Discover all symbols
    klines_dir = os.path.join(args.datapath, 'futures', 'um', 'monthly', 'klines')
    if not os.path.isdir(klines_dir):
        klines_dir = os.path.join(args.datapath, 'futures', 'um', 'daily', 'klines')
    all_symbols = sorted([d for d in os.listdir(klines_dir)
                          if os.path.isdir(os.path.join(klines_dir, d)) and d.endswith('USDT')])
    print(f'Found {len(all_symbols)} symbols')

    # Count total CSV files for progress bar
    total_files = 0
    for symbol in all_symbols:
        for subdir in ['monthly', 'daily']:
            sym_dir = os.path.join(args.datapath, 'futures', 'um', subdir, 'klines', symbol, args.interval)
            if os.path.isdir(sym_dir):
                total_files += sum(1 for f in os.listdir(sym_dir) if f.endswith('.csv'))

    # day_volume[date] = {symbol: quote_volume}
    day_volume = defaultdict(lambda: defaultdict(float))

    for symbol in tqdm(all_symbols, desc='Scanning symbols'):
        for subdir in ['monthly', 'daily']:
            sym_dir = os.path.join(args.datapath, 'futures', 'um', subdir, 'klines', symbol, args.interval)
            if not os.path.isdir(sym_dir):
                continue
            for fname in os.listdir(sym_dir):
                if not fname.endswith('.csv'):
                    continue
                fpath = os.path.join(sym_dir, fname)
                try:
                    df = pd.read_csv(fpath, usecols=['open_time', 'quote_volume'])
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    for d, day_df in df.groupby(df['open_time'].dt.date):
                        day_volume[d][symbol] = day_df['quote_volume'].sum()
                except Exception:
                    continue

    # For each day, rank symbols and take top N
    ever_top = set()
    n_days = len(day_volume)

    for d, vols in tqdm(sorted(day_volume.items()), desc='Ranking days', total=n_days):
        ranked = sorted(vols.items(), key=lambda x: -x[1])
        for sym, _ in ranked[:args.top]:
            ever_top.add(sym)

    selected = sorted(ever_top)
    print(f'\nSelected {len(selected)} symbols (ever in top {args.top} by daily volume across {n_days} days)')
    print(f'Sample: {selected[:10]}')

    with open(args.output, 'w') as f:
        for sym in selected:
            f.write(sym + '\n')

    print(f'Written to {args.output}')


if __name__ == '__main__':
    main()
