#!/usr/bin/env python3
"""bcorr - Compute PnL return correlation between alphas.

Usage:
  python bcorr.py pnl/combo_main pnl/           # one vs all in directory
  python bcorr.py pnl/alpha1 pnl/alpha2         # one vs one
  python bcorr.py pnl/combo_main pnl/ -d 200    # last 200 days only

Output: sorted by correlation (lowest first = most diversifying).
"""
import os
import sys
import numpy as np
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor


def read_pnl_rets(path: str, n_keep: int = 0) -> dict[str, float]:
    """Read PnL file, return {timestamp: ret} dict."""
    rets = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            ts = parts[0]
            ret = float(parts[4])
            rets[ts] = ret
    # Keep last n_keep entries
    if n_keep > 0 and len(rets) > n_keep:
        keys = list(rets.keys())[-n_keep:]
        rets = {k: rets[k] for k in keys}
    return rets


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float('nan')
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = np.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
    if denom == 0:
        return float('nan')
    return float(np.dot(dx, dy) / denom)


def compute_corr(base_rets: dict, path: str, n_keep: int) -> tuple[str, float]:
    other_rets = read_pnl_rets(path, n_keep)
    # Align by timestamp
    common = set(base_rets.keys()) & set(other_rets.keys())
    if len(common) < 10:
        return (path, float('nan'))
    common = sorted(common)
    x = np.array([base_rets[t] for t in common])
    y = np.array([other_rets[t] for t in common])
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return (path, float('nan'))
    return (path, pearson(x[valid], y[valid]))


def main():
    parser = ArgumentParser()
    parser.add_argument('base', help='Base PnL file')
    parser.add_argument('target', help='Target PnL file or directory')
    parser.add_argument('-d', '--days', type=int, default=365 * 2)
    parser.add_argument('-j', '--jobs', type=int, default=8)
    args = parser.parse_args()

    # Detect daily_factor from base file
    base_rets = read_pnl_rets(args.base)
    times = set(t.split('_')[1] if '_' in t else '' for t in base_rets.keys())
    daily_factor = max(1, len(times))
    n_keep = args.days * daily_factor

    base_rets = read_pnl_rets(args.base, n_keep)

    # Collect target paths
    if os.path.isdir(args.target):
        targets = [os.path.join(args.target, f) for f in os.listdir(args.target)
                   if os.path.isfile(os.path.join(args.target, f)) and f != os.path.basename(args.base)]
    else:
        targets = [args.target]

    # Compute correlations in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(compute_corr, base_rets, p, n_keep) for p in targets]
        results = [f.result() for f in futures]

    # Sort by correlation
    results.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 999)

    for path, corr in results:
        name = os.path.basename(path)
        if np.isnan(corr):
            print(f'{name:<30} nan')
        else:
            print(f'{name:<30} {corr:+.4f}')


if __name__ == '__main__':
    main()
