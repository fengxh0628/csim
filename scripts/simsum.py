#!/usr/bin/env python3
"""simsum - Summarize PnL results (adapted from fsimpy).

Usage:
  python simsum.py pnl/combo_main
  python simsum.py pnl/combo_main -t M          # monthly breakdown
  python simsum.py pnl/combo_main -s 20260401 -e 20260430
  python simsum.py pnl/combo_main -b pnl/mom5d  # marginal vs base
"""
import pandas as pd
from datetime import datetime
from math import sqrt
from argparse import ArgumentParser

ANNUAL_FACTOR = 365  # crypto trades every day


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pnlfile', type=str)
    parser.add_argument('-s', '--startdate', type=str)
    parser.add_argument('-e', '--enddate', type=str)
    parser.add_argument('-t', '--resolution', type=str, default='Y')
    parser.add_argument('-b', '--base', type=str, default=None, help='Base pnl file for marginal analysis')
    return parser.parse_args()


def get_stats(pnl):
    pnl = pnl[pnl['long'] > 0]
    if len(pnl) == 0:
        return None
    daily_factor = len(set(pnl.index.time))
    start_dt, end_dt = pnl.index[0].strftime('%Y%m%d'), pnl.index[-1].strftime('%Y%m%d')

    hold_val = pnl['hold_val'] / daily_factor
    pnl_val = pnl['pnl'] / daily_factor
    ret_val = pnl['ret'] / daily_factor

    long = pnl['long'].mean()
    short = pnl['short'].mean()
    tvr = pnl['trade_val'].sum() / hold_val.sum() * 100
    ir = ret_val.mean() / ret_val.std() * sqrt(ANNUAL_FACTOR) if ret_val.std() > 0 else 0
    ic = pnl['ic'].mean()
    icir = ic / pnl['ic'].std() if pnl['ic'].std() > 0 else 0
    bp_mgn = pnl_val.sum() / pnl['trade_val'].sum() * 10000 if pnl['trade_val'].sum() > 0 else 0
    ret = ret_val.mean() * 100 * ANNUAL_FACTOR
    pwin = (pnl['pnl'] > 0.).sum() / len(pnl) * 100

    def _get_dd(df):
        dd, cpnl, high, dh, ds, de = 0., 0., 0., df.index[0], df.index[0], df.index[0]
        for dt, val in df.iterrows():
            cpnl += val['pnl']
            if cpnl > high:
                dh = dt
                high = cpnl
            cdd = cpnl - high
            if cdd < dd:
                ds, de, dd = dh, dt, cdd
        dd = -dd / long * 100 if long > 0 else 0
        return dd, ds, de

    dd_max, dd_start, dd_end = 0, pnl.index[0], pnl.index[0]
    for _, gpnl in pnl.groupby(pnl.index.time):
        dd, ds, de = _get_dd(gpnl)
        if dd > dd_max:
            dd_max = dd
            dd_start = ds.strftime('%Y%m%d')
            dd_end = de.strftime('%Y%m%d')

    return f'{start_dt:>10}{end_dt:>10}{long:10.2f}{short:10.2f}{ret:10.2f}{tvr:10.2f}{ir:10.2f}{dd_max:10.2f}{dd_start:>10}{dd_end:>10}{bp_mgn:10.2f}{pwin:10.2f}{ic:10.4f}{icir:10.2f}'


def summary(pnl, resolution):
    res = ['%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s' %
        tuple('from to long short return tvr shrp dd dd_start dd_end bp_mgn winrate ic icir'.split(' '))]
    for _, gpnl in pnl.groupby(pnl.index.to_period(resolution)):
        s = get_stats(gpnl)
        if s is not None:
            res.append(s)
    res.append('')
    s = get_stats(pnl)
    if s is not None:
        res.append(s)
    res = '\n'.join(res)
    return res


if __name__ == '__main__':
    args = parse_args()
    pnl = pd.read_csv(args.pnlfile, sep=' ', header=None, index_col=0, parse_dates=True,
                       date_format='%Y%m%d%H%M%S',
                       names='pnl long short ret hold_val trade_val lnum snum ic'.split())
    if args.startdate:
        start_dt = datetime.strptime(args.startdate, '%Y%m%d')
        pnl = pnl.loc[start_dt:]
    if args.enddate:
        end_dt = datetime.strptime(args.enddate, '%Y%m%d')
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
        pnl = pnl.loc[:end_dt]

    if args.base:
        base = pd.read_csv(args.base, sep=' ', header=None, index_col=0, parse_dates=True,
                           date_format='%Y%m%d%H%M%S',
                           names='pnl long short ret hold_val trade_val lnum snum ic'.split())
        if args.startdate:
            base = base.loc[start_dt:]
        if args.enddate:
            base = base.loc[:end_dt]

        common = pnl.index.intersection(base.index)
        delta = pd.DataFrame({
            'pnl': pnl.loc[common, 'pnl'] - base.loc[common, 'pnl'],
            'long': pnl.loc[common, 'long'],
            'short': pnl.loc[common, 'short'],
            'ret': pnl.loc[common, 'ret'] - base.loc[common, 'ret'],
            'hold_val': pnl.loc[common, 'hold_val'],
            'trade_val': pnl.loc[common, 'trade_val'],
            'lnum': pnl.loc[common, 'lnum'],
            'snum': pnl.loc[common, 'snum'],
            'ic': pnl.loc[common, 'ic'] - base.loc[common, 'ic'],
        }, index=common)

        print(summary(delta, args.resolution))
    else:
        print(summary(pnl, args.resolution))
