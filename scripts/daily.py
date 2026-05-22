#!/usr/bin/env python3
"""Daily production script: download data, update cache, generate positions.

Usage:
  python3 scripts/daily.py --config prod.yml

Cron (UTC 00:05 daily):
  5 0 * * * cd /home/xiaohangfeng/repos/cst/csim && python3 scripts/daily.py --config prod.yml >> logs/daily.log 2>&1

Steps:
  1. Download latest klines from Binance (incremental)
  2. Update memmap cache with new data
  3. Run alpha pipeline to generate target positions
  4. Output positions to positions/{date}.csv
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# Add csim root to path
CSIM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CSIM_ROOT))


def download_data(datapath: str, symbols: list[str], interval: str):
    """Download latest klines from Binance."""
    try:
        from binance_historical_data import BinanceDataDumper
    except ImportError:
        print("ERROR: pip install binance-historical-data")
        sys.exit(1)

    dumper = BinanceDataDumper(
        path_dir_where_to_dump=datapath,
        asset_class="um",
        data_type="klines",
        data_frequency=interval,
    )

    today = datetime.date.today()
    # Download last 3 days to handle any gaps
    start = today - datetime.timedelta(days=3)

    print(f'[1/3] Downloading {interval} klines ({start} -> {today})...')
    dumper.dump_data(
        tickers=symbols,
        date_start=start,
        date_end=today,
        is_to_update_existing=True,
    )
    print('  Download complete.')


def update_cache(config_path: str) -> dict:
    """Update memmap cache and return config dict."""
    import yaml

    # Fresh module state
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith(('core.', 'modules.', 'lib.')):
            del sys.modules[mod_name]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Override enddate to today, mode to write
    today = datetime.date.today()
    cfg['constants']['enddate'] = int(today.strftime('%Y%m%d'))
    cfg['constants']['mode'] = 'w'
    cfg['constants']['verbose'] = False

    # Write temp config
    tmp_config = '/tmp/csim_daily_config.yml'
    with open(tmp_config, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    from core.sim_config import init_simcfg
    simcfg = init_simcfg(tmp_config)

    from core.universe import init_universe
    init_universe()

    print('[2/3] Updating memmap cache...')
    for module_id, (module, mcfg) in simcfg.modules.items():
        if mcfg.get('handler') == 'datahandler':
            module.create(mcfg)
    print('  Cache updated.')

    return cfg


def generate_positions(cfg: dict):
    """Run alpha pipeline on latest data, output target positions."""
    from core.sim_config import simcfg
    from core.universe import univbase
    from core.sim_node import create_node
    from core.tree_utils import preorder_iter
    from lib import fast
    import numpy as np

    # Build portfolio
    portfolio_cfg = cfg['portfolio']
    portfolio = create_node(portfolio_cfg)
    for node in preorder_iter(portfolio):
        node.module.children = node.children

    # Find the latest interval with valid data
    from core.data_registry import get_data_registry
    dr = get_data_registry()
    close_data = dr.getdata('itvl.close')
    # Search backwards from end for first row with any valid data
    idx = close_data.shape[0] - 1
    while idx > 0 and not np.any(np.isfinite(close_data[idx, :])):
        idx -= 1

    print(f'[3/3] Generating positions at idx={idx} ({univbase.itvl_to_timestamp(idx)})...')

    # Run portfolio tree
    portfolio.run(idx)

    # Scale to booksize
    booksize = cfg['constants']['booksize']
    alpha = fast.nioscale(portfolio.module.alpha, booksize)

    # Output positions
    today = datetime.date.today()
    pos_dir = os.path.join(str(CSIM_ROOT), 'positions')
    os.makedirs(pos_dir, exist_ok=True)

    positions = {}
    for si, symbol in enumerate(univbase.symbols):
        if np.isfinite(alpha[si]) and abs(alpha[si]) > 1e-8:
            positions[symbol] = float(alpha[si])

    # Save as CSV
    pos_file = os.path.join(pos_dir, f'{today.strftime("%Y%m%d")}.csv')
    with open(pos_file, 'w') as f:
        f.write('symbol,weight\n')
        for sym, wt in sorted(positions.items(), key=lambda x: -abs(x[1])):
            f.write(f'{sym},{wt:.6f}\n')

    # Save as JSON (easier to parse programmatically)
    json_file = os.path.join(pos_dir, f'{today.strftime("%Y%m%d")}.json')
    with open(json_file, 'w') as f:
        json.dump({
            'date': today.strftime('%Y%m%d'),
            'timestamp': univbase.itvl_to_timestamp(idx),
            'booksize': booksize,
            'n_long': sum(1 for v in positions.values() if v > 0),
            'n_short': sum(1 for v in positions.values() if v < 0),
            'positions': positions,
        }, f, indent=2)

    # Print summary
    long_pos = {k: v for k, v in positions.items() if v > 0}
    short_pos = {k: v for k, v in positions.items() if v < 0}
    print(f'  Positions: {len(long_pos)}L / {len(short_pos)}S')
    print(f'  Top long:  {sorted(long_pos.items(), key=lambda x: -x[1])[:5]}')
    print(f'  Top short: {sorted(short_pos.items(), key=lambda x: x[1])[:5]}')
    print(f'  Output: {pos_file}')


def main():
    parser = argparse.ArgumentParser(description='Daily: download + cache + positions')
    parser.add_argument('--config', required=True, help='Path to config.yml')
    parser.add_argument('--skip-download', action='store_true', help='Skip download step')
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    os.makedirs(os.path.join(str(CSIM_ROOT), 'logs'), exist_ok=True)
    print(f'=== csim daily run: {datetime.date.today()} ===')

    # Load config for symbols/datapath
    import yaml
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    # Step 1: Download
    if not args.skip_download:
        datapath = None
        for m in raw_cfg['modules']:
            if m.get('handler') == 'datahandler' and 'datapath' in m:
                datapath = m['datapath']
                break
        symbols = raw_cfg['universe']['symbols']
        interval = raw_cfg['constants'].get('interval', '5m')
        download_data(datapath, symbols, interval)

    # Step 2: Update cache
    cfg = update_cache(config_path)

    # Step 3: Generate positions
    generate_positions(cfg)

    print('=== Done ===')


if __name__ == '__main__':
    main()
