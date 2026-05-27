#!/usr/bin/env python3
"""Verify universe_mask and klines data validity in cache."""
import os
import sys
import numpy as np
import yaml
import msgpack
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    cache_dir = '/home/fengxh/csimcache/5m'
    univ_path = os.path.join(cache_dir, '__universe')
    universe_dir = os.path.join(cache_dir, 'universe')
    klines_dir = os.path.join(cache_dir, 'klines')

    # Load dates
    dates_path = os.path.join(univ_path, 'dates.mpk')
    if not os.path.exists(dates_path):
        print(f"Error: {dates_path} not found")
        return
    with open(dates_path, 'rb') as f:
        dates_int = msgpack.unpackb(f.read(), raw=False)
    dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in dates_int]
    n_dates = len(dates)
    bpd = 288

    # Load universe_mask meta and data
    mask_meta_path = os.path.join(universe_dir, '.meta')
    if not os.path.exists(mask_meta_path):
        print(f"Error: {mask_meta_path} not found. Run cache generation first.")
        return
    with open(mask_meta_path, 'r') as f:
        mask_meta = yaml.safe_load(f)
    mask_shape = tuple(mask_meta['fields']['universe_mask']['shape'])
    mask_path = os.path.join(universe_dir, 'universe_mask')
    mask = np.memmap(mask_path, dtype=bool, mode='r', shape=mask_shape)

    # Load klines meta and data
    klines_meta_path = os.path.join(klines_dir, '.meta')
    with open(klines_meta_path, 'r') as f:
        klines_meta = yaml.safe_load(f)
    
    close_shape = tuple(klines_meta['fields']['close']['shape'])
    close_path = os.path.join(klines_dir, 'close')
    close = np.memmap(close_path, dtype=np.float32, mode='r', shape=close_shape)
    
    qvol_shape = tuple(klines_meta['fields']['quote_volume']['shape'])
    qvol_path = os.path.join(klines_dir, 'quote_volume')
    qvol = np.memmap(qvol_path, dtype=np.float32, mode='r', shape=qvol_shape)

    print(f"Loaded mask shape: {mask.shape}")
    print(f"Loaded close shape: {close.shape}")
    print(f"Loaded qvol shape: {qvol_shape}")
    print(f"Total dates: {n_dates}")
    
    start_2021 = datetime(2021, 1, 1).date()
    try:
        start_di = next(i for i, d in enumerate(dates) if d >= start_2021)
    except StopIteration:
        print("No dates after 2021 found.")
        return

    errors = 0
    checked = 0
    # Check every day
    for di in range(start_di, n_dates):
        # Check mask count
        day_mask = mask[di * bpd, :]
        n_valid = day_mask.sum()
        
        if n_valid != 50:
            print(f"  [WARN] Date {dates[di]}: mask count = {n_valid} (expected 50)")
            errors += 1
            continue
            
        # Check klines data for valid symbols
        valid_indices = np.where(day_mask)[0]
        # Check first bar of the day
        idx = di * bpd
        close_vals = close[idx, valid_indices]
        qvol_vals = qvol[idx, valid_indices]
        
        nan_close = np.isnan(close_vals).sum()
        nan_qvol = np.isnan(qvol_vals).sum()
        
        if nan_close > 0 or nan_qvol > 0:
            print(f"  [WARN] Date {dates[di]}: {nan_close} NaN in close, {nan_qvol} NaN in qvol among valid symbols")
            errors += 1
        else:
            checked += 1
            
    if errors == 0:
        print(f"\nSUCCESS: Checked {checked} days after 2021. All have exactly 50 valid symbols with valid klines data.")
    else:
        print(f"\nFAILED: Found {errors} issues.")

if __name__ == '__main__':
    main()
