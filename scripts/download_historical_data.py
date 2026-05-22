#!/usr/bin/env python3
import os
import datetime
from binance_historical_data import BinanceDataDumper

# 加载 Universe 列表
script_dir = os.path.dirname(os.path.abspath(__file__))
universe_path = os.path.join(script_dir, '..', 'data', 'universe_top50_liquid.txt')
with open(universe_path, 'r') as f:
    symbols = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(symbols)} symbols from universe_top50_liquid.txt")

# 使用绝对路径 ~/binance_data
dump_path = os.path.expanduser("~/binance_data")

# 2. 下载资金费率相关数据 (Premium Index Klines)
print("=== 开始下载 Premium Index Klines (5m) 数据 ===")
funding_dumper = BinanceDataDumper(
    path_dir_where_to_dump=dump_path,
    asset_class="um",
    data_type="premiumIndexKlines",
    data_frequency="5m",
)
funding_dumper.dump_data(tickers=symbols)
print("Premium Index Klines 数据下载完成！\n")
