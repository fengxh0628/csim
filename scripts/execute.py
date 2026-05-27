#!/usr/bin/env python3
"""Execute target positions on Binance USDT-M perpetual futures.

Usage:
  python execute.py --positions signal.csv
  python execute.py --positions signal.csv --dry-run
  python execute.py --positions signal.csv --max-notional 2500

Supports CSV format (opdump output).

Workflow:
  1. Read target positions (CSV)
  2. Normalize weights: long side = 0.5, short side = -0.5
  3. Query current positions from Binance
  4. Compute delta (target - current)
  5. Execute: limit orders in parallel -> wait -> cancel unfilled -> market sweep
  6. Log execution results

Requires: pip install python-binance
Environment variables: BINANCE_API_KEY, BINANCE_API_SECRET
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from binance.client import Client
    from binance.enums import *
except ImportError:
    print("ERROR: pip install python-binance")
    sys.exit(1)


class Executor:

    def __init__(self, api_key: str, api_secret: str, dry_run: bool = False):
        self.client = Client(api_key, api_secret)
        self.dry_run = dry_run
        self.log_entries = []

    def get_current_positions(self) -> dict[str, float]:
        """Query current perpetual futures positions from Binance."""
        positions = {}
        self._account = self.client.futures_account()
        for pos in self._account['positions']:
            amt = float(pos['positionAmt'])
            if amt != 0:
                positions[pos['symbol']] = amt
        return positions

    def get_account_balance(self) -> float:
        """Get total wallet balance (USDT) from futures account."""
        if not hasattr(self, '_account'):
            self._account = self.client.futures_account()
        return float(self._account['totalWalletBalance'])

    def get_mark_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get current mark prices for symbols."""
        prices = {}
        tickers = self.client.futures_mark_price()
        for t in tickers:
            if t['symbol'] in symbols:
                prices[t['symbol']] = float(t['markPrice'])
        return prices

    def get_symbol_info(self, symbol: str) -> dict:
        """Get trading rules (tick size, lot size, min notional)."""
        info = self.client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                filters = {f['filterType']: f for f in s['filters']}
                return {
                    'pricePrecision': s['pricePrecision'],
                    'quantityPrecision': s['quantityPrecision'],
                    'tickSize': float(filters['PRICE_FILTER']['tickSize']),
                    'stepSize': float(filters['LOT_SIZE']['stepSize']),
                    'minQty': float(filters['LOT_SIZE']['minQty']),
                    'minNotional': float(filters.get('MIN_NOTIONAL', {}).get('notional', 5)),
                }
        return None

    def round_price(self, price: float, tick_size: float) -> float:
        precision = len(str(tick_size).rstrip('0').split('.')[-1])
        return round(round(price / tick_size) * tick_size, precision)

    def round_qty(self, qty: float, step_size: float) -> float:
        precision = len(str(step_size).rstrip('0').split('.')[-1])
        return round(int(qty / step_size) * step_size, precision)

    def compute_deltas(self, target_positions: dict[str, float],
                       current_positions: dict[str, float],
                       prices: dict[str, float],
                       max_notional: float) -> dict[str, float]:
        """Compute position deltas in coin units.

        target_positions: {symbol: weight} where weight is fraction of booksize
        current_positions: {symbol: coin_amount} from exchange
        max_notional: total notional in USDT
        """
        # Convert target weights to coin amounts
        target_coins = {}
        for sym, weight in target_positions.items():
            if sym in prices and prices[sym] > 0:
                target_coins[sym] = (weight * max_notional) / prices[sym]

        # Compute deltas
        all_symbols = set(list(target_coins.keys()) + list(current_positions.keys()))
        deltas = {}
        for sym in all_symbols:
            target = target_coins.get(sym, 0.0)
            current = current_positions.get(sym, 0.0)
            delta = target - current
            if abs(delta) > 0:
                deltas[sym] = delta

        return deltas

    def execute(self, target_positions: dict[str, float], max_notional: float,
                limit_offset_bps: float = 2.0, timeout_seconds: int = 600):
        """Execute full rebalance.

        All limit orders placed in parallel, then wait once, then sweep unfilled.
        limit_offset_bps: offset from mark price for limit orders (maker)
        """
        print(f'[execute] Target: {len(target_positions)} positions, notional=${max_notional:.0f}')

        # Get current state
        current = self.get_current_positions()
        all_symbols = list(set(list(target_positions.keys()) + list(current.keys())))
        prices = self.get_mark_prices(all_symbols)

        print(f'[execute] Current: {len(current)} positions')

        # Compute deltas
        deltas = self.compute_deltas(target_positions, current, prices, max_notional)

        # Filter small deltas (< $5 notional)
        deltas = {sym: d for sym, d in deltas.items()
                  if sym in prices and abs(d * prices[sym]) > 5}

        print(f'[execute] Deltas: {len(deltas)} orders to place')

        if not deltas:
            return []

        # Phase 1: Place all limit orders in parallel
        pending_orders = []
        results = []

        for sym, delta in deltas.items():
            side = 'BUY' if delta > 0 else 'SELL'
            price = prices[sym]
            offset = price * limit_offset_bps / 10000
            limit_price = price - offset if side == 'BUY' else price + offset

            info = self.get_symbol_info(sym)
            if info is None:
                results.append({'symbol': sym, 'status': 'error', 'msg': 'symbol not found'})
                continue

            qty = self.round_qty(abs(delta), info['stepSize'])
            if qty < info['minQty']:
                results.append({'symbol': sym, 'status': 'skipped', 'msg': f'qty {qty} < minQty {info["minQty"]}'})
                continue

            limit_price = self.round_price(limit_price, info['tickSize'])
            notional = qty * price
            print(f'  {sym:12} {side:4} qty={qty:.4f} ~${notional:.0f} @ {limit_price:.4f}')

            if self.dry_run:
                results.append({'symbol': sym, 'side': side, 'qty': qty,
                                'price': limit_price, 'status': 'dry_run'})
                continue

            try:
                order = self.client.futures_create_order(
                    symbol=sym, side=side, type='LIMIT',
                    timeInForce='GTC', quantity=qty, price=str(limit_price),
                )
                pending_orders.append((sym, order['orderId'], qty, side, info))
            except Exception as e:
                results.append({'symbol': sym, 'status': 'error', 'msg': str(e)})

        if self.dry_run or not pending_orders:
            self.log_entries.extend(results)
            return results

        # Phase 2: Wait for fills
        print(f'\n[execute] Waiting {timeout_seconds}s for {len(pending_orders)} limit orders...')
        time.sleep(timeout_seconds)

        # Phase 3: Check fills, cancel unfilled, market sweep
        for sym, order_id, qty, side, info in pending_orders:
            try:
                status = self.client.futures_get_order(symbol=sym, orderId=order_id)
            except Exception as e:
                results.append({'symbol': sym, 'status': 'error', 'msg': str(e)})
                continue

            filled_qty = float(status['executedQty'])

            if status['status'] == 'FILLED':
                results.append({'symbol': sym, 'side': side, 'qty': qty,
                                'filled': filled_qty, 'status': 'filled'})
                continue

            # Cancel unfilled/partial
            if status['status'] not in ('CANCELED', 'EXPIRED'):
                try:
                    self.client.futures_cancel_order(symbol=sym, orderId=order_id)
                except Exception:
                    pass

            # Market sweep remainder
            remaining = self.round_qty(qty - filled_qty, info['stepSize'])
            if remaining >= info['minQty']:
                try:
                    self.client.futures_create_order(
                        symbol=sym, side=side, type='MARKET', quantity=remaining,
                    )
                    filled_qty = qty
                except Exception as e:
                    results.append({'symbol': sym, 'side': side, 'status': 'partial',
                                    'filled': filled_qty, 'remaining': remaining, 'msg': str(e)})
                    continue

            results.append({'symbol': sym, 'side': side, 'qty': qty,
                            'filled': filled_qty, 'status': 'filled'})

        self.log_entries.extend(results)
        return results

    def save_log(self, log_dir: str):
        """Save execution log."""
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'exec_{ts}.json')
        with open(log_file, 'w') as f:
            json.dump(self.log_entries, f, indent=2)
        print(f'[execute] Log saved: {log_file}')


def load_positions(path: str) -> dict[str, float]:
    """Load positions from CSV file and normalize weights.

    CSV format (opdump): header ',alpha', rows 'SYMBOL,value'

    Returns normalized weights: long side sums to 0.5, short side to -0.5.
    """
    raw = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                sym = row[0].strip()
                val = float(row[1])
                raw[sym] = val

    long_scores = {sym: v for sym, v in raw.items() if v > 0}
    short_scores = {sym: v for sym, v in raw.items() if v < 0}

    long_sum = sum(long_scores.values())
    short_sum = sum(abs(v) for v in short_scores.values())

    weights = {}
    if long_sum > 0:
        for sym, v in long_scores.items():
            weights[sym] = 0.5 * v / long_sum
    if short_sum > 0:
        for sym, v in short_scores.items():
            weights[sym] = -0.5 * abs(v) / short_sum

    return weights


def main():
    parser = argparse.ArgumentParser(description='Execute target positions on Binance')
    parser.add_argument('--positions', required=True, help='Path to positions file (CSV)')
    parser.add_argument('--max-notional', type=float, default=0, help='Total notional in USDT (0=use account balance)')
    parser.add_argument('--dry-run', action='store_true', help='Print orders without executing')
    parser.add_argument('--timeout', type=int, default=600, help='Seconds to wait for limit fill (default 10min)')
    parser.add_argument('--offset-bps', type=float, default=2.0, help='Limit order offset from mark price (bps)')
    parser.add_argument('--leverage', type=int, default=3, help='Leverage to set for all symbols (default 3)')
    args = parser.parse_args()

    target = load_positions(args.positions)

    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    if not api_key or not api_secret:
        if not args.dry_run:
            print("ERROR: Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
            sys.exit(1)
        api_key = api_secret = 'dummy'

    executor = Executor(api_key, api_secret, dry_run=args.dry_run)

    max_notional = args.max_notional
    if max_notional <= 0 and not args.dry_run:
        executor.get_current_positions()
        max_notional = executor.get_account_balance()
        print(f'[execute] Account balance: ${max_notional:.2f}')
    elif max_notional <= 0:
        max_notional = 10000

    if not args.dry_run:
        for sym in target.keys():
            try:
                executor.client.futures_change_leverage(symbol=sym, leverage=args.leverage)
            except Exception:
                pass
        print(f'[execute] Leverage set to {args.leverage}x')

    print(f'=== Execute: {args.positions} ({"DRY RUN" if args.dry_run else "LIVE"}) ===')
    results = executor.execute(
        target_positions=target,
        max_notional=max_notional,
        limit_offset_bps=args.offset_bps,
        timeout_seconds=args.timeout,
    )

    filled = sum(1 for r in results if r.get('status') == 'filled')
    skipped = sum(1 for r in results if r.get('status') == 'skipped')
    errors = sum(1 for r in results if r.get('status') == 'error')
    print(f'\n[execute] Done: {filled} filled, {skipped} skipped, {errors} errors')

    csim_root = Path(__file__).resolve().parent.parent
    executor.save_log(str(csim_root / 'logs'))


if __name__ == '__main__':
    main()
