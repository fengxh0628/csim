"""PnL statistics for crypto strategy.

Entry/exit price: VWAP over [idx+1, idx+exec_bars] (simulates execution slippage).
Hold return: from entry VWAP to next rebalance's entry VWAP.
exec_bars is configurable (default: 60 bars = 1 hour for 1m data).
"""
import numpy as np
from scipy.stats import spearmanr

from modules.statsbase import StatsBase
from core.universe import univbase
from core.sim_config import simcfg


class StatsSimple(StatsBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.iclose = self.dr.getdata('close')
        self.ivol = self.dr.getdata('volume')
        self.iqvol = self.dr.getdata('quote_volume')
        self.trading_cost = float(cfg.get('trading_cost', 0.))
        # Execution window: bars after signal to compute entry/exit VWAP
        # Default exec_bars: 30 min worth of bars
        self.exec_bars = int(cfg.get('exec_bars', 30 // univbase.interval_minutes))
        # Hold period: auto-detect from rebalance config, or override manually
        self.hold_bars = int(cfg.get('hold_bars', self._get_hold_bars()))
        # Funding rate cost (optional, if funding data loaded)
        self.funding = self.dr.getdata('funding.rate') 

    def _get_hold_bars(self) -> int:
        """Infer hold_bars from rebalance_times config."""
        rebalance_times = simcfg.constants.get('rebalance_times', ['00:00'])
        if isinstance(rebalance_times, str):
            rebalance_times = [rebalance_times]
        n_per_day = len(rebalance_times)
        return univbase.bars_per_day // n_per_day

    def _vwap(self, start: int, end: int) -> np.ndarray:
        """Compute VWAP over [start, end) for all symbols."""
        end = min(end, univbase.n_intervals)
        if start >= end:
            return self.iclose[start, :] if start < univbase.n_intervals else np.full(univbase.n_symbols, np.nan)
        qv = np.nansum(self.iqvol[start:end, :], axis=0)
        vol = np.nansum(self.ivol[start:end, :], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = np.where(vol > 0, qv / vol, np.nan)
        return vwap

    def calculate(self, idx: int, alpha: np.ndarray, prevalpha: np.ndarray):
        # Entry VWAP: bars [idx+1, idx+1+exec_bars)
        entry_start = idx + 1
        entry_end = entry_start + self.exec_bars
        if entry_end >= univbase.n_intervals:
            return None

        # Exit VWAP: bars [idx+hold_bars+1, idx+hold_bars+1+exec_bars)
        exit_start = idx + self.hold_bars + 1
        exit_end = exit_start + self.exec_bars
        if exit_end >= univbase.n_intervals:
            return None

        entry_vwap = self._vwap(entry_start, entry_end)
        exit_vwap = self._vwap(exit_start, exit_end)

        valid_price = (entry_vwap > 0) & (exit_vwap > 0) & np.isfinite(entry_vwap) & np.isfinite(exit_vwap)
        true_rets = np.full(len(alpha), np.nan, dtype=np.float32)
        true_rets[valid_price] = exit_vwap[valid_price] / entry_vwap[valid_price] - 1.0

        # PnL
        pnl = np.nansum(alpha * true_rets)

        # Transaction costs and funding
        # delta = alpha - prevalpha: fill NaN→0 so entering/exiting positions count
        a = np.where(np.isfinite(alpha), alpha, 0.0)
        p = np.where(np.isfinite(prevalpha), prevalpha, 0.0)
        trade_val = np.nansum(np.abs(a - p))
        if self.trading_cost > 0:
            pnl -= trade_val * self.trading_cost

        # Funding is charged every 8h. The 5m data has 96 bars per 8h interval.
        # Take the average funding rate per 8h interval to avoid overcounting.
        bars_per_8h = 8 * 60 // univbase.interval_minutes
        fwd_end = min(idx + self.hold_bars, self.funding.shape[0])
        fr_slice = self.funding[idx:fwd_end, :]
        # Reshape to (n_periods, bars_per_8h, n_symbols) and average within each period
        n_periods = fr_slice.shape[0] // bars_per_8h
        if n_periods > 0:
            fr_reshaped = fr_slice[:n_periods * bars_per_8h, :].reshape(n_periods, bars_per_8h, fr_slice.shape[1])
            fr_avg = np.nanmean(fr_reshaped, axis=1)  # (n_periods, n_symbols)
            fr_sum = np.nansum(fr_avg, axis=0)  # sum over 8h periods
            funding_cost = np.nansum(alpha * fr_sum)
            pnl -= funding_cost

        # Position stats
        lnum = int((alpha > 0).sum())
        snum = int((alpha < 0).sum())
        long_val = np.nansum(np.where(alpha > 0, alpha, 0.0))
        short_val = np.nansum(np.where(alpha < 0, alpha, 0.0))
        hold_val = np.nansum(np.abs(alpha))
        ret = pnl / hold_val if hold_val > 0 else 0.0

        # IC (rank correlation alpha vs forward return)
        valid_mask = np.isfinite(alpha) & np.isfinite(true_rets)
        if valid_mask.sum() >= 3:
            ic = spearmanr(alpha[valid_mask], true_rets[valid_mask])[0]
        else:
            ic = np.nan

        ts_str = univbase.itvl_to_timestamp(idx)
        return (ts_str, pnl, long_val, short_val, ret, hold_val, trade_val, lnum, snum, ic)


def create(cfg: dict) -> StatsSimple:
    return StatsSimple(cfg)
