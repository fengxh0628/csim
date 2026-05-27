"""Statistical sector clustering on daily returns.

Two methods:
  raw:        Spearman correlation on raw returns
  residual:   Spearman correlation on BTC-residualized returns (BTC = group 0)

Registered fields:
  group.k5     (n_intervals, n_symbols) float32 — raw returns, 5 groups
  group.k10    (n_intervals, n_symbols) float32 — raw returns, 10 groups
  group.k5r    (n_intervals, n_symbols) float32 — BTC-residualized, 6 groups (0=BTC, 1..5)
  group.k10r   (n_intervals, n_symbols) float32 — BTC-residualized, 11 groups (0=BTC, 1..10)
  -1 means insufficient data for that symbol

Config:
  lookback:  90       days of daily returns for correlation estimation
  (recomputed automatically when universe mask changes, i.e. each month)
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from modules.dmgrbase import DmgrBase
from core.universe import univbase


class DmgrGrouping(DmgrBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.lookback = int(cfg.get('lookback', 90))
        n_sym = univbase.n_symbols
        self._curr = {
            'k5': np.full(n_sym, -1, dtype=np.float32),
            'k10': np.full(n_sym, -1, dtype=np.float32),
            'k5r': np.full(n_sym, -1, dtype=np.float32),
            'k10r': np.full(n_sym, -1, dtype=np.float32),
        }
        self._last_month = None
        self._btc_idx = None
        for i, sym in enumerate(univbase.symbols):
            if sym == 'BTCUSDT':
                self._btc_idx = i
                break

    def dependencies(self) -> list[str]:
        return ['close', 'universe_mask']

    def initialize(self):
        super().initialize()
        self.add_itvl_data(['group.k5', 'group.k10', 'group.k5r', 'group.k10r'])

    def load_data(self):
        if self.mode == 'r':
            return

        iclose = self.dr.getdata('close')
        universe_mask = self.dr.getdata('universe_mask')
        fields = {
            'k5': self.dr.getdata('group.k5'),
            'k10': self.dr.getdata('group.k10'),
            'k5r': self.dr.getdata('group.k5r'),
            'k10r': self.dr.getdata('group.k10r'),
        }

        print(f'[{self.id}] Computing cluster groups '
              f'(lookback={self.lookback}d, aligned to universe months)...')

        n_dates = univbase.n_dates
        n_sym = univbase.n_symbols
        bpd = univbase.bars_per_day

        daily_close = np.full((n_dates, n_sym), np.nan, dtype=np.float64)
        for di in range(n_dates):
            sl = univbase.day_slice(di)
            daily_close[di] = iclose[sl.stop - 1, :]

        daily_rets = np.full_like(daily_close, np.nan)
        daily_rets[1:] = daily_close[1:] / daily_close[:-1] - 1.0
        daily_rets[0] = 0.0

        for di in range(n_dates):
            ym = (univbase.dates[di].year, univbase.dates[di].month)
            if di >= self.lookback and ym != self._last_month:
                itvl = di * bpd
                self._compute_clusters(daily_rets, di, universe_mask[itvl, :])
                self._last_month = ym

            sl = univbase.day_slice(di)
            for key, data in fields.items():
                data[sl, :] = self._curr[key][np.newaxis, :]

        for data in fields.values():
            data.flush()
        print(f'[{self.id}] Done')

    def _cluster_on_returns(self, returns: np.ndarray, use_common: bool = True):
        """Cluster symbols on return matrix (n_days, n_syms) using Spearman + Ward.

        Returns cluster labels 0..k-1, or None if insufficient data.
        """
        if use_common:
            common = np.all(np.isfinite(returns), axis=1)
            if common.sum() < 10:
                return None
            clean = returns[common, :]
        else:
            clean = returns

        ranks = np.argsort(np.argsort(clean, axis=0), axis=0).astype(np.float64) + 1.0

        corr = np.corrcoef(ranks.T)
        np.clip(corr, -1.0, 1.0, out=corr)

        dists = np.sqrt(2.0 * (1.0 - corr))
        Z = linkage(squareform(dists, checks=False), method='ward')

        labels5 = fcluster(Z, t=5, criterion='maxclust') - 1
        labels10 = fcluster(Z, t=10, criterion='maxclust') - 1
        return labels5.astype(np.float32), labels10.astype(np.float32)

    def _compute_clusters(self, daily_rets: np.ndarray, di: int, mask: np.ndarray):
        start = di - self.lookback
        ret_window = daily_rets[start:di, :]

        min_valid = max(20, self.lookback // 3)
        valid = np.sum(np.isfinite(ret_window), axis=0) >= min_valid

        # Only cluster symbols that are in the active universe
        valid = valid & mask

        if valid.sum() < 10:
            return

        self._curr['k5'][:] = -1
        self._curr['k10'][:] = -1
        self._curr['k5r'][:] = -1
        self._curr['k10r'][:] = -1

        vret = ret_window[:, valid]

        # --- raw returns clustering ---
        labels = self._cluster_on_returns(vret)
        if labels is not None:
            vidx = np.where(valid)[0]
            self._curr['k5'][vidx] = labels[0]
            self._curr['k10'][vidx] = labels[1]

        # --- BTC-residualized clustering ---
        if self._btc_idx is None or not valid[self._btc_idx]:
            return

        btc_idx = self._btc_idx

        # Exclude BTC from the set
        valid_no_btc = valid.copy()
        valid_no_btc[btc_idx] = False

        vret_nb = ret_window[:, valid_no_btc]
        common = np.all(np.isfinite(vret_nb), axis=1)
        if common.sum() < 10:
            return

        btc_ret = ret_window[common, btc_idx]
        resid = np.empty_like(vret_nb[common, :])
        for j in range(resid.shape[1]):
            y = vret_nb[common, j]
            num = np.sum(btc_ret * y)
            den = np.sum(btc_ret * btc_ret)
            if den > 0:
                beta = num / den
                resid[:, j] = y - beta * btc_ret
            else:
                resid[:, j] = y

        labels_r = self._cluster_on_returns(resid, use_common=False)
        if labels_r is not None:
            vidx_nb = np.where(valid_no_btc)[0]
            self._curr['k5r'][btc_idx] = 0.0
            self._curr['k10r'][btc_idx] = 0.0
            self._curr['k5r'][vidx_nb] = labels_r[0] + 1.0
            self._curr['k10r'][vidx_nb] = labels_r[1] + 1.0


def create(cfg: dict) -> DmgrGrouping:
    mgr = DmgrGrouping(cfg)
    mgr.initialize()
    return mgr
