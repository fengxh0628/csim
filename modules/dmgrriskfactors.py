"""Risk factor exposures for neutralization.

Computes rolling factor exposures for each symbol. Currently:
  - beta: rolling beta to BTC

Registered fields:
  risk.beta   (n_dates, n_symbols) float32

Future factors can be added here (e.g., size, volatility, momentum factor).
"""
import numpy as np

from modules.dmgrbase import DmgrBase
from core.universe import univbase
from lib import fast
from core.sim_config import simcfg


class DmgrRiskFactors(DmgrBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.lookback = int(cfg.get('lookback', 60))
        self.btc_idx = None
        for i, sym in enumerate(univbase.symbols):
            if sym == 'BTCUSDT':
                self.btc_idx = i
                break

    def dependencies(self) -> list[str]:
        return ['itvl.close']

    def initialize(self):
        super().initialize()
        self.add_itvl_data(['risk.beta'])

    def load_data(self):
        if self.mode == 'r':
            return

        iclose = self.dr.getdata('itvl.close')
        beta = self.dr.getdata('risk.beta')
        bpd = univbase.bars_per_day

        if self.btc_idx is None:
            print(f'[{self.id}] WARNING: BTCUSDT not in universe, skipping beta')
            return

        print(f'[{self.id}] Computing risk factors (lookback={self.lookback}d)...')

        for di in range(self.lookback, univbase.n_dates):
            # Sample close at end of each day (last bar)
            indices = [(d + 1) * bpd - 1 for d in range(di - self.lookback, di + 1)]
            close = iclose[indices, :]  # (lookback+1, n_sym)

            if close.shape[0] < 10:
                continue

            rets = close[1:, :] / close[:-1, :] - 1.0
            btc_rets = np.tile(rets[:, self.btc_idx:self.btc_idx+1], (1, univbase.n_symbols))
            day_beta = fast.niobeta(btc_rets, rets)

            # Forward-fill: same beta for all bars in this day
            sl = univbase.day_slice(di)
            beta[sl, :] = day_beta[np.newaxis, :]

        beta.flush()
        print(f'[{self.id}] Done')


def create(cfg: dict) -> DmgrRiskFactors:
    mgr = DmgrRiskFactors(cfg)
    mgr.initialize()
    return mgr
