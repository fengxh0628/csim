from datetime import datetime
from lib import fast
import numpy as np
from scipy.stats import rankdata
import os

from modules.opbase import OpBase
from core.sim_config import simcfg
from core.universe import univbase

ANNUAL_FACTOR = 365

class OpProfitAna(OpBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.close = self.dr.getdata('close')
        self.days = cfg.get('days', 1)
        self.delay = cfg.get('delay', 6)
        self.startdate = datetime.strptime(str(cfg.get('startdate', simcfg.constants.get('startdate'))), "%Y%m%d").date()
        self.enddate = datetime.strptime(str(cfg.get('enddate', simcfg.constants.get('enddate'))), "%Y%m%d").date()
        self.split = int(cfg.get('split', 10))
        self.hist = []

        if bool(cfg.get('dump', False)):
            os.makedirs('pnl', exist_ok=True)
            buffer_size = simcfg.constants.get('pnl_buffer_size', 8192)
            self.pfastream = open(f'pnl/{cfg.get("id", "unknown")}.pfa', 'w', buffering=buffer_size)


    def apply(self, idx, alpha):
        didx = univbase.itvl_to_didx(idx)
        tidx = idx % univbase.bars_per_day
        cur_date = univbase.dates[didx]

        if cur_date < self.startdate or cur_date > self.enddate:
            return

        n_bars = self.days * univbase.bars_per_day
        if idx + n_bars + 1 > univbase.n_intervals:
            return
        close = self.close[idx:idx + n_bars + 1, :]
        true_rets = (close[-1] / close[0] - 1.) / self.days
        true_rets[np.isinf(true_rets)] = 0.

        true_rets -= np.nanmean(true_rets)
        valid = np.isfinite(alpha)
        alpha_valid = alpha[valid]
        rets_valid = true_rets[valid]
        if len(alpha_valid) < self.split:
            return
        rank = ((rankdata(alpha_valid) - 1.) / len(alpha_valid) * self.split).astype(int)
        grouped_rets = [np.nanmean(rets_valid[rank == s]) for s in range(self.split)]
        if hasattr(self, 'pfastream'):
            self.pfastream.write(f'{cur_date.strftime("%Y%m%d")}{tidx:04d} {grouped_rets[-1]}\n')
        self.hist.append(grouped_rets)

        bars_per_day = univbase.bars_per_day
        if cur_date >= self.enddate and not hasattr(self, '_printed'):
            self._printed = True
            rets = fast.niomean(np.array(self.hist)) * 100 * ANNUAL_FACTOR
            results = ', '.join([f'{s + 1}: {r:6.2f}' for s, r in zip(range(self.split), rets)])
            print(f'    ProfitAna from {self.startdate} to {self.enddate}    {results}')
            if hasattr(self, 'pfastream'):
                self.pfastream.close()
                delattr(self, 'pfastream')


def create(cfg):
    return OpProfitAna(cfg)
