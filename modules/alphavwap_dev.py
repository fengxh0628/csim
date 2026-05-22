"""VWAP deviation alpha."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration


class AlphaVwapDev(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.lookback_bars = parse_duration(cfg.get('lookback', '1d'))
        self.iclose = self.dr.getdata('itvl.close')
        self.ivol = self.dr.getdata('itvl.volume')
        self.iqvol = self.dr.getdata('itvl.quote_volume')

    def generate(self, idx: int) -> None:
        if idx < self.lookback_bars:
            return

        sl = slice(idx - self.lookback_bars, idx)
        total_qvol = np.nansum(self.iqvol[sl, :], axis=0)
        total_vol = np.nansum(self.ivol[sl, :], axis=0)
        close = self.iclose[idx, :]

        valid = (total_vol > 0) & np.isfinite(close) & (close > 0)
        vwap = np.full(univbase.n_symbols, np.nan, dtype=np.float32)
        vwap[valid] = total_qvol[valid] / total_vol[valid]

        v = valid & (vwap > 0)
        self.alpha[v] = (close[v] - vwap[v]) / vwap[v]


def create(cfg: dict) -> AlphaVwapDev:
    return AlphaVwapDev(cfg)
