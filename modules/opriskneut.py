"""Risk neutralization: cross-sectional regression on a single factor, take residual.

Stack multiple opriskneut in config to neutralize multiple factors:
  operations:
    - { moduleId: opriskneut, factor: risk.beta }
    - { moduleId: opriskneut, factor: risk.size }
"""
import numpy as np

from modules.opbase import OpBase
from core.universe import univbase
from core.data_registry import get_data_registry
from lib import fast


class OpRiskNeut(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dr = get_data_registry()
        self.factor = self.dr.getdata(cfg.get('factor', 'risk.beta'))
        self.mode = int(cfg.get('mode', 0))  # 0=raw factor, 1=rank factor first

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        if idx >= self.factor.shape[0]:
            return

        fct = self.factor[idx, :].copy()

        # Valid: both alpha and factor finite
        v = np.isfinite(alpha) & np.isfinite(fct)
        if v.sum() < 3:
            return

        # Optional: rank the factor first (mode=1)
        if self.mode == 1:
            fct[v] = fast.power(fct[v], 1.0)

        # Fill NaN factor with mean for symbols that have alpha
        fct[~np.isfinite(fct) & np.isfinite(alpha)] = np.nanmean(fct[v])

        # Cross-sectional regression: alpha_i = a + b * factor_i + residual_i
        f = fct[v]
        a = alpha[v]
        f_mean = np.mean(f)
        a_mean = np.mean(a)
        f_demean = f - f_mean
        denom = np.dot(f_demean, f_demean)
        if denom <= 0:
            return
        b = np.dot(f_demean, a - a_mean) / denom
        intercept = a_mean - b * f_mean

        # Take residual
        alpha[v] = a - intercept - b * f


def create(cfg: dict) -> OpRiskNeut:
    return OpRiskNeut(cfg)
