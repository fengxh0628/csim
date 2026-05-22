"""Top N selection: long top N, short bottom N, weight proportional to alpha magnitude."""
import numpy as np

from modules.opbase import OpBase


class OpTopN(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.long_n = int(cfg.get('long_n', 5))
        self.short_n = int(cfg.get('short_n', 5))

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        v = np.isfinite(alpha)
        if v.sum() < self.long_n + self.short_n:
            return

        vals = alpha[v]
        indices = np.where(v)[0]
        order = np.argsort(vals)

        new_alpha = np.full_like(alpha, np.nan)

        # Bottom short_n → short, weight by alpha magnitude
        short_idx = indices[order[:self.short_n]]
        short_vals = np.abs(alpha[short_idx])
        short_sum = short_vals.sum()
        if short_sum > 0:
            new_alpha[short_idx] = -(short_vals / short_sum) * 0.5

        # Top long_n → long, weight by alpha magnitude
        long_idx = indices[order[-self.long_n:]]
        long_vals = np.abs(alpha[long_idx])
        long_sum = long_vals.sum()
        if long_sum > 0:
            new_alpha[long_idx] = (long_vals / long_sum) * 0.5

        alpha[:] = new_alpha


def create(cfg: dict) -> OpTopN:
    return OpTopN(cfg)
