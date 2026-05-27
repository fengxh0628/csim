"""Exponential moving average decay."""
import numpy as np

from modules.opbase import OpBase
from core.utils import parse_duration


class OpEmaDecay(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.itvls_bars = parse_duration(cfg.get('itvls', '5d'))
        self.old_values = None
        self._prev_idx = None

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        if self.old_values is None:
            self.old_values = np.full_like(alpha, np.nan)
            self._prev_idx = idx

        v = np.isfinite(alpha)
        elapsed = idx - self._prev_idx if self._prev_idx is not None else 1
        elapsed = max(elapsed, 1)
        N = self.itvls_bars / elapsed
        factor = 2.0 / (N + 1.0)
        new_values = np.where(
            np.isnan(self.old_values[v]),
            alpha[v],
            factor * alpha[v] + (1.0 - factor) * self.old_values[v]
        )
        alpha[v] = new_values
        self.old_values[v] = new_values
        self._prev_idx = idx

    def archive(self) -> list[str]:
        return ['old_values', '_prev_idx']


def create(cfg: dict) -> OpEmaDecay:
    return OpEmaDecay(cfg)
