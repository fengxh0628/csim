"""Exponential moving average decay."""
import numpy as np

from modules.opbase import OpBase


class OpEmaDecay(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.itvls = cfg.get('itvls', 5)
        self.old_values = None

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        if self.old_values is None:
            self.old_values = np.full_like(alpha, np.nan)

        v = np.isfinite(alpha)
        factor = 2.0 / (self.itvls + 1.0)
        new_values = np.where(
            np.isnan(self.old_values[v]),
            alpha[v],
            factor * alpha[v] + (1.0 - factor) * self.old_values[v]
        )
        alpha[v] = new_values
        self.old_values[v] = new_values

    def archive(self) -> list[str]:
        return ['old_values']


def create(cfg: dict) -> OpEmaDecay:
    return OpEmaDecay(cfg)
