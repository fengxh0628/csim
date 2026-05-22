"""Rank-based power transform: maps alpha to uniform ranks then applies power."""
import numpy as np

from modules.opbase import OpBase
from lib import fast


class OpPower(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.exp = cfg.get('exp', 1.0)

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        alpha[:] = fast.power(alpha, self.exp)


def create(cfg: dict) -> OpPower:
    return OpPower(cfg)
