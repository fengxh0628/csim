"""Winsorize: cap outliers at +/- N sigma."""
import numpy as np

from modules.opbase import OpBase
from lib import fast


class OpWinsorize(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.sigma = cfg.get('sigma', 3.0)

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        alpha[:] = fast.winsorize(alpha, self.sigma)


def create(cfg: dict) -> OpWinsorize:
    return OpWinsorize(cfg)
