"""Simple alpha: directly load a pre-computed data field as alpha."""
import numpy as np

from modules.alphabase import AlphaBase


class AlphaSimple(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.data = self.dr.getdata(cfg['dataname'])

    def generate(self, idx: int) -> None:
        if self.data.ndim == 1:
            self.alpha[:] = self.data[:]
        elif self.data.ndim == 2:
            # (itvls, num) - read at idx
            if idx < self.data.shape[0]:
                self.alpha[:] = self.data[idx, :]


def create(cfg: dict) -> AlphaSimple:
    return AlphaSimple(cfg)
