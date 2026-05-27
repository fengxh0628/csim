"""Cross-sectional momentum alpha: N-period return up to current bar."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration
from lib import fast


class AlphaTest(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.data1 = self.dr.getdata(cfg.get('data1', 'close'))
        self.data2 = self.dr.getdata(cfg.get('data2', 'close'))
        self.itvls = parse_duration(cfg.get('lookback', '3d'))
        self.direction = int(cfg.get('direction', 1))

    def generate(self, idx: int) -> None:
        didx = idx - self.delay
        valid = self.get_valid(idx)
        start_idx = didx - self.itvls
        end_idx = didx + 1

        data1 = self.data1[start_idx:end_idx, valid]
        data2 = self.data2[start_idx:end_idx, valid]

        data = data1[1:] / data1[:-1] - 1.

        self.alpha[valid] = fast.niostdup(data) / fast.niostddown(data) * self.direction


def create(cfg: dict) -> AlphaTest:
    return AlphaTest(cfg)
