"""Cross-sectional momentum alpha: N-period return up to current bar."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration
from lib import fast


class AlphaTest(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.data = self.dr.getdata(cfg.get('data', 'funding.rate'))
        self.itvls = parse_duration(cfg.get('lookback', '2d'))
        self.direction = int(cfg.get('direction', 1))

    def generate(self, idx: int) -> None:
        didx = idx - self.delay
        valid = self.get_valid(idx)
        start_idx = didx - self.itvls
        end_idx = didx + 1

        data = self.data[start_idx:end_idx, valid]

        self.alpha[valid] = fast.niomean(data) / fast.niostd(data) * self.direction


def create(cfg: dict) -> AlphaTest:
    return AlphaTest(cfg)
