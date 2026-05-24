"""Cross-sectional momentum alpha: N-period return up to current bar."""
import numpy as np

from modules.alphabase import AlphaBase
from core.universe import univbase
from core.utils import parse_duration
from lib import fast


class AlphaTest(AlphaBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.data = self.dr.getdata(cfg.get('data', ''))
        self.itvls = parse_duration(cfg.get('lookback', '1d'))
        self.direction = int(cfg.get('direction', 1))

    def generate(self, idx: int) -> None:
        valid = self.get_valid(idx)
        start_idx = idx + 1 - self.itvls
        end_idx = idx + 1

        data = self.data[start_idx:end_idx, valid]
        data = data[1:] / data[:-1] - 1.
        inddata = np.tile(np.nanmean(data, axis=1, keepdims=True), (1, valid.sum()))

        self.alpha[valid] = fast.niocorr(inddata, data) * self.direction


def create(cfg: dict) -> AlphaTest:
    return AlphaTest(cfg)
