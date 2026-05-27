"""Moving average decay: smooths alpha over N intervals."""
import numpy as np
from collections import deque

from modules.opbase import OpBase
from core.universe import univbase
from core.utils import parse_duration
from lib import fast


class OpDecay(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.itvls_bars = parse_duration(cfg.get('itvls', '5d'))
        self.hist = deque()
        self._hist_idx = deque()

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        self.hist.append(alpha.copy())
        self._hist_idx.append(idx)
        while self._hist_idx and (idx - self._hist_idx[0]) >= self.itvls_bars:
            self._hist_idx.popleft()
            self.hist.popleft()

        data = np.array(list(self.hist), dtype=np.float32)
        v = np.any(np.isfinite(data), axis=0)
        alpha[v] = np.nanmean(data[:, v], axis=0)
        alpha[~v] = np.nan

    def archive(self) -> list[str]:
        return ['hist', '_hist_idx']


def create(cfg: dict) -> OpDecay:
    return OpDecay(cfg)
