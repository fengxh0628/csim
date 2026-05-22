"""Moving average decay: smooths alpha over N intervals."""
import numpy as np
from collections import deque

from modules.opbase import OpBase
from core.universe import univbase
from lib import fast


class OpDecay(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.itvls = cfg.get('itvls', 5)
        self.hist = deque(maxlen=self.itvls)

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        self.hist.append(alpha.copy())
        data = np.array(list(self.hist), dtype=np.float32)
        v = np.any(np.isfinite(data), axis=0)
        alpha[v] = np.nanmean(data[:, v], axis=0)
        alpha[~v] = np.nan

    def archive(self) -> list[str]:
        return ['hist']


def create(cfg: dict) -> OpDecay:
    return OpDecay(cfg)
