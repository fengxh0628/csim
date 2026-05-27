"""Base class for alpha modules."""
import numpy as np

from core.data_registry import get_data_registry
from core.universe import univbase


class AlphaBase:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dr = get_data_registry()
        self.alpha = np.full(univbase.n_symbols, np.nan, dtype=np.float32)
        self.children = []
        self.delay = int(cfg.get('delay', 1))
        # Tradeable mask (optional, from dmgrfilter or dmgruniverse)
        if self.dr.has('universe_mask'):
            self._tradeable = self.dr.getdata('universe_mask')
        elif self.dr.has('tradeable'):
            self._tradeable = self.dr.getdata('tradeable')
        else:
            self._tradeable = None

    def get_valid(self, idx: int) -> np.ndarray:
        """Get bool mask of tradeable symbols at interval idx."""
        if self._tradeable is None:
            return np.ones(univbase.n_symbols, dtype=bool)
        return self._tradeable[idx, :].copy()

    def generate(self, idx: int) -> None:
        """Generate alpha at interval index idx."""
        pass

    def archive(self) -> list[str]:
        return []
