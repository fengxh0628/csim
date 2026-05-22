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
        # Tradeable mask (optional, from dmgrfilter)
        self._tradeable = self.dr.getdata('tradeable') if self.dr.has('tradeable') else None

    def get_valid(self, idx: int) -> np.ndarray:
        """Get bool mask of tradeable symbols at interval idx."""
        if self._tradeable is None:
            return np.ones(univbase.n_symbols, dtype=bool)
        return self._tradeable[idx, :]

    def generate(self, idx: int) -> None:
        """Generate alpha at interval index idx."""
        pass

    def archive(self) -> list[str]:
        return []
