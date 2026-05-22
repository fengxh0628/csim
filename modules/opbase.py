"""Base class for operations (alpha transformations)."""
import numpy as np

from core.data_registry import get_data_registry
from core.universe import univbase


class OpBase:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dr = get_data_registry()

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        pass

    def archive(self) -> list[str]:
        return []
