"""Base class for statistics modules."""
from core.data_registry import get_data_registry


class StatsBase:

    def __init__(self, cfg: dict):
        self.dr = get_data_registry()
        self.scale = cfg.get('scale', True)

    def calculate(self, didx, alpha, prevalpha):
        pass
