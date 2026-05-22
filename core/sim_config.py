"""Configuration management for crypto simulation."""
import os
import yaml
import importlib.util
from typing import Optional, Any
from types import ModuleType


def _expandvars(cfg: Any):
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            cfg[k] = _expandvars(v)
    elif isinstance(cfg, str):
        cfg = os.path.expandvars(cfg)
    elif isinstance(cfg, list):
        cfg = [_expandvars(c) for c in cfg]
    return cfg


class SimConfig:

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        self._config: dict = _expandvars(raw_config)
        self._modules: Optional[dict] = None
        self._validate_config()

    def _validate_config(self):
        required = ['constants', 'universe', 'modules', 'portfolio']
        for section in required:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")

        constants = self._config['constants']
        for c in ['startdate', 'enddate', 'booksize']:
            if c not in constants:
                raise ValueError(f"Missing required constant: {c}")

    @property
    def constants(self) -> dict:
        return self._config.get('constants', {})

    @property
    def universe(self) -> dict:
        return self._config.get('universe', {})

    @property
    def modules(self) -> dict[str, tuple[ModuleType, dict]]:
        if self._modules is None:
            self._modules = {}
            for cfg in self._config['modules']:
                module_id = cfg['id']
                module_path = cfg['path']
                if not os.path.isabs(module_path):
                    module_path = os.path.abspath(module_path)
                spec = importlib.util.spec_from_file_location(module_id, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._modules[module_id] = (module, cfg)
        return self._modules

    @property
    def stats(self):
        cfg = self._config.get('stats', None)
        if cfg is None:
            return None
        return self.modules[cfg['moduleId']][0].create(cfg)

    @property
    def portfolio(self):
        cfg = self._config.get('portfolio', None)
        if cfg is None:
            return None
        from core.sim_node import create_node
        from core.tree_utils import preorder_iter
        portfolio = create_node(cfg)
        for node in preorder_iter(portfolio):
            node.module.children = node.children
        return portfolio


simcfg: Optional[SimConfig] = None


def init_simcfg(config_path: str) -> SimConfig:
    global simcfg
    if simcfg is None:
        simcfg = SimConfig(config_path)
    return simcfg
