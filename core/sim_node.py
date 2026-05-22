"""Tree-based simulation node: runs alpha generation + operations pipeline."""
import numpy as np
import os
from typing import Optional, Any

from core.sim_config import simcfg
from core.tree_utils import TreeNode
from core.universe import univbase
from lib import fast


class SimNode(TreeNode):

    def __init__(self, cfg: dict, parent: Optional['SimNode'] = None):
        super().__init__(parent)
        self.id = cfg['id']
        self.module = simcfg.modules[cfg['moduleId']][0].create(cfg)
        self.operations = []
        for op_cfg in cfg.get('operations', []):
            op = simcfg.modules[op_cfg['moduleId']][0].create({'id': self.id, **op_cfg})
            self.operations.append(op)
        self._booksize = simcfg.constants['booksize']
        self._verbose = simcfg.constants.get('verbose', False)
        self._prevalpha = np.zeros(1, dtype=np.float32)

    def run(self, idx: int) -> None:
        """Run alpha pipeline at interval index idx."""
        for child in self.children:
            child.run(idx)

        self.module.alpha[:] = np.nan
        self.module.generate(idx)

        for op in self.operations:
            op.apply(idx, self.module.alpha)

        if simcfg.stats:
            alpha = fast.nioscale(self.module.alpha, self._booksize) if simcfg.stats.scale else self.module.alpha
            self._output_stats(idx, alpha, self._prevalpha)
            self._prevalpha = alpha.copy()

    def _output_stats(self, idx: int, alpha: np.ndarray, prevalpha: np.ndarray) -> None:
        record = simcfg.stats.calculate(idx, alpha, prevalpha)
        if record is None:
            return
        ts_str, pnl, long_val, short_val, ret, hold_val, trade_val, lnum, snum, ic = record
        if not hasattr(self, 'pnlstream'):
            os.makedirs('pnl', exist_ok=True)
            self.pnlstream = open(f'pnl/{self.id}', 'w', buffering=8192)
        self.pnlstream.write(f'{ts_str} {pnl:.6f} {long_val:.6f} {short_val:.6f} {ret:.6f} {hold_val:.6f} {trade_val:.6f} {lnum} {snum} {ic:.6f}\n')
        if self._verbose:
            print(f'{ts_str}  {self.id:20} pnl={pnl:10.6f} ret={ret:8.5f} tv={trade_val:8.4f} L={lnum} S={snum} ic={ic:.4f}')

    def close_pnl_stream(self) -> None:
        if hasattr(self, 'pnlstream'):
            self.pnlstream.close()
            delattr(self, 'pnlstream')

    def get_archive(self) -> dict:
        return {
            'module': {var: getattr(self.module, var) for var in self.module.archive()},
            'operations': [{var: getattr(op, var) for var in op.archive()} for op in self.operations],
            'children': [child.get_archive() for child in self.children],
            '_prevalpha': self._prevalpha,
        }

    def set_archive(self, archive: dict) -> None:
        for key, val in archive.items():
            if key == 'module':
                for k, v in val.items():
                    setattr(self.module, k, v)
            elif key == 'operations':
                for op, op_vars in zip(self.operations, val):
                    for k, v in op_vars.items():
                        setattr(op, k, v)
            elif key == 'children':
                for child, child_vars in zip(self.children, val):
                    child.set_archive(child_vars)
            elif key == '_prevalpha':
                self._prevalpha = val


def create_node(cfg: dict, parent: Optional[SimNode] = None) -> SimNode:
    node = SimNode(cfg, parent)
    for child_cfg in cfg.get('children', []):
        create_node(child_cfg, node)
    return node
