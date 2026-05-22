"""Dump alpha values to CSV files for debugging/analysis."""
import numpy as np
import pandas as pd
import os

from modules.opbase import OpBase
from core.universe import univbase


class OpDump(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        base_dumpdir = cfg.get('dumpdir', None)
        if base_dumpdir is None:
            raise ValueError('dumpdir is mandatory for opdump')
        node_id = cfg.get('id', 'unknown')
        self.dumpdir = os.path.join(base_dumpdir, node_id)
        os.makedirs(self.dumpdir, exist_ok=True)

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        df = pd.DataFrame({'alpha': alpha}, index=univbase.symbols)
        df = df[df['alpha'].notna()]
        
        # Get timestamp string from interval index
        ts_str = univbase.itvl_to_timestamp(idx)
        
        df.to_csv(f'{self.dumpdir}/{ts_str}.csv')


def create(cfg: dict) -> OpDump:
    return OpDump(cfg)
