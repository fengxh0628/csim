"""Group neutralization: subtract group mean within each cluster.

Groups are defined by cluster.group (produced by dmgrcluster).
For each interval, alpha values are demeaned within each cluster group,
removing common group-level exposure.

Usage in config:
   operations:
     - moduleId: opgroupneut
       field: group.k10        # default; also supports group.k5
"""
import numpy as np

from modules.opbase import OpBase
from core.data_registry import get_data_registry


class OpGroupNeut(OpBase):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dr = get_data_registry()
        self.field = cfg.get('field', 'group.k10')
        self.groups = self.dr.getdata(self.field)

    def apply(self, idx: int, alpha: np.ndarray) -> None:
        if idx >= self.groups.shape[0]:
            return

        glabels = self.groups[idx, :]
        v = np.isfinite(alpha) & (glabels >= 0)

        if v.sum() < 2:
            return

        labels_u = np.unique(glabels[v])
        for g in labels_u:
            gmask = (glabels == g) & v
            if gmask.sum() < 2:
                continue
            gmean = np.nanmean(alpha[gmask])
            alpha[gmask] -= gmean


def create(cfg: dict) -> OpGroupNeut:
    return OpGroupNeut(cfg)
