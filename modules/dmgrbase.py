"""Base class for data managers with memmap storage.

Pre-allocates extra space along the time axis (default: 365 days) to avoid
daily resize. Only resizes when the pre-allocated space runs out.
"""
import numpy as np
import os
import yaml

from core.data_registry import get_data_registry
from core.sim_config import simcfg
from core.universe import univbase

# Pre-allocate this many extra days to avoid frequent resizes
PREALLOC_DAYS = 365


class DmgrBase:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.id = cfg['id']
        self.dr = get_data_registry()
        self.data_dir = cfg.get('cachepath', os.path.join(simcfg.constants['datacache'], self.id))
        self.mode = cfg.get('mode', simcfg.constants.get('mode', 'r'))
        self._rebuilt = False

    def initialize(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for field in self.dependencies():
            self.dr.register_dependency(self.id, field)

    def dependencies(self) -> list[str]:
        """Override to declare fields this dmgr depends on."""
        return []

    def load_data(self):
        pass

    def get_update_idx(self) -> int:
        """Get the last fully-loaded interval index."""
        meta_file = os.path.join(self.data_dir, '.meta')
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        return meta.get('update_idx', 0)

    def set_update_idx(self, idx: int):
        """Save progress: all data up to this interval index is loaded."""
        meta_file = os.path.join(self.data_dir, '.meta')
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        meta['update_idx'] = idx
        with open(meta_file, 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)

    def add_itvl_data(self, fields: list[str], dtype=np.float32, init_val=np.nan):
        """Allocate (n_intervals, n_symbols) memmap arrays with pre-allocation."""
        shape = (univbase.n_intervals, univbase.n_symbols)
        alloc_shape = self._alloc_shape(shape, time_axis=0)
        self._add_data(fields, dtype, init_val, shape, alloc_shape)


    def _alloc_shape(self, shape: tuple, time_axis: int = 0) -> tuple:
        """Add pre-allocation padding along the time axis."""
        if self.mode == 'r':
            return shape
        prealloc_days = simcfg.constants.get('prealloc_days', PREALLOC_DAYS)
        # For itvl data: pad by prealloc_days * bars_per_day
        # For daily data: pad by prealloc_days
        if shape[time_axis] == univbase.n_intervals:
            pad = prealloc_days * univbase.bars_per_day
        else:
            pad = prealloc_days
        alloc = list(shape)
        alloc[time_axis] += pad
        return tuple(alloc)

    def _add_data(self, fields: list[str], dtype, init_val, shape: tuple, alloc_shape: tuple):
        """Create or open memmap files.

        shape: the logical shape (what we need now)
        alloc_shape: the physical shape on disk (with pre-allocation padding)
        """
        meta_file = os.path.join(self.data_dir, '.meta')

        if not os.path.exists(meta_file):
            if self.mode == 'r':
                raise ValueError(f'{self.id}: no cache found but mode is read-only')
            self._rebuilt = True
            meta = {
                'fields': {},
                'datapath': self.cfg.get('datapath', ''),
                'update_idx': 0,
            }
            for field in fields:
                print(f'  Creating memmap {self.id}/{field}: {alloc_shape}')
                data_path = os.path.join(self.data_dir, field)
                data = np.memmap(data_path, dtype=dtype, mode='w+', shape=alloc_shape)
                data[:] = init_val
                data.flush()
                self.dr.setdata(field, data)
                meta['fields'][field] = {'shape': list(alloc_shape)}
            with open(meta_file, 'w') as f:
                yaml.dump(meta, f, default_flow_style=False)
        else:
            with open(meta_file, 'r') as f:
                meta = yaml.safe_load(f)

            # Datapath changed -> full rebuild
            if meta.get('datapath', '') != self.cfg.get('datapath', ''):
                if self.mode == 'r':
                    raise ValueError(f'{self.id}: datapath changed but mode is read-only')
                self._rebuilt = True
                meta['datapath'] = self.cfg.get('datapath', '')
                meta['update_idx'] = 0
                for field in fields:
                    print(f'  Rebuilding memmap {self.id}/{field}: {alloc_shape}')
                    data_path = os.path.join(self.data_dir, field)
                    data = np.memmap(data_path, dtype=dtype, mode='w+', shape=alloc_shape)
                    data[:] = init_val
                    data.flush()
                    self.dr.setdata(field, data)
                    meta['fields'][field] = {'shape': list(alloc_shape)}
                with open(meta_file, 'w') as f:
                    yaml.dump(meta, f, default_flow_style=False)
            else:
                for field in fields:
                    data_path = os.path.join(self.data_dir, field)

                    # New field
                    if field not in meta['fields']:
                        if self.mode == 'r':
                            raise ValueError(f'{self.id}/{field}: not found but mode is r')
                        print(f'  Creating memmap {self.id}/{field}: {alloc_shape}')
                        data = np.memmap(data_path, dtype=dtype, mode='w+', shape=alloc_shape)
                        data[:] = init_val
                        data.flush()
                        self.dr.setdata(field, data)
                        meta['fields'][field] = {'shape': list(alloc_shape)}
                        continue

                    disk_shape = tuple(meta['fields'][field]['shape'])

                    if shape[0] > disk_shape[0] or shape[1] > disk_shape[1]:
                        # Need more space than currently allocated -> resize with new padding
                        if self.mode == 'r':
                            raise ValueError(f'{self.id}/{field}: needs {shape} but disk has {disk_shape}, mode is r')
                        print(f'  Resizing {self.id}/{field}: {disk_shape} -> {alloc_shape}')
                        old_data = np.memmap(data_path, dtype=dtype, mode='r', shape=disk_shape).copy()
                        data = np.memmap(data_path, dtype=dtype, mode='w+', shape=alloc_shape)
                        data[:] = init_val
                        slices = tuple(slice(0, min(o, n)) for o, n in zip(disk_shape, alloc_shape))
                        data[slices] = old_data[slices]
                        data.flush()
                        self.dr.setdata(field, data)
                        meta['fields'][field] = {'shape': list(alloc_shape)}
                    else:
                        # Disk has enough space, just open it
                        mm_mode = 'r' if self.mode == 'r' else 'r+'
                        data = np.memmap(data_path, dtype=dtype, mode=mm_mode, shape=disk_shape)
                        self.dr.setdata(field, data)

                with open(meta_file, 'w') as f:
                    yaml.dump(meta, f, default_flow_style=False)

        # Register field -> dmgr mapping for DAG resolution
        for field in fields:
            self.dr.set_dmgr_from_data(field, self.id)
