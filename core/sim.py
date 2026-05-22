"""Main simulation loop: iterate over rebalance points, run portfolio tree."""
import pickle
import os
import numpy as np
from datetime import datetime

from core.sim_config import simcfg
from core.sim_node import SimNode
from core.tree_utils import preorder_iter
from core.universe import univbase


def run(portfolio: SimNode) -> None:
    start_date = datetime.strptime(str(simcfg.constants['startdate']), '%Y%m%d').date()
    end_date = datetime.strptime(str(simcfg.constants['enddate']), '%Y%m%d').date()

    start_didx = univbase.date_to_idx(start_date)
    end_didx = univbase.date_to_idx(end_date)

    warmup_days = simcfg.constants.get('warmup', 1)
    warmup_didx = start_didx + warmup_days

    rebalance_times = simcfg.constants.get('rebalance_times', ['00:00'])
    if isinstance(rebalance_times, str):
        rebalance_times = [rebalance_times]

    offsets = []
    for t in rebalance_times:
        h, m = map(int, t.split(':'))
        offset = (h * 60 + m) // univbase.interval_minutes
        offsets.append(offset)

    # Load checkpoint
    start_i = 0
    checkpoint_path = simcfg.constants.get('checkpoint', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        with open(checkpoint_path, 'rb') as f:
            archive = pickle.load(f)
        start_i = archive['start_i']
        portfolio.set_archive(archive['portfolio'])

    # Main loop
    n_runs = 0
    for di in range(warmup_didx, end_didx + 1):
        if di < start_didx:
            continue
        day_start = di * univbase.bars_per_day
        for offset in offsets:
            idx = day_start + offset
            if idx >= univbase.n_intervals:
                continue
            if n_runs < start_i:
                n_runs += 1
                continue
            portfolio.run(idx)
            n_runs += 1

    # Save checkpoint
    if checkpoint_path:
        archive = {'start_i': n_runs, 'portfolio': portfolio.get_archive()}
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(archive, f)

    # Close streams
    for node in preorder_iter(portfolio):
        node.close_pnl_stream()

    print(f'Simulation complete: {n_runs} rebalance points '
          f'({univbase.itvl_to_date(start_didx * univbase.bars_per_day)} -> '
          f'{univbase.itvl_to_date(end_didx * univbase.bars_per_day)})')
