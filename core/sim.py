"""Main simulation loop: iterate over rebalance points, run portfolio tree."""
import pickle
import os
import numpy as np

from core.sim_config import simcfg
from core.sim_node import SimNode
from core.tree_utils import preorder_iter
from core.universe import univbase


def run(portfolio: SimNode) -> None:
    # Build rebalance schedule from config
    # rebalance_times: list of "HH:MM" (UTC) to rebalance each day
    # e.g., ["00:00"] for once daily, ["00:00", "12:00"] for twice daily
    rebalance_times = simcfg.constants.get('rebalance_times', ['00:00'])
    if isinstance(rebalance_times, str):
        rebalance_times = [rebalance_times]

    warmup_days = simcfg.constants.get('warmup', 1)
    warmup_itvls = warmup_days * univbase.bars_per_day

    # Convert HH:MM to bar offsets within a day
    offsets = []
    for t in rebalance_times:
        h, m = map(int, t.split(':'))
        offset = (h * 60 + m) // univbase.interval_minutes
        offsets.append(offset)

    # Generate schedule: for each day, add each offset
    schedule_list = []
    for di in range(univbase.n_dates):
        day_start = di * univbase.bars_per_day
        for offset in offsets:
            idx = day_start + offset
            if idx < univbase.n_intervals:
                schedule_list.append(idx)
    schedule = np.array(schedule_list, dtype=np.int64)

    # Filter by warmup
    schedule = schedule[schedule >= warmup_itvls]

    # Load checkpoint
    start_idx = 0
    checkpoint_path = simcfg.constants.get('checkpoint', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        with open(checkpoint_path, 'rb') as f:
            archive = pickle.load(f)
        start_idx = archive['start_idx']
        portfolio.set_archive(archive['portfolio'])

    # Main loop
    for i in range(start_idx, len(schedule)):
        idx = int(schedule[i])
        portfolio.run(idx)

    # Save checkpoint
    if checkpoint_path:
        archive = {'start_idx': len(schedule), 'portfolio': portfolio.get_archive()}
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(archive, f)

    # Close streams
    for node in preorder_iter(portfolio):
        node.close_pnl_stream()

    idx_start = int(schedule[0]) if len(schedule) > 0 else 0
    idx_end = int(schedule[-1]) if len(schedule) > 0 else 0
    print(f'Simulation complete: {len(schedule)} rebalance points '
          f'({univbase.itvl_to_date(idx_start)} -> {univbase.itvl_to_date(idx_end)})')
