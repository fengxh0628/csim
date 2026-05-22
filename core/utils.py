"""Utility functions."""
from core.universe import univbase


def parse_duration(s) -> int:
    """Convert duration string to number of bars.

    Supports: '5m', '1h', '4h', '1d', '7d', '20m', etc.
    Also accepts int (treated as number of days for backward compatibility).
    """
    if isinstance(s, (int, float)):
        # Legacy: bare number = days
        return int(s) * univbase.bars_per_day

    s = str(s).strip().lower()
    if s.endswith('d'):
        return int(s[:-1]) * univbase.bars_per_day
    elif s.endswith('h'):
        return int(s[:-1]) * (60 // univbase.interval_minutes)
    elif s.endswith('m'):
        return int(s[:-1]) // univbase.interval_minutes
    else:
        # Bare number = days
        return int(s) * univbase.bars_per_day
