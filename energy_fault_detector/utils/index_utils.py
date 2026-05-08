"""Utilities for handling DatetimeIndex and MultiIndex consistently."""

from typing import Optional, Tuple
import pandas as pd


def resolve_groupby_level(index: pd.Index, groupby_level: Optional[str] = "auto") -> Optional[int]:
    """Resolve a groupby level specification for a given index.

    Args:
        index: The DataFrame/Series index to inspect.
        groupby_level: One of:
            - None: no grouping.
            - "auto": auto-detect the first non-datetime level in a MultiIndex.
            - str/int: explicit level name or position.

    Returns:
        The resolved level (int position or str name), or None if no grouping applies.
    """
    if groupby_level is None:
        return None

    if groupby_level != "auto":
        # Explicit level — validate it exists
        if isinstance(index, pd.MultiIndex):
            if isinstance(groupby_level, int):
                if groupby_level < index.nlevels:
                    return groupby_level
            elif groupby_level in index.names:
                return groupby_level
        return None

    # Auto-detect: find the first non-datetime level in a MultiIndex
    if not isinstance(index, pd.MultiIndex):
        return None

    for i, level in enumerate(index.levels):
        if not isinstance(level, pd.DatetimeIndex):
            return i

    return None


def get_datetime_level(index: pd.Index) -> Optional[int]:
    """Return the position of the first DatetimeIndex level, or None."""
    if isinstance(index, pd.DatetimeIndex):
        return None  # The index itself is the datetime
    if isinstance(index, pd.MultiIndex):
        for i, level in enumerate(index.levels):
            if isinstance(level, pd.DatetimeIndex):
                return i
    return None
