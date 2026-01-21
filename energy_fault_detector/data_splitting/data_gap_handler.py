from typing import Tuple, Optional

import numpy as np


def shift_array(arr: np.ndarray, num: int, fill_value=None) -> np.ndarray:
    """Shift a NumPy array by a given number of positions.
    Elements shifted out of the array are replaced with ``fill_value``.

    Args:
        arr: Input NumPy array to shift.
        num: Number of positions to shift:
            * ``num > 0`` shifts to the right,
            * ``num < 0`` shifts to the left,
            * ``num == 0`` returns the array unchanged.

        fill_value: Value used to fill the newly created positions. Defaults to None.

    Returns:
        A new NumPy array with the same shape as ``arr`` containing the shifted values.
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


class DataGapHandler:
    """Handle data gaps in a time series given a target sampling frequency.

    Data gaps are identified wherever the difference between consecutive timestamps exceeds the target frequency.
    The handler then exposes convenience methods to check whether a timestamp or interval lies within a gap and
    to retrieve the next gap after a given timestamp.

    Attributes:
        freq: Target frequency as a ``np.timedelta64``.
        data_gaps: Optional array of shape (n_gaps, 2) with (gap_start, gap_end) timestamps, or None if no gaps.
    """

    def __init__(self, timestamps: np.ndarray, freq: np.timedelta64) -> None:
        """Initialize the DataGapHandler.

        Args:
            timestamps: 1D NumPy array of time stamps (e.g. dtype ``datetime64[ns]``) sorted in ascending order.
            freq: Target sampling frequency as a ``np.timedelta64``.
        """
        self.freq = freq
        self.data_gaps = self._get_data_gaps(timestamps)

    def _get_data_gaps(self, timestamps: np.ndarray) -> Optional[np.ndarray]:
        """Find data gaps based on the target frequency.

        A data gap is detected when the difference between consecutive timestamps exceeds ``self.freq``. For each gap,
        the start is defined as the earlier timestamp, and the end as the later timestamp bordering the gap.

        Args:
            timestamps: 1D NumPy array of time stamps sorted in ascending order.

        Returns:
            An array of shape (n_gaps, 2) with gap (start, end) timestamps, or None if no gaps are found.
        """
        starts = timestamps[shift_array(timestamps, -1) - timestamps > self.freq]
        ends = timestamps[timestamps - shift_array(timestamps, 1) > self.freq]
        data_gaps = np.array(list(zip(starts, ends)))

        if len(data_gaps) == 0:
            return None

        data_gaps = data_gaps[np.argsort(data_gaps[:, 0])]
        return data_gaps

    def is_in_gap(self, timestamp: np.datetime64) -> bool:
        """Check whether a single timestamp lies within any detected data gap.

        Args:
            timestamp: Timestamp to check.

        Returns:
            True if the timestamp is inside at least one gap, False otherwise.
        """
        if self.data_gaps is None:
            return False
        return np.any((self.data_gaps[:, 0] <= timestamp) & (timestamp <= self.data_gaps[:, 1]))

    def has_data_gaps(self, start: np.datetime64, end: np.datetime64) -> bool:
        """Check whether any data gap overlaps with a given time interval.

        Args:
            start: Start of the interval (inclusive).
            end: End of the interval (inclusive).

        Returns:
            True if at least one data gap overlaps the interval, False otherwise.
        """
        if self.data_gaps is None:
            return False
        return np.any((self.data_gaps[:, 0] < end) & (self.data_gaps[:, 1] > start))

    def get_next_gap_after(
        self,
        start_timestamp: np.datetime64,
    ) -> Optional[Tuple[np.datetime64, np.datetime64]]:
        """Get the next data gap following a given timestamp.

        Args:
            start_timestamp: Reference timestamp. The first gap with ``gap_start > start_timestamp`` is returned.

        Returns:
            A tuple ``(gap_start, gap_end)`` if a next gap exists, otherwise None.
        """
        if self.data_gaps is None:
            return None
        for gap_start, gap_end in self.data_gaps:
            if gap_start > start_timestamp:
                return gap_start, gap_end
        return None
