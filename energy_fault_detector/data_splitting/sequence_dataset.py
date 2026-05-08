from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from .data_gap_handler import DataGapHandler


# TODO: consider tf.keras.preprocessing.timeseries_dataset_from_array (needs masking)
# TODO: support tz-aware timestamps
class SequenceDatasetBuilder:
    """Build sequence datasets from a time-series DataFrame with a DatetimeIndex.

    This class provides two main dataset builders:

      * ``build_sliding_dataset`` for seq2seq models (sequence → sequence),
      * ``build_seq2one_dataset`` for seq2one models (sequence → single timestep).

    Timestamps are handled as datetime64[ns] (tz-naive, effectively UTC), and when mapping back we localize to the
    original tz.

    Features:

      * Sliding windows with configurable overlap.
      * Data gaps handled via :class:`DataGapHandler` (windows crossing gaps are dropped, so windows represent
        contiguous data).
      * Optional ``conditional_features`` split out as separate input sequences.

    Args:
        sequence_length: Number of time steps in each sequence.
        ts_freq: Expected sampling frequency as ``np.timedelta64``.
        stride: Stride between consecutive windows (sequence_length = disjoint).
        pad_incomplete: If True, resample to a regular grid and pad missing values with ``pad_value`` before windowing.
        pad_value: Value to use when padding during resampling if ``pad_incomplete`` is True.
    """

    def __init__(
        self,
        sequence_length: int,
        ts_freq: np.timedelta64,
        stride: int = 1,
        pad_incomplete: bool = False,
        pad_value: float = 0.0,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0.")
        if stride <= 0 or stride > sequence_length:
            raise ValueError("stride must be in [1, sequence_length].")

        self.sequence_length = sequence_length
        self.ts_freq = ts_freq
        self.stride = stride
        self.pad_incomplete = pad_incomplete
        self.pad_value = pad_value

    def _resample_if_needed(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Resample to a regular grid and pad, if ``pad_incomplete`` is True."""
        if not self.pad_incomplete:
            return data_frame

        start, end = data_frame.index[0], data_frame.index[-1]
        new_index = pd.date_range(start, end, freq=pd.Timedelta(self.ts_freq))
        return data_frame.reindex(new_index, fill_value=self.pad_value)

    def _compute_valid_windows(
        self,
        timestamps: np.ndarray,
        gap_handler: DataGapHandler,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return start-indices and timestamps for windows that do not cross gaps.

        Args:
            timestamps: 1D array of timestamps (e.g. datetime64) sorted in ascending order.
            gap_handler: DataGapHandler instance for these timestamps.

        Returns:
            A tuple ``(starts, window_timestamps)`` where:

              * ``starts`` is a 1D int array of valid window start indices,
              * ``window_timestamps`` is an array of shape (n_windows, sequence_length) with timestamps.

        Raises:
            ValueError: If no valid windows can be created (all cross data gaps).
        """

        n_samples = len(timestamps)
        if n_samples < self.sequence_length:
            raise ValueError("No valid windows found (series shorter than sequence_length).")

        # Fast path: no gaps at all → all windows are valid
        if gap_handler.data_gaps is None:
            # starts: 0, step, 2*step, ...
            starts = np.arange(0, n_samples - self.sequence_length + 1, self.stride, dtype=int)
            # Build an index matrix of shape (n_windows, sequence_length)
            window_idx = starts[:, None] + np.arange(self.sequence_length)[None, :]
            window_timestamps = timestamps[window_idx]
            return starts, window_timestamps

        starts: List[int] = []
        window_timestamps: List[np.ndarray] = []

        for start_idx in range(0, n_samples - self.sequence_length + 1, self.stride):
            start_ts = timestamps[start_idx]
            end_ts = timestamps[start_idx + self.sequence_length - 1]
            if gap_handler.has_data_gaps(start_ts, end_ts):
                continue

            starts.append(start_idx)
            window_timestamps.append(timestamps[start_idx: start_idx + self.sequence_length])

        if not starts:
            raise ValueError("No valid windows found (all windows cross data gaps).")

        return np.array(starts, dtype=int), np.array(window_timestamps)

    def build_sliding_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int,
        conditional_features: Optional[List[str]] = None,
        shuffle: bool = True,
        predict_mode: bool = False,
    ) -> Tuple[tf.data.Dataset, np.ndarray]:
        """Create a seq2seq tf.data.Dataset from a time-series DataFrame.

        Supports a simple DatetimeIndex or a MultiIndex with exactly one
        datetime-like level. In the MultiIndex case, windows are built per group
        (first non-datetime level) and concatenated.

        Args:
            df: Time-series data with DatetimeIndex, already preprocessed.
            batch_size: Batch size for the dataset.
            conditional_features: Optional list of column names to treat as conditional features.
            shuffle: Whether to shuffle sequences (only relevant when ``training`` is True).
            predict_mode: If True, the dataset is used for inference. To ensure we can predict all timestamps,
                we set stride to 1. TODO: consider stride = sequence_length.

        Returns:
            A tuple ``(dataset, window_timestamps)`` where:

              * ``dataset`` is a tf.data.Dataset for training or inference,
              * ``window_timestamps`` is an array of shape (n_windows, sequence_length)

                with timestamps for each window.
        """
        if isinstance(df.index, pd.DatetimeIndex):
            df, original_tz = self._strip_tz(df)
        else:
            original_tz = None  # MultiIndex or other, we don't handle tz here

        original_stride = self.stride
        if predict_mode:
            self.stride = 1

        (main_arr, cond_arr), window_timestamps = self._build_arrays_sliding(
            df, conditional_features=conditional_features
        )

        if cond_arr is not None:
            inputs = (main_arr, cond_arr)
            targets = main_arr
        else:
            inputs = main_arr
            targets = main_arr

        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(window_timestamps))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        window_timestamps = self._restore_tz(window_timestamps, original_tz)
        self.stride = original_stride
        return dataset, window_timestamps

    def build_seq2one_dataset(
            self,
            df: pd.DataFrame,
            batch_size: int,
            conditional_features: Optional[List[str]] = None,
            shuffle: bool = True,
            predict_mode: bool = False,
    ) -> Tuple[tf.data.Dataset, np.ndarray]:
        """Create a seq2one tf.data.Dataset from a time-series DataFrame.

        Inputs are sequences of length ``sequence_length``, targets are the main features
        at the last timestep of each sequence.

        Supports a simple DatetimeIndex or a MultiIndex with exactly one
        datetime-like level. In the MultiIndex case, windows are built per group
        (first non-datetime level) and concatenated.

        Args:
            df: Time-series data with DatetimeIndex, already preprocessed.
            batch_size: Batch size for the dataset.
            conditional_features: Optional list of column names to treat as conditional features.
            shuffle: Whether to shuffle sequences (only relevant when ``training`` is True).
            predict_mode: If True, the dataset is used for inference. To ensure we can predict all timestamps,
                we set stride to 1. TODO: consider stride = sequence_length.

        Returns:
            A tuple ``(dataset, window_timestamps)`` where:

              * ``dataset`` is a tf.data.Dataset for training or inference,
              * ``window_timestamps`` is an array of shape (n_windows, sequence_length)
                with timestamps for each window.
        """
        if isinstance(df.index, pd.DatetimeIndex):
            df, original_tz = self._strip_tz(df)
        else:
            original_tz = None

        original_stride = self.stride
        if predict_mode:
            self.stride = 1

        (main_arr, cond_arr), target_arr, window_timestamps = self._build_arrays_seq2one(
            df, conditional_features=conditional_features
        )

        if cond_arr is not None:
            inputs = (main_arr, cond_arr)
        else:
            inputs = main_arr

        dataset = tf.data.Dataset.from_tensor_slices((inputs, target_arr))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(window_timestamps))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        window_timestamps = self._restore_tz(window_timestamps, original_tz)
        self.stride = original_stride
        return dataset, window_timestamps

    def _strip_tz(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
        """Strip timezone from the DataFrame index, preserving local time values.

        Returns:
            Tuple of (tz-naive DataFrame, original tz or None).
        """
        tz = df.index.tz
        if tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df, tz

    def _restore_tz(self, window_timestamps: np.ndarray, tz) -> np.ndarray:
        """Re-attach timezone to a window_timestamps array.

        Args:
            window_timestamps: Array of shape (n_windows, sequence_length).
            tz: Timezone to restore, or None (no-op).

        Returns:
            window_timestamps with timezone restored (dtype=object if tz is not None).
        """
        if tz is not None:
            flat = pd.DatetimeIndex(window_timestamps.ravel()).tz_localize(tz)
            window_timestamps = np.array(flat, dtype=object).reshape(window_timestamps.shape)
        return window_timestamps

    @staticmethod
    def _extract_datetime_and_group(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[int], Optional[int]]:
        """Normalize index handling for DatetimeIndex and MultiIndex.

        Returns:
            df_sorted: DataFrame sorted by index.
            datetime_level_idx: index of datetime level if MultiIndex, otherwise None.
            group_level_idx: index of non-datetime level used for grouping (if MultiIndex), otherwise None.
        """
        df = df.sort_index()

        if isinstance(df.index, pd.DatetimeIndex):
            return df, None, None

        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                "DataFrame index must be a DatetimeIndex or MultiIndex with a datetime level."
            )

        datetime_level_idx = None
        non_datetime_levels = []
        for i, lvl in enumerate(df.index.levels):
            if isinstance(lvl, pd.DatetimeIndex):
                datetime_level_idx = i
            else:
                non_datetime_levels.append(i)

        if datetime_level_idx is None:
            raise ValueError(
                "MultiIndex must contain at least one DatetimeIndex level for sequence models."
            )

        group_level_idx = non_datetime_levels[0] if non_datetime_levels else None
        return df, datetime_level_idx, group_level_idx

    def _build_arrays_sliding(
            self,
            df: pd.DataFrame,
            conditional_features: Optional[List[str]] = None,
    ) -> Tuple[Tuple[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        """Build arrays (inputs, targets, timestamps) for seq2seq from DataFrame.

        Handles both DatetimeIndex and MultiIndex (per-group).
        """
        df, dt_level_idx, group_level_idx = self._extract_datetime_and_group(df)

        all_main_seqs: List[np.ndarray] = []
        all_cond_seqs: List[np.ndarray] = []
        all_timestamps: List[np.ndarray] = []

        def _process_one(frame: pd.DataFrame) -> None:
            frame_resampled = self._resample_if_needed(frame)
            timestamps = frame_resampled.index.values
            gap_handler = DataGapHandler(timestamps, self.ts_freq)
            starts, window_ts = self._compute_valid_windows(timestamps, gap_handler)

            if conditional_features:
                cond_cols = list(conditional_features)
                input_cols = [c for c in frame_resampled.columns if c not in cond_cols]

                main_vals = frame_resampled[input_cols].values.astype("float32")
                cond_vals = frame_resampled[cond_cols].values.astype("float32")

                window_idx = starts[:, None] + np.arange(self.sequence_length)[None, :]
                all_main_seqs.append(main_vals[window_idx])
                all_cond_seqs.append(cond_vals[window_idx])
            else:
                vals = frame_resampled.values.astype("float32")
                window_idx = starts[:, None] + np.arange(self.sequence_length)[None, :]
                all_main_seqs.append(vals[window_idx])

            all_timestamps.append(window_ts)

        if dt_level_idx is None:
            # simple DatetimeIndex
            _process_one(df)
        else:
            # MultiIndex – process per group if group_level_idx exists
            if group_level_idx is None:
                # Only datetime levels – treat as one series
                frame = df.copy()
                frame.index = frame.index.get_level_values(dt_level_idx)
                _process_one(frame)
            else:
                for _, frame in df.groupby(level=group_level_idx):
                    frame_single = frame.copy()
                    frame_single.index = frame_single.index.get_level_values(dt_level_idx)
                    _process_one(frame_single)

        main_arr = np.vstack(all_main_seqs)
        cond_arr = np.vstack(all_cond_seqs) if all_cond_seqs else None
        ts_arr = np.vstack(all_timestamps)
        return (main_arr, cond_arr), ts_arr

    def _build_arrays_seq2one(
            self,
            df: pd.DataFrame,
            conditional_features: Optional[List[str]] = None,
    ) -> Tuple[Tuple[np.ndarray, Optional[np.ndarray]], np.ndarray, np.ndarray]:
        """Build arrays (inputs, targets, timestamps) for seq2one from DataFrame.

        Handles both DatetimeIndex and MultiIndex (per-group).
        """
        df, dt_level_idx, group_level_idx = self._extract_datetime_and_group(df)

        all_main_seqs: List[np.ndarray] = []
        all_cond_seqs: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_timestamps: List[np.ndarray] = []
        all_group_keys: List[np.ndarray] = []

        def _process_one(frame: pd.DataFrame, group_key=None) -> None:
            frame_resampled = self._resample_if_needed(frame)
            timestamps = frame_resampled.index.values
            gap_handler = DataGapHandler(timestamps, self.ts_freq)
            starts, window_ts = self._compute_valid_windows(timestamps, gap_handler)

            if conditional_features:
                cond_cols = list(conditional_features)
                input_cols = [c for c in frame_resampled.columns if c not in cond_cols]

                main_vals = frame_resampled[input_cols].values.astype("float32")
                cond_vals = frame_resampled[cond_cols].values.astype("float32")

                window_idx = starts[:, None] + np.arange(self.sequence_length)[None, :]
                main_seqs = main_vals[window_idx]  # (N, T, F_main)
                targets = main_seqs[:, -1, :]  # (N, F_main)

                all_main_seqs.append(main_seqs)
                all_cond_seqs.append(cond_vals[window_idx])
                all_targets.append(targets)
            else:
                vals = frame_resampled.values.astype("float32")
                window_idx = starts[:, None] + np.arange(self.sequence_length)[None, :]
                main_seqs = vals[window_idx]  # (N, T, F)
                targets = main_seqs[:, -1, :]  # (N, F)

                all_main_seqs.append(main_seqs)
                all_targets.append(targets)

            all_timestamps.append(window_ts)
            if group_key is not None:
                all_group_keys.append(np.full(len(starts), group_key))

        if dt_level_idx is None:
            _process_one(df)
        else:
            if group_level_idx is None:
                frame = df.copy()
                frame.index = frame.index.get_level_values(dt_level_idx)
                _process_one(frame)
            else:
                for _, frame in df.groupby(level=group_level_idx):
                    frame_single = frame.copy()
                    frame_single.index = frame_single.index.get_level_values(dt_level_idx)
                    _process_one(frame_single)

        main_arr = np.vstack(all_main_seqs)
        cond_arr = np.vstack(all_cond_seqs) if all_cond_seqs else None
        target_arr = np.vstack(all_targets)
        ts_arr = np.vstack(all_timestamps)
        return (main_arr, cond_arr), target_arr, ts_arr
