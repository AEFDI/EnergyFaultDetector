from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from .data_gap_handler import DataGapHandler


# TODO: consider tf.keras.preprocessing.timeseries_dataset_from_array when no data gaps are present
#       include masking of resampled data (data gaps) and missing values (padding?)
class SequenceDatasetBuilder:
    """Build sequence datasets from a time-series DataFrame with a DatetimeIndex.

    This class provides two main dataset builders:

      * ``build_sliding_dataset`` for seq2seq models (sequence → sequence),
      * ``build_seq2one_dataset`` for seq2one models (sequence → single timestep).

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
        step = self.stride

        starts: List[int] = []
        window_timestamps: List[np.ndarray] = []

        for start_idx in range(0, n_samples - self.sequence_length + 1, step):
            start_ts = timestamps[start_idx]
            end_ts = timestamps[start_idx + self.sequence_length - 1]
            if gap_handler.has_data_gaps(start_ts, end_ts):
                continue

            starts.append(start_idx)
            window_timestamps.append(timestamps[start_idx : start_idx + self.sequence_length])

        if not starts:
            raise ValueError("No valid windows found (all windows cross data gaps).")

        return np.array(starts, dtype=int), np.array(window_timestamps)

    def build_sliding_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int,
        conditional_features: Optional[List[str]] = None,
        shuffle: bool = True,
    ) -> Tuple[tf.data.Dataset, np.ndarray]:
        """Create a seq2seq tf.data.Dataset from a time-series DataFrame.

        Args:
            df: Time-series data with DatetimeIndex, already preprocessed.
            batch_size: Batch size for the dataset.
            conditional_features: Optional list of column names to treat as conditional features.
            shuffle: Whether to shuffle sequences (only relevant when ``training`` is True).

        Returns:
            A tuple ``(dataset, window_timestamps)`` where:

              * ``dataset`` is a tf.data.Dataset for training or inference,
              * ``window_timestamps`` is an array of shape (n_windows, sequence_length)

                with timestamps for each window.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        df_resampled = self._resample_if_needed(df)
        timestamps = df_resampled.index.values

        gap_handler = DataGapHandler(timestamps, self.ts_freq)
        starts, window_timestamps = self._compute_valid_windows(timestamps, gap_handler)

        if conditional_features:
            cond_cols = list(conditional_features)
            input_cols = [c for c in df_resampled.columns if c not in cond_cols]

            main_values = df_resampled[input_cols].values.astype("float32")
            cond_values = df_resampled[cond_cols].values.astype("float32")

            main_values_tf = tf.convert_to_tensor(main_values, dtype=tf.float32)
            cond_values_tf = tf.convert_to_tensor(cond_values, dtype=tf.float32)

            def map_fn(start_idx: tf.Tensor):
                start_idx = tf.cast(start_idx, tf.int32)
                end_idx = start_idx + self.sequence_length

                main_seq = main_values_tf[start_idx:end_idx]  # (T, F_main)
                cond_seq = cond_values_tf[start_idx:end_idx]  # (T, F_cond)
                inputs = (main_seq, cond_seq)
                targets = main_seq
                return inputs, targets

        else:
            values = df_resampled.values.astype("float32")
            values_tf = tf.convert_to_tensor(values, dtype=tf.float32)

            def map_fn(start_idx: tf.Tensor):
                start_idx = tf.cast(start_idx, tf.int32)
                end_idx = start_idx + self.sequence_length

                main_seq = values_tf[start_idx:end_idx]  # (T, F)
                inputs = main_seq
                targets = main_seq
                return inputs, targets

        starts_ds = tf.data.Dataset.from_tensor_slices(starts)
        dataset = starts_ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(starts))

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset, window_timestamps

    def build_seq2one_dataset(
            self,
            df: pd.DataFrame,
            batch_size: int,
            conditional_features: Optional[List[str]] = None,
            shuffle: bool = True,
    ) -> Tuple[tf.data.Dataset, np.ndarray]:
        """Create a seq2one tf.data.Dataset from a time-series DataFrame.

        Inputs are sequences of length ``sequence_length``, target is the last timestep of each sequence.

        Args:
            df: Time-series data with DatetimeIndex, already preprocessed.
            batch_size: Batch size for the dataset.
            conditional_features: Optional list of column names to treat as conditional features.
            shuffle: Whether to shuffle sequences (only relevant when ``training`` is True).

        Returns:
            A tuple ``(dataset, window_timestamps)`` where:

              * ``dataset`` is a tf.data.Dataset for training or inference,
              * ``window_timestamps`` is an array of shape (n_windows, sequence_length)

                with timestamps for each window.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        data_frame_resampled = self._resample_if_needed(df)
        timestamps = data_frame_resampled.index.values

        gap_handler = DataGapHandler(timestamps, self.ts_freq)
        starts, window_timestamps = self._compute_valid_windows(timestamps, gap_handler)

        if conditional_features:
            conditional_columns = list(conditional_features)
            input_columns = [c for c in data_frame_resampled.columns if c not in conditional_columns]

            main_values = data_frame_resampled[input_columns].values.astype("float32")
            conditional_values = data_frame_resampled[conditional_columns].values.astype("float32")

            main_values_tf = tf.convert_to_tensor(main_values, dtype=tf.float32)
            conditional_values_tf = tf.convert_to_tensor(conditional_values, dtype=tf.float32)

            def map_fn(start_idx: tf.Tensor):
                start_idx = tf.cast(start_idx, tf.int32)
                end_idx = start_idx + self.sequence_length

                main_seq = main_values_tf[start_idx:end_idx]  # (T, F_main)
                cond_seq = conditional_values_tf[start_idx:end_idx]  # (T, F_cond)
                inputs = (main_seq, cond_seq)
                target = main_seq[-1, :]  # (F_main,)
                return inputs, target

        else:
            values = data_frame_resampled.values.astype("float32")
            values_tf = tf.convert_to_tensor(values, dtype=tf.float32)

            def map_fn(start_idx: tf.Tensor):
                start_idx = tf.cast(start_idx, tf.int32)
                end_idx = start_idx + self.sequence_length

                main_seq = values_tf[start_idx:end_idx]  # (T, F)
                inputs = main_seq
                target = main_seq[-1, :]  # (F,)
                return inputs, target

        starts_ds = tf.data.Dataset.from_tensor_slices(starts)
        dataset = starts_ds.map(
            map_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(starts))

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset, window_timestamps

