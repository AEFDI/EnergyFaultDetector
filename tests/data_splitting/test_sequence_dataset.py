import unittest
from typing import List

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class TestSequenceDatasetBuilder(unittest.TestCase):
    """Unit tests for SequenceDatasetBuilder."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.sequence_length = 4
        self.ts_freq = np.timedelta64(10, "m")
        self.overlap = self.sequence_length - 1  # stride = 1

        # Simple regularly sampled time series: 20 timesteps, 3 features
        timestamps = pd.date_range("2025-01-01", periods=20, freq="10min")
        self.df = pd.DataFrame(
            np.random.random(size=(20, 3)),
            index=timestamps,
            columns=["f1", "f2", "f3"],
        )

        self.builder = SequenceDatasetBuilder(
            sequence_length=self.sequence_length,
            ts_freq=self.ts_freq,
            overlap=self.overlap,
            pad_incomplete=False,
        )

    def test_init_valid_params(self) -> None:
        builder = SequenceDatasetBuilder(
            sequence_length=5,
            ts_freq=np.timedelta64(10, "m"),
            overlap=4,
            pad_incomplete=False,
        )
        self.assertEqual(builder.sequence_length, 5)
        self.assertEqual(builder.overlap, 4)

    def test_init_invalid_sequence_length(self) -> None:
        with self.assertRaises(ValueError):
            _ = SequenceDatasetBuilder(
                sequence_length=0,
                ts_freq=self.ts_freq,
                overlap=0,
            )

    def test_init_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            _ = SequenceDatasetBuilder(
                sequence_length=4,
                ts_freq=self.ts_freq,
                overlap=4,  # must be < sequence_length
            )

    def test_build_sliding_dataset_no_conditional_train(self) -> None:
        batch_size = 4
        dataset, window_timestamps = self.builder.build_sliding_dataset(
            df=self.df,
            batch_size=batch_size,
            conditional_features=None,
            shuffle=False,
        )

        # For stride = 1: n_windows = n_samples - sequence_length + 1
        expected_windows = len(self.df) - self.sequence_length + 1
        self.assertEqual(window_timestamps.shape, (expected_windows, self.sequence_length))

        # Check shapes and that inputs == targets for first batch
        first_batch_inputs, first_batch_targets = next(dataset.as_numpy_iterator())
        # Shape: (batch_size, sequence_length, n_features)
        self.assertEqual(first_batch_inputs.shape[1], self.sequence_length)
        self.assertEqual(first_batch_inputs.shape[2], self.df.shape[1])
        assert_array_almost_equal(first_batch_inputs, first_batch_targets)

        # Check that the first window corresponds to df.iloc[0:sequence_length]
        expected_first = self.df.values[0 : self.sequence_length]
        assert_array_almost_equal(first_batch_inputs[0], expected_first)

    def test_build_sliding_dataset_conditional_train(self) -> None:
        batch_size = 4
        conditional_features = ["f3"]
        dataset, window_timestamps = self.builder.build_sliding_dataset(
            df=self.df,
            batch_size=batch_size,
            conditional_features=conditional_features,
            shuffle=False,
        )

        expected_windows = len(self.df) - self.sequence_length + 1
        self.assertEqual(window_timestamps.shape, (expected_windows, self.sequence_length))

        (inputs_main, inputs_cond), targets = next(dataset.as_numpy_iterator())

        # Main features: all except "f3" -> 2 features
        self.assertEqual(inputs_main.shape[2], 2)
        # Conditional features: "f3" -> 1 feature
        self.assertEqual(inputs_cond.shape[2], 1)
        # Targets: same as main
        self.assertEqual(targets.shape[2], 2)

        # Check that inputs_main equals targets for first batch
        assert_array_almost_equal(inputs_main, targets)

        # Check that first window's main features match df[["f1","f2"]].iloc[0:sequence_length]
        expected_main = self.df[["f1", "f2"]].values[0 : self.sequence_length]
        assert_array_almost_equal(inputs_main[0], expected_main)

        # Check that first window's conditional matches df[["f3"]]
        expected_cond = self.df[["f3"]].values[0 : self.sequence_length]
        assert_array_almost_equal(inputs_cond[0], expected_cond)

    def test_compute_valid_windows_respects_gaps(self) -> None:
        # Create a gap between index 9 and 10 (bigger than ts_freq)
        timestamps_with_gap: List[pd.Timestamp] = []
        for i in range(20):
            if i == 10:
                # Skip ahead by 1 hour to create a gap > ts_freq
                timestamps_with_gap.append(
                    timestamps_with_gap[-1] + pd.Timedelta(hours=1)
                )
            else:
                if not timestamps_with_gap:
                    timestamps_with_gap.append(pd.Timestamp("2025-01-01 00:00:00"))
                else:
                    timestamps_with_gap.append(
                        timestamps_with_gap[-1] + pd.Timedelta(minutes=10)
                    )

        df_gap = pd.DataFrame(
            np.random.random(size=(20, 3)),
            index=pd.DatetimeIndex(timestamps_with_gap),
            columns=["f1", "f2", "f3"],
        )

        builder = SequenceDatasetBuilder(
            sequence_length=self.sequence_length,
            ts_freq=self.ts_freq,
            overlap=self.overlap,
            pad_incomplete=False,
        )

        dataset, window_timestamps = builder.build_sliding_dataset(
            df=df_gap,
            batch_size=4,
            conditional_features=None,
            shuffle=False,
        )

        # Check that no window contains timestamps where diff > ts_freq
        window_timestamps_np = window_timestamps
        diffs = np.diff(window_timestamps_np, axis=1)
        self.assertTrue(np.all(diffs <= self.ts_freq))

        # At least one window should have been dropped compared to gap-free case
        expected_windows_no_gap = len(self.df) - self.sequence_length + 1
        self.assertLess(window_timestamps.shape[0], expected_windows_no_gap)

    def test_build_seq2one_dataset_no_conditional(self) -> None:
        batch_size = 4
        dataset, window_timestamps = self.builder.build_seq2one_dataset(
            df=self.df,
            batch_size=batch_size,
            conditional_features=None,
            shuffle=False,
        )

        expected_windows = len(self.df) - self.sequence_length + 1
        self.assertEqual(window_timestamps.shape, (expected_windows, self.sequence_length))

        inputs_batch, targets_batch = next(dataset.as_numpy_iterator())

        # Inputs: (batch, T, F)
        self.assertEqual(inputs_batch.shape[1], self.sequence_length)
        self.assertEqual(inputs_batch.shape[2], self.df.shape[1])

        # Targets: (batch, F)
        self.assertEqual(targets_batch.shape[1], self.df.shape[1])

        # Check that targets correspond to the last timestep of each input window
        assert_array_almost_equal(inputs_batch[:, -1, :], targets_batch)

    def test_build_seq2one_dataset_conditional(self) -> None:
        batch_size = 4
        conditional_features = ["f3"]
        dataset, window_timestamps = self.builder.build_seq2one_dataset(
            df=self.df,
            batch_size=batch_size,
            conditional_features=conditional_features,
            shuffle=False,
        )

        expected_windows = len(self.df) - self.sequence_length + 1
        self.assertEqual(window_timestamps.shape, (expected_windows, self.sequence_length))

        (inputs_main, inputs_cond), targets_batch = next(dataset.as_numpy_iterator())

        # Main features: f1, f2
        self.assertEqual(inputs_main.shape[2], 2)
        # Conditional: f3
        self.assertEqual(inputs_cond.shape[2], 1)
        # Targets: last timestep of main features -> shape (batch, 2)
        self.assertEqual(targets_batch.shape[1], 2)

        assert_array_almost_equal(inputs_main[:, -1, :], targets_batch)

    def test_non_datetime_index_raises(self) -> None:
        df_no_time_index = self.df.reset_index(drop=True)
        with self.assertRaises(ValueError):
            _ = self.builder.build_sliding_dataset(
                df=df_no_time_index,
                batch_size=4,
                conditional_features=None,
                shuffle=False,
            )

    def test_too_short_series_raises(self) -> None:
        # Fewer rows than sequence_length => _compute_valid_windows yields no starts
        timestamps_short = pd.date_range("2025-01-01", periods=2, freq="10min")
        df_short = pd.DataFrame(
            np.random.random(size=(2, 3)),
            index=timestamps_short,
            columns=["f1", "f2", "f3"],
        )

        builder_short = SequenceDatasetBuilder(
            sequence_length=4,
            ts_freq=self.ts_freq,
            overlap=3,
            pad_incomplete=False,
        )

        with self.assertRaises(ValueError):
            _ = builder_short.build_sliding_dataset(
                df=df_short,
                batch_size=2,
                conditional_features=None,
                shuffle=False,
            )
