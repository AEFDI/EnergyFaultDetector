import unittest
from typing import Dict

import numpy as np
import pandas as pd

from energy_fault_detector.autoencoders.bidirectional_lstm_seq2one_autoencoder import (
    BidirectionalLSTMSeq2OneAutoencoder,
)
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class TestBidirectionalLSTMSeq2OneAutoencoder(unittest.TestCase):
    """Unit tests for the BidirectionalLSTMSeq2OneAutoencoder."""

    def setUp(self) -> None:
        np.random.seed(42)

        self.sequence_length = 5
        self.ts_freq = np.timedelta64(10, "m")

        timestamps = pd.date_range("2025-01-01", periods=40, freq="10min")
        self.df = pd.DataFrame(
            np.random.random(size=(40, 3)),
            index=timestamps,
            columns=["f1", "f2", "f3"],
        )

        self.sequence_builder = SequenceDatasetBuilder(
            sequence_length=self.sequence_length,
            ts_freq=self.ts_freq,
            stride=1,
            pad_incomplete=False,
        )

        self.params: Dict = {
            "sequence_builder": self.sequence_builder,
            "layers": [8, 4],
            "dropout_rate": 0.0,
            "regularization": 0.01,
            "stateful": False,
            "merge_mode": "sum",
            "conditional_features": None,
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 2,
            "loss_name": "mean_squared_error",
            "metrics": ["mean_absolute_error"],
            "decay_rate": None,
            "decay_steps": None,
            "early_stopping": False,
            "patience": 3,
            "min_delta": 1e-4,
            "noise": 0.0,
            "verbose": 0,
        }

        self.autoencoder = BidirectionalLSTMSeq2OneAutoencoder(**self.params)

    def test_fit_builds_model_and_history(self) -> None:
        """Model and history should be created after fit."""
        self.autoencoder.fit(self.df)

        self.assertIsNotNone(self.autoencoder.model)
        self.assertIsNotNone(self.autoencoder.encoder)
        self.assertIn("loss", self.autoencoder.history)
        self.assertEqual(len(self.autoencoder.history["loss"]), self.autoencoder.epochs)

        input_shape = self.autoencoder.model.input_shape
        output_shape = self.autoencoder.model.output_shape

        self.assertEqual(len(input_shape), 3)
        self.assertEqual(input_shape[1], self.sequence_length)
        self.assertEqual(input_shape[2], 3)

        self.assertEqual(len(output_shape), 2)
        self.assertEqual(output_shape[1], 3)

    def test_predict_returns_last_timestep_per_window(self) -> None:
        """Predict should return one row per window, aligned to last timestamp."""
        self.autoencoder.fit(self.df)
        reconstruction = self.autoencoder.predict(self.df)

        self.assertIsInstance(reconstruction, pd.DataFrame)

        n_windows = len(self.df) - self.sequence_length + 1
        self.assertEqual(reconstruction.shape, (n_windows, self.df.shape[1]))
        self.assertListEqual(list(reconstruction.columns), list(self.df.columns))

        expected_index = self.df.index[self.sequence_length - 1 :]
        self.assertTrue(reconstruction.index.equals(expected_index))

    def test_fit_and_predict_with_conditional_features(self) -> None:
        """Conditional features should be passed as a second model input."""
        timestamps = pd.date_range("2025-02-01", periods=30, freq="10min")
        df_cond = pd.DataFrame(
            np.random.random(size=(30, 4)),
            index=timestamps,
            columns=["f1", "f2", "f3", "cond"],
        )

        builder = SequenceDatasetBuilder(
            sequence_length=self.sequence_length,
            ts_freq=self.ts_freq,
            stride=1,
            pad_incomplete=False,
        )

        ae_cond = BidirectionalLSTMSeq2OneAutoencoder(
            sequence_builder=builder,
            layers=[8, 4],
            dropout_rate=0.0,
            regularization=0.01,
            stateful=False,
            merge_mode="sum",
            conditional_features=["cond"],
            learning_rate=0.001,
            batch_size=8,
            epochs=2,
            loss_name="mean_squared_error",
            metrics=["mean_absolute_error"],
            decay_rate=None,
            decay_steps=None,
            early_stopping=False,
            patience=3,
            min_delta=1e-4,
            noise=0.0,
            verbose=0,
        )

        ae_cond.fit(df_cond)
        reconstruction = ae_cond.predict(df_cond)

        self.assertListEqual(list(reconstruction.columns), ["f1", "f2", "f3"])

        n_windows = len(df_cond) - self.sequence_length + 1
        self.assertEqual(reconstruction.shape[0], n_windows)

        expected_index = df_cond.index[self.sequence_length - 1 :]
        self.assertTrue(reconstruction.index.equals(expected_index))

        self.assertEqual(len(ae_cond.model.inputs), 2)
        self.assertEqual(len(ae_cond.encoder.inputs), 2)

    def test_timezone_aware_index_supported(self) -> None:
        """Predict and reconstruction error should preserve a timezone-aware index."""
        timestamps_tz = pd.date_range(
            "2025-01-01",
            periods=40,
            freq="10min",
            tz="Europe/Berlin",
        )
        df_tz = pd.DataFrame(
            np.random.random(size=(40, 3)),
            index=timestamps_tz,
            columns=["f1", "f2", "f3"],
        )

        ae_tz = BidirectionalLSTMSeq2OneAutoencoder(**self.params)
        ae_tz.fit(df_tz)

        reconstruction = ae_tz.predict(df_tz)
        n_windows = len(df_tz) - self.sequence_length + 1

        self.assertEqual(reconstruction.shape, (n_windows, df_tz.shape[1]))
        self.assertListEqual(list(reconstruction.columns), list(df_tz.columns))

        expected_index = df_tz.index[self.sequence_length - 1 :]
        self.assertEqual(reconstruction.index.tz, df_tz.index.tz)
        self.assertTrue(reconstruction.index.equals(expected_index))

        recon_error = ae_tz.get_reconstruction_error(df_tz)
        self.assertEqual(recon_error.index.tz, df_tz.index.tz)
        self.assertTrue(recon_error.index.equals(reconstruction.index))

    def test_invalid_merge_mode_raises(self) -> None:
        """Unsupported merge modes should fail during initialization."""
        with self.assertRaises(ValueError):
            BidirectionalLSTMSeq2OneAutoencoder(
                sequence_builder=self.sequence_builder,
                merge_mode="invalid",
            )

    def test_default_merge_mode_is_sum(self) -> None:
        """The default merge mode should match the selected project configuration."""
        self.assertEqual(self.autoencoder.merge_mode, "sum")


if __name__ == "__main__":
    unittest.main()
