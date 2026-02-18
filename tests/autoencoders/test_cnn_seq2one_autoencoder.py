import unittest
from typing import Dict

import numpy as np
import pandas as pd

from energy_fault_detector.autoencoders.cnn_seq2one_autoencoder import (
    CNNSeq2OneAutoencoder,
)
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class TestCNNSeq2OneAutoencoder(unittest.TestCase):
    """Unit tests for the CNNSeq2OneAutoencoder."""

    def setUp(self) -> None:
        np.random.seed(42)

        self.sequence_length = 5
        self.ts_freq = np.timedelta64(10, "m")

        # Simple time-series data: 40 timesteps, 3 features
        timestamps = pd.date_range("2025-01-01", periods=40, freq="10min")
        self.df = pd.DataFrame(
            np.random.random(size=(40, 3)),
            index=timestamps,
            columns=["f1", "f2", "f3"],
        )

        self.sequence_builder = SequenceDatasetBuilder(
            sequence_length=self.sequence_length,
            ts_freq=self.ts_freq,
            overlap=self.sequence_length - 1,  # stride = 1
            pad_incomplete=False,
        )

        self.params: Dict = {
            "sequence_builder": self.sequence_builder,
            "filters": [16, 8],
            "kernel_size": 3,
            "strides": 1,
            "dropout_rate": 0.0,
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
        }

        self.autoencoder = CNNSeq2OneAutoencoder(**self.params)

    # ------------------------------------------------------------------ #
    # Basic fit / predict (no conditional features)
    # ------------------------------------------------------------------ #

    def test_fit_builds_model_and_history(self) -> None:
        """Model and history should be created after fit."""
        self.autoencoder.fit(self.df, verbose=0)

        self.assertIsNotNone(self.autoencoder.model)
        self.assertIsNotNone(self.autoencoder.encoder)
        self.assertIn("loss", self.autoencoder.history)
        self.assertEqual(len(self.autoencoder.history["loss"]), self.autoencoder.epochs)

        # Check model input / output shape
        input_shape = self.autoencoder.model.input_shape  # (None, T, F_main)
        output_shape = self.autoencoder.model.output_shape  # (None, F_main)

        self.assertEqual(len(input_shape), 3)
        self.assertEqual(input_shape[1], self.sequence_length)
        self.assertEqual(input_shape[2], 3)  # n_main_features

        self.assertEqual(len(output_shape), 2)
        self.assertEqual(output_shape[1], 3)  # seq2one: one vector of F_main

    def test_predict_returns_last_timestep_per_window(self) -> None:
        """Predict should return one row per window, aligned to last timestamp."""
        self.autoencoder.fit(self.df, verbose=0)
        reconstruction = self.autoencoder.predict(self.df, verbose=0)

        self.assertIsInstance(reconstruction, pd.DataFrame)

        n_windows = len(self.df) - self.sequence_length + 1
        self.assertEqual(reconstruction.shape, (n_windows, self.df.shape[1]))
        self.assertListEqual(list(reconstruction.columns), list(self.df.columns))

        # Index: last timestamp of each window
        expected_index = self.df.index[self.sequence_length - 1 :]
        self.assertTrue(reconstruction.index.equals(expected_index))

    # ------------------------------------------------------------------ #
    # Conditional features
    # ------------------------------------------------------------------ #

    def test_fit_and_predict_with_conditional_features(self) -> None:
        """When using conditional features, model should have two inputs and predict only main features."""
        # Add one conditional feature 'cond'
        timestamps = pd.date_range("2025-02-01", periods=30, freq="10min")
        df_cond = pd.DataFrame(
            np.random.random(size=(30, 4)),
            index=timestamps,
            columns=["f1", "f2", "f3", "cond"],
        )

        builder = SequenceDatasetBuilder(
            sequence_length=self.sequence_length,
            ts_freq=self.ts_freq,
            overlap=self.sequence_length - 1,
            pad_incomplete=False,
        )

        ae_cond = CNNSeq2OneAutoencoder(
            sequence_builder=builder,
            filters=[16, 8],
            kernel_size=3,
            strides=1,
            dropout_rate=0.0,
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
        )

        ae_cond.fit(df_cond, verbose=0)
        reconstruction = ae_cond.predict(df_cond, verbose=0)

        # Only main features (all except "cond") should be reconstructed
        self.assertListEqual(list(reconstruction.columns), ["f1", "f2", "f3"])

        # Check number of windows and index alignment (last timestamps)
        n_windows = len(df_cond) - self.sequence_length + 1
        self.assertEqual(reconstruction.shape[0], n_windows)

        expected_index = df_cond.index[self.sequence_length - 1 :]
        self.assertTrue(reconstruction.index.equals(expected_index))

        # Model and encoder should have two inputs
        # CNNSeq2OneAutoencoder.create_model uses a list for inputs if conditional
        self.assertEqual(len(ae_cond.model.inputs), 2)
        self.assertEqual(len(ae_cond.encoder.inputs), 2)


if __name__ == "__main__":
    unittest.main()
