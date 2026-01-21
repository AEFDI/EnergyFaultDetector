import os
import shutil
import unittest
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from energy_fault_detector.autoencoders.lstm_seq2seq_autoencoder import LSTMSeqAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class TestLSTMSeqAutoencoder(unittest.TestCase):
    """Unit tests for the LSTMSeqAutoencoder."""

    def setUp(self) -> None:
        np.random.seed(42)

        # Simple time-series data: 50 timesteps, 3 features
        self.sequence_length = 5
        self.ts_freq = np.timedelta64(10, "m")
        timestamps = pd.date_range("2025-01-01", periods=50, freq="10min")
        self.train_data = pd.DataFrame(
            np.random.random(size=(50, 3)),
            index=timestamps,
            columns=["f1", "f2", "f3"],
        )

        val_timestamps = pd.date_range("2025-01-02", periods=10, freq="10min")
        self.val_data = pd.DataFrame(
            np.random.random(size=(10, 3)),
            index=val_timestamps,
            columns=["f1", "f2", "f3"],
        )

        self.sequence_builder = SequenceDatasetBuilder(
            sequence_length=self.sequence_length,
            ts_freq=self.ts_freq,
            overlap=self.sequence_length - 1,  # stride 1
            pad_incomplete=False,
        )

        self.params: Dict = {
            "sequence_builder": self.sequence_builder,
            "layers": [8, 4],
            "dropout_rate": 0.0,
            "regularization": 0.01,
            "stateful": False,
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

        self.autoencoder = LSTMSeqAutoencoder(**self.params)
        self.saved_models_dir = "./test_models_lstm_seq_ae"

    def tearDown(self) -> None:
        """Remove any saved model directory after tests."""
        shutil.rmtree(self.saved_models_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Basic initialization
    # ------------------------------------------------------------------ #

    def test_init(self) -> None:
        self.assertEqual(self.autoencoder.layers, [8, 4])
        self.assertEqual(self.autoencoder.dropout_rate, 0.0)
        self.assertEqual(self.autoencoder.regularization, 0.01)
        self.assertEqual(self.autoencoder.batch_size, 8)
        self.assertEqual(self.autoencoder.loss_name, "mean_squared_error")
        self.assertEqual(self.autoencoder.metrics, ["mean_absolute_error"])

    # ------------------------------------------------------------------ #
    # Fitting and model structure
    # ------------------------------------------------------------------ #

    def test_fit_builds_model_and_history(self) -> None:
        self.autoencoder.fit(self.train_data, verbose=0)

        self.assertIsNotNone(self.autoencoder.model)
        self.assertIsNotNone(self.autoencoder.encoder)
        # History should contain 'loss' with length == epochs
        self.assertIn("loss", self.autoencoder.history)
        self.assertEqual(len(self.autoencoder.history["loss"]), self.autoencoder.epochs)

        # Check input / output shapes: (None, T, F)
        input_shape = self.autoencoder.model.input_shape
        output_shape = self.autoencoder.model.output_shape

        # Only one input when no conditional features
        self.assertEqual(len(input_shape), 3)
        self.assertEqual(input_shape[1], self.sequence_length)
        self.assertEqual(input_shape[2], 3)  # n_main_features

        self.assertEqual(output_shape[1], self.sequence_length)
        self.assertEqual(output_shape[2], 3)

    def test_fit_with_validation(self) -> None:
        self.autoencoder.fit(self.train_data, x_val=self.val_data, verbose=0)
        self.assertIsNotNone(self.autoencoder.model)
        self.assertIn("loss", self.autoencoder.history)
        self.assertEqual(len(self.autoencoder.history["loss"]), self.autoencoder.epochs)

    # ------------------------------------------------------------------ #
    # Predict, encode, reconstruction error
    # ------------------------------------------------------------------ #

    def test_predict_returns_dataframe_with_same_shape(self) -> None:
        self.autoencoder.fit(self.train_data, verbose=0)
        reconstruction = self.autoencoder.predict(self.train_data, verbose=0)

        self.assertIsInstance(reconstruction, pd.DataFrame)
        self.assertEqual(reconstruction.shape, self.train_data.shape)
        self.assertListEqual(list(reconstruction.columns), list(self.train_data.columns))
        self.assertTrue(reconstruction.index.equals(self.train_data.index))

    def test_predict_not_fitted_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = self.autoencoder.predict(self.train_data, verbose=0)

    def test_encode_returns_latent_vectors(self) -> None:
        self.autoencoder.fit(self.train_data, verbose=0)
        latent = self.autoencoder.encode(self.train_data)

        # Number of latent features should match last layer size
        self.assertEqual(latent.shape[1], self.autoencoder.layers[-1])
        # There should be at least one latent vector (number of windows)
        self.assertGreater(latent.shape[0], 0)

    def test_get_reconstruction_error_shape_and_index(self) -> None:
        self.autoencoder.fit(self.train_data, verbose=0)
        error = self.autoencoder.get_reconstruction_error(self.train_data, verbose=0)

        self.assertIsInstance(error, pd.DataFrame)
        self.assertEqual(error.shape, self.train_data.shape)
        self.assertListEqual(list(error.columns), list(self.train_data.columns))
        self.assertTrue(error.index.equals(self.train_data.index))

    # ------------------------------------------------------------------ #
    # Tuning
    # ------------------------------------------------------------------ #

    def test_tune_extends_history(self) -> None:
        self.autoencoder.fit(self.train_data, verbose=0)
        initial_len = len(self.autoencoder.history["loss"])

        self.autoencoder.tune(
            self.train_data,
            x_val=self.val_data,
            tune_epochs=1,
            learning_rate=0.0005,
            verbose=0,
        )

        self.assertEqual(
            len(self.autoencoder.history["loss"]),
            initial_len + 1,
        )

    # ------------------------------------------------------------------ #
    # Conditional features behaviour
    # ------------------------------------------------------------------ #

    def test_conditional_features_use_two_inputs_and_reconstruct_only_main(self) -> None:
        # Data with one conditional feature
        timestamps = pd.date_range("2025-01-03", periods=40, freq="10min")
        data_conditional = pd.DataFrame(
            np.random.random(size=(40, 3)),
            index=timestamps,
            columns=["f1", "f2", "cond1"],
        )

        builder = SequenceDatasetBuilder(
            sequence_length=self.sequence_length,
            ts_freq=self.ts_freq,
            overlap=self.sequence_length - 1,
            pad_incomplete=False,
        )

        ae_cond = LSTMSeqAutoencoder(
            sequence_builder=builder,
            layers=[8, 4],
            dropout_rate=0.0,
            regularization=0.01,
            stateful=False,
            conditional_features=["cond1"],
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

        ae_cond.fit(data_conditional, verbose=0)
        reconstruction = ae_cond.predict(data_conditional, verbose=0)

        # Only main features should be reconstructed
        self.assertListEqual(list(reconstruction.columns), ["f1", "f2"])

        # Model should have two inputs (main + conditional)
        self.assertEqual(len(ae_cond.model.inputs), 2)
        self.assertEqual(len(ae_cond.encoder.inputs), 2)

    # ------------------------------------------------------------------ #
    # Save and load
    # ------------------------------------------------------------------ #

    def test_save_and_load(self) -> None:
        # Fit and save
        self.autoencoder.fit(self.train_data, verbose=0)
        self.autoencoder.save(self.saved_models_dir, overwrite=True)

        # Create a new instance with the same builder (will be overwritten on load)
        new_ae = LSTMSeqAutoencoder()
        new_ae.load(self.saved_models_dir)

        # Basic attributes should match
        self.assertEqual(new_ae.layers, self.autoencoder.layers)
        self.assertEqual(new_ae.dropout_rate, self.autoencoder.dropout_rate)
        self.assertEqual(new_ae.regularization, self.autoencoder.regularization)
        self.assertEqual(new_ae.batch_size, self.autoencoder.batch_size)
        self.assertEqual(new_ae.loss_name, self.autoencoder.loss_name)
        self.assertDictEqual(new_ae.history, self.autoencoder.history)

        # Model weights should be identical
        for original_weights, loaded_weights in zip(
            self.autoencoder.model.weights,
            new_ae.model.weights,
        ):
            assert_array_almost_equal(original_weights, loaded_weights)
