"""Compatibility tests for loading and applying pre-Pydantic saved models.

The fixtures under ``tests/test_data/old_models/`` were created with older
versions of ``energy-fault-detector`` (before the Pydantic config migration)
using the script ``tests/test_data/old_models/create_legacy_model_fixtures.py``.

For Keras 3 fixtures (``py312_efd071``) the full pipeline is exercised:
``FaultDetector.load`` + ``predict``.  For Keras 2 fixtures (``py311_efd060``)
the autoencoder cannot be loaded by Keras 3 (``keras.src.engine.functional``
was renamed), so only the non-Keras components (config, data preprocessor,
threshold selector, anomaly score) are loaded and applied individually.
"""

import os
import unittest

import numpy as np
import pandas as pd

from energy_fault_detector.anomaly_scores import RMSEScore
from energy_fault_detector.autoencoders import LSTMSeq2OneAutoencoder, MultilayerAutoencoder
from energy_fault_detector.config import Config
from energy_fault_detector.core import FaultDetectionResult
from energy_fault_detector.data_preprocessing import DataPreprocessor
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.threshold_selectors import QuantileThresholdSelector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_MODELS_DIR = os.path.join(PROJECT_ROOT, 'tests', 'test_data', 'old_models')


def make_dense_data(n_rows: int = 80) -> pd.DataFrame:
    """Create deterministic tabular data matching the dense fixture training data.

    Args:
        n_rows: Number of samples to generate.

    Returns:
        Numeric DataFrame with a 10-minute DatetimeIndex.
    """
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=n_rows, freq="10min")
    phase = np.linspace(0, 4 * np.pi, n_rows)
    return pd.DataFrame(
        {
            "temperature": 50 + 3 * np.sin(phase) + rng.normal(0, 0.15, n_rows),
            "pressure": 10 + 0.5 * np.cos(phase) + rng.normal(0, 0.05, n_rows),
            "power": 100 + 8 * np.sin(phase / 2) + rng.normal(0, 0.3, n_rows),
        },
        index=index,
    )


def make_sequence_data(n_rows: int = 120) -> pd.DataFrame:
    """Create deterministic time-series data matching the sequence fixture training data.

    Args:
        n_rows: Number of samples to generate.

    Returns:
        Numeric DataFrame with a regular 10-minute DatetimeIndex.
    """
    rng = np.random.default_rng(123)
    index = pd.date_range("2025-02-01", periods=n_rows, freq="10min")
    phase = np.linspace(0, 6 * np.pi, n_rows)
    return pd.DataFrame(
        {
            "sensor_a": np.sin(phase) + rng.normal(0, 0.03, n_rows),
            "sensor_b": np.cos(phase) + rng.normal(0, 0.03, n_rows),
            "sensor_c": 0.3 * np.sin(phase / 2) + rng.normal(0, 0.02, n_rows),
        },
        index=index,
    )


class TestLegacyModelCompatibility(unittest.TestCase):
    """Load and apply saved-model fixtures created before the Pydantic config migration."""

    def _assert_loaded_components(self, detector: FaultDetector) -> None:
        """Verify all model components and config were loaded."""
        self.assertIsNotNone(detector.autoencoder)
        self.assertIsNotNone(detector.data_preprocessor)
        self.assertIsNotNone(detector.threshold_selector)
        self.assertIsNotNone(detector.anomaly_score)
        self.assertIsNotNone(detector.config)

    def _assert_predict_results(self, result: FaultDetectionResult, expected_len: int) -> None:
        """Verify prediction results are sane and internally consistent.

        Args:
            result: Output of ``FaultDetector.predict``.
            expected_len: Expected number of rows in the anomaly score / predictions.
        """
        self.assertFalse(result.anomaly_score.empty)
        self.assertTrue(np.all(np.isfinite(result.anomaly_score.values)))
        self.assertEqual(len(result.anomaly_score), expected_len)
        self.assertIsInstance(result.predicted_anomalies, pd.Series)
        self.assertEqual(result.predicted_anomalies.dtype, bool)
        self.assertTrue(result.predicted_anomalies.index.equals(result.anomaly_score.index))

    @staticmethod
    def _load_non_keras_components(model_path: str) -> tuple:
        """Load config, data preprocessor, threshold selector, and anomaly score.

        Bypasses ``FaultDetector.load`` (which also loads the Keras autoencoder)
        so the non-Keras pickled components can be tested independently.

        Args:
            model_path: Directory containing the saved model (config.yaml + sub-dirs).

        Returns:
            Tuple of (Config, DataPreprocessor, QuantileThresholdSelector, RMSEScore).
        """
        config = Config(os.path.join(model_path, 'config.yaml'))

        data_preprocessor = DataPreprocessor()
        data_preprocessor.load(os.path.join(model_path, 'data_preprocessor'))

        threshold_selector = QuantileThresholdSelector()
        threshold_selector.load(os.path.join(model_path, 'threshold_selector'))

        anomaly_score = RMSEScore()
        anomaly_score.load(os.path.join(model_path, 'anomaly_score'))

        return config, data_preprocessor, threshold_selector, anomaly_score

    def _assert_non_keras_pipeline(self, model_path: str, data: pd.DataFrame,
                                   expected_ae_name: str) -> None:
        """Load non-Keras components, run data through the partial pipeline.

        Args:
            model_path: Directory containing the saved model.
            data: Synthetic data matching the fixture's training data.
            expected_ae_name: Expected ``config.train.autoencoder.name`` value.
        """
        config, data_preprocessor, threshold_selector, anomaly_score = \
            self._load_non_keras_components(model_path)

        self.assertEqual(config.train.autoencoder.name, expected_ae_name)
        self.assertIsInstance(data_preprocessor, DataPreprocessor)
        self.assertIsInstance(threshold_selector, QuantileThresholdSelector)
        self.assertIsInstance(anomaly_score, RMSEScore)

        prepped = data_preprocessor.transform(data)
        self.assertFalse(prepped.empty)

        rng = np.random.default_rng(0)
        recon_error = pd.DataFrame(
            rng.standard_normal(prepped.shape),
            index=prepped.index, columns=prepped.columns,
        )
        scores = anomaly_score.transform(recon_error)
        self.assertFalse(scores.empty)
        self.assertTrue(np.all(np.isfinite(scores.values)))

        predicted = threshold_selector.predict(scores)
        self.assertEqual(len(predicted), len(scores))

    def test_dense_py312_efd071(self) -> None:
        """Dense MultilayerAutoencoder saved with EFD 0.7.1 / Python 3.12 / Keras 3."""
        model_path = os.path.join(OLD_MODELS_DIR, 'py312_efd071', 'dense')
        detector = FaultDetector.load(model_path=model_path)
        self._assert_loaded_components(detector)
        self.assertIsInstance(detector.autoencoder, MultilayerAutoencoder)
        self.assertEqual(detector.config.train.autoencoder.name, 'default')

        result = detector.predict(make_dense_data())
        self._assert_predict_results(result, expected_len=80)

    def test_sequence_py312_efd071(self) -> None:
        """LSTM seq2one autoencoder saved with EFD 0.7.1 / Python 3.12 / Keras 3."""
        model_path = os.path.join(OLD_MODELS_DIR, 'py312_efd071', 'sequence')
        detector = FaultDetector.load(model_path=model_path)
        self._assert_loaded_components(detector)
        self.assertIsInstance(detector.autoencoder, LSTMSeq2OneAutoencoder)
        self.assertEqual(detector.config.train.autoencoder.name, 'lstm_seq2one')

        result = detector.predict(make_sequence_data())
        self._assert_predict_results(result, expected_len=113)

    def test_dense_py311_efd060(self) -> None:
        """Non-Keras components from EFD 0.6.0 / Python 3.11 / Keras 2 fixture (dense).

        The Keras 2 autoencoder cannot be loaded by Keras 3, but the config,
        data preprocessor, threshold selector, and anomaly score are plain
        pickle/YAML and should load and run.
        """
        model_path = os.path.join(OLD_MODELS_DIR, 'py311_efd060', 'dense')
        self._assert_non_keras_pipeline(model_path, make_dense_data(), 'default')

    def test_sequence_py311_efd060(self) -> None:
        """Non-Keras components from EFD 0.6.0 / Python 3.11 / Keras 2 fixture (sequence).

        The Keras 2 autoencoder cannot be loaded by Keras 3, but the config,
        data preprocessor, threshold selector, and anomaly score are plain
        pickle/YAML and should load and run.
        """
        model_path = os.path.join(OLD_MODELS_DIR, 'py311_efd060', 'sequence')
        self._assert_non_keras_pipeline(model_path, make_sequence_data(), 'lstm_seq2one')
