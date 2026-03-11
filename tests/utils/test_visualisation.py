import unittest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_fault_detector.utils import visualisation as viz
from energy_fault_detector.fault_detector import FaultDetector


class DummyAutoencoder:
    """Minimal autoencoder stub with training history."""
    def __init__(self):
        self.history = {
            "loss": [1.0, 0.5, 0.25],
            "val_loss": [1.1, 0.6, 0.3],
        }


class DummyPredictions:
    """Predictions stub compatible with FaultDetector.predict result."""
    def __init__(self, reconstruction: pd.DataFrame):
        self.reconstruction = reconstruction
        idx = reconstruction.index
        self.anomaly_score = pd.Series(
            np.linspace(0, 1, len(idx)), index=idx, name="score"
        )
        self.predicted_anomalies = pd.Series(False, index=idx)

    def criticality(self, normal_idx=None, max_criticality=1000):
        # trivial constant criticality, same index as anomaly_score
        return pd.Series(0, index=self.anomaly_score.index, name="criticality")


class DummyFDWithDP:
    """FaultDetector-like stub for plot_reconstruction_with_model."""
    def __init__(self, data: pd.DataFrame):
        self.autoencoder = DummyAutoencoder()
        # simple sklearn pipeline as data_preprocessor
        self.data_preprocessor = Pipeline([("scaler", StandardScaler())])

        # initial prediction: reconstruction = input data
        self._base_recon = data.copy()

    def predict(self, sensor_data: pd.DataFrame) -> DummyPredictions:
        # For testing, use the provided sensor_data itself as reconstruction
        return DummyPredictions(sensor_data.copy())


class TestVisualisation(unittest.TestCase):
    def test_plot_learning_curve_with_autoencoder(self):
        ae = DummyAutoencoder()
        fig, ax = viz.plot_learning_curve(ae)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_learning_curve_with_fault_detector(self):
        fd = FaultDetector()
        fd.autoencoder = DummyAutoencoder()

        fig, ax = viz.plot_learning_curve(fd)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_reconstruction_with_model(self):
        # Synthetic data
        idx = pd.date_range("2025-01-01", periods=10, freq="h")
        data = pd.DataFrame(
            {
                "a": np.arange(10, dtype=float),
                "b": np.linspace(0.0, 1.0, 10),
            },
            index=idx,
        )

        fd = DummyFDWithDP(data)

        fig, ax = viz.plot_reconstruction_with_model(fd, data)
        self.assertIsInstance(fig, plt.Figure)

        axes = [ax] if not isinstance(ax, np.ndarray) else ax.flatten()
        # by default, plot all reconstruction columns -> 2 subplots
        self.assertEqual(len(axes), 2)
        plt.close(fig)

    def test_plot_arcana_mean_importances(self):
        importances = pd.Series(
            {
                "f1": 0.1,
                "f2": 0.4,
                "f3": 0.2,
                "f4": 0.8,
            }
        )

        fig, ax = viz.plot_arcana_mean_importances(importances, top_n_features=2)
        self.assertIsInstance(fig, plt.Figure)

        # Expect 2 horizontal bars (top 2 features)
        bars = ax.patches
        self.assertEqual(len(bars), 2)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
