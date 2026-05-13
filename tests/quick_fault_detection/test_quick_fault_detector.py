"""Integration tests for the quick_fault_detector pipeline."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
import pandas as pd

from energy_fault_detector.quick_fault_detection.quick_fault_detector import (
    quick_fault_detector,
    analyze_event,
)
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult


def _make_synthetic_data(n_rows=200, n_features=5, freq="10min"):
    """Create synthetic time-series data for testing."""
    index = pd.date_range("2024-01-01", periods=n_rows, freq=freq)
    data = pd.DataFrame(
        np.random.randn(n_rows, n_features),
        index=index,
        columns=[f"sensor_{i}" for i in range(n_features)],
    )
    return data


def _make_csv_files(tmp_dir, n_rows=200, n_features=5):
    """Write synthetic train/test CSVs and return paths."""
    train_data = _make_synthetic_data(n_rows, n_features)
    test_data = _make_synthetic_data(n_rows, n_features)

    train_path = os.path.join(tmp_dir, "train.csv")
    test_path = os.path.join(tmp_dir, "test.csv")

    train_data.to_csv(train_path, index_label="timestamp")
    test_data.to_csv(test_path, index_label="timestamp")

    return train_path, test_path


class TestQuickFaultDetectorIntegration(unittest.TestCase):
    """Integration tests for the quick_fault_detector pipeline."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.train_path, self.test_path = _make_csv_files(self.tmp_dir)

    @patch("energy_fault_detector.quick_fault_detection.quick_fault_detector.generate_output_plots")
    @patch("energy_fault_detector.quick_fault_detection.quick_fault_detector.FaultDetector")
    def test_full_pipeline_no_anomalies(self, MockFaultDetector, mock_plots):
        """Pipeline runs end-to-end when no anomalies are detected."""
        # Setup mock
        mock_model = MockFaultDetector.return_value
        mock_model.fit.return_value = None

        # predict returns no anomalies
        n_rows = 200
        mock_result = MagicMock(spec=FaultDetectionResult)
        mock_result.predicted_anomalies = pd.Series(
            [False] * n_rows,
            index=pd.date_range("2024-01-01", periods=n_rows, freq="10min"),
            name="anomaly",
        )
        mock_model.predict.return_value = mock_result

        result, events = quick_fault_detector(
            csv_data_path=self.train_path,
            csv_test_data_path=self.test_path,
            time_column_name="timestamp",
            automatic_optimization=False,
            min_anomaly_length=18,
            save_dir=self.tmp_dir,
        )

        mock_model.fit.assert_called_once()
        mock_model.predict.assert_called_once()
        self.assertEqual(len(events), 0)
        mock_plots.assert_called_once()

    @patch("energy_fault_detector.quick_fault_detection.quick_fault_detector.generate_output_plots")
    @patch("energy_fault_detector.quick_fault_detection.quick_fault_detector.FaultDetector")
    def test_full_pipeline_with_anomalies(self, MockFaultDetector, mock_plots):
        """Pipeline runs ARCANA when anomalies are detected."""
        n_rows = 200
        mock_model = MockFaultDetector.return_value
        mock_model.fit.return_value = None

        # Create anomaly block long enough to form an event
        anomalies = [False] * 50 + [True] * 30 + [False] * 120
        index = pd.date_range("2024-01-01", periods=n_rows, freq="10min")

        mock_result = MagicMock(spec=FaultDetectionResult)
        mock_result.predicted_anomalies = pd.Series(
            anomalies,
            index=index,
            name="anomaly",
        )
        mock_model.predict.return_value = mock_result

        # Mock ARCANA
        bias_df = pd.DataFrame(
            np.random.randn(30, 5),
            columns=[f"sensor_{i}" for i in range(5)],
            index=index[50:80],
        )
        mock_model.run_root_cause_analysis.return_value = (
            bias_df,
            pd.DataFrame(),  # tracked_losses
            [],  # tracked_bias
        )

        result, events = quick_fault_detector(
            csv_data_path=self.train_path,
            csv_test_data_path=self.test_path,
            time_column_name="timestamp",
            automatic_optimization=False,
            min_anomaly_length=18,
            save_dir=self.tmp_dir,
        )

        self.assertGreater(len(events), 0)
        mock_model.run_root_cause_analysis.assert_called()

    @patch("energy_fault_detector.quick_fault_detection.quick_fault_detector.generate_output_plots")
    @patch("energy_fault_detector.quick_fault_detection.quick_fault_detector.FaultDetector")
    def test_features_to_exclude_passed_to_config(self, MockFaultDetector, mock_plots):
        """Verify features_to_exclude reaches the config."""
        n_rows = 200
        mock_model = MockFaultDetector.return_value
        mock_model.fit.return_value = None
        mock_result = MagicMock(spec=FaultDetectionResult)
        mock_result.predicted_anomalies = pd.Series(
            [False] * n_rows,
            index=pd.date_range("2024-01-01", periods=n_rows, freq="10min"),
            name="anomaly",
        )
        mock_model.predict.return_value = mock_result

        quick_fault_detector(
            csv_data_path=self.train_path,
            csv_test_data_path=self.test_path,
            time_column_name="timestamp",
            features_to_exclude=["sensor_0"],
            automatic_optimization=False,
            save_dir=self.tmp_dir,
        )

        # Verify the config passed to FaultDetector contains the exclusion
        call_kwargs = MockFaultDetector.call_args
        config = call_kwargs[1]["config"] if "config" in call_kwargs[1] else call_kwargs[0][0]
        # Check that column_selector step has sensor_0 excluded
        steps = config["train"]["data_preprocessor"]["steps"]
        col_selector = next(s for s in steps if s["name"] == "column_selector")
        self.assertIn("sensor_0", col_selector["params"]["features_to_exclude"])


class TestCLI(unittest.TestCase):
    """Test the CLI entry point."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.train_path, self.test_path = _make_csv_files(self.tmp_dir)

    @patch("energy_fault_detector.quick_fault_detection.quick_fault_detector.generate_output_plots")
    @patch("energy_fault_detector.quick_fault_detection.quick_fault_detector.FaultDetector")
    @patch("energy_fault_detector.quick_fault_detection.quick_fault_detector.load_train_test_data")
    def test_cli_invocation(self, mock_load, MockFaultDetector, mock_plots):
        """CLI parses args and calls quick_fault_detector."""
        from energy_fault_detector.main import main
        import sys

        # Mock data loading
        n_rows = 200
        index = pd.date_range("2024-01-01", periods=n_rows, freq="10min")
        train_data = pd.DataFrame(np.random.randn(n_rows, 5), index=index,
                                  columns=[f"s_{i}" for i in range(5)])
        test_data = train_data.copy()
        mock_load.return_value = (train_data, None, test_data)

        # Mock FaultDetector
        mock_model = MockFaultDetector.return_value
        mock_model.fit.return_value = None
        mock_result = MagicMock(spec=FaultDetectionResult)
        mock_result.predicted_anomalies = pd.Series([False] * n_rows, index=index, name="anomaly")
        mock_result.save = MagicMock()
        mock_model.predict.return_value = mock_result

        results_dir = os.path.join(self.tmp_dir, "results")

        with patch.object(
                sys, "argv",
                ["quick_fault_detector", str(self.train_path), "--results_dir", results_dir],
        ):
            main()

        mock_load.assert_called_once()
        MockFaultDetector.assert_called_once()
        mock_model.predict.assert_called_once()


class TestAnalyzeEvent(unittest.TestCase):
    """Unit tests for the analyze_event helper."""

    def test_returns_importances_and_losses(self):
        mock_detector = MagicMock()
        event_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=["a", "b", "c"],
            index=pd.date_range("2024-01-01", periods=20, freq="10min"),
        )
        bias = pd.DataFrame(
            np.array([[0.1, -0.5, 0.0]] * 20),
            columns=["a", "b", "c"],
            index=event_data.index,
        )
        mock_detector.run_root_cause_analysis.return_value = (bias, pd.DataFrame(), [])

        importances, losses = analyze_event(mock_detector, event_data, track_losses=False)

        self.assertEqual(len(importances), 3)
        # calculate_mean_arcana_importances normalizes: 0.5 / (0.1 + 0.5 + 0.0)
        self.assertAlmostEqual(importances["b"], 0.5 / 0.6, places=5)
        self.assertAlmostEqual(importances["a"], 0.1 / 0.6, places=5)
        self.assertAlmostEqual(importances["c"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
