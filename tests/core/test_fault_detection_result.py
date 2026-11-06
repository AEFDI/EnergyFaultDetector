
import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from energy_fault_detector.core import FaultDetectionResult


class TestFaultDetectionResultSaveLoad(unittest.TestCase):

    def setUp(self):
        # Create sample data
        index = pd.date_range("2023-01-01", periods=5, freq="h")

        self.predicted_anomalies = pd.Series([False, True, False, True, False], index=index, name="anomaly")
        self.reconstruction = pd.DataFrame({
            "sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sensor_2": [2.0, 3.0, 4.0, 5.0, 6.0]
        }, index=index)
        self.recon_error = pd.DataFrame({
            "sensor_1": [0.1, 0.2, 0.1, 0.3, 0.1],
            "sensor_2": [0.2, 0.1, 0.3, 0.1, 0.2]
        }, index=index)
        self.anomaly_score = pd.Series([0.1, 0.9, 0.2, 0.8, 0.15], index=index, name="score")

        # Optional fields
        self.bias_data = pd.DataFrame({"bias": [0.01, 0.02]}, index=index[:2])
        self.arcana_losses = pd.DataFrame({"loss_a": [0.1, 0.2], "loss_b": [0.05, 0.1]}, index=index[:2])
        self.tracked_bias = [
            pd.DataFrame({"bias_step_0": [0.01]}, index=[index[0]]),
            pd.DataFrame({"bias_step_1": [0.02]}, index=[index[1]])
        ]

        # Instantiate object
        self.fdr = FaultDetectionResult(
            predicted_anomalies=self.predicted_anomalies,
            reconstruction=self.reconstruction,
            recon_error=self.recon_error,
            anomaly_score=self.anomaly_score,
            bias_data=self.bias_data,
            arcana_losses=self.arcana_losses,
            tracked_bias=self.tracked_bias
        )

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save result
            self.fdr.save(tmp_path)

            # Load result back
            loaded_fdr = FaultDetectionResult.load(tmp_path)

            # Compare core attributes
            pd.testing.assert_series_equal(loaded_fdr.predicted_anomalies, self.fdr.predicted_anomalies, check_freq=False)
            pd.testing.assert_frame_equal(loaded_fdr.reconstruction, self.fdr.reconstruction, check_freq=False)
            pd.testing.assert_frame_equal(loaded_fdr.recon_error, self.fdr.recon_error, check_freq=False)
            pd.testing.assert_series_equal(loaded_fdr.anomaly_score, self.fdr.anomaly_score, check_freq=False)

            # Compare optional attributes
            pd.testing.assert_frame_equal(loaded_fdr.bias_data, self.fdr.bias_data, check_freq=False)
            pd.testing.assert_frame_equal(loaded_fdr.arcana_losses, self.fdr.arcana_losses, check_freq=False)

            # Compare tracked_bias list of DataFrames
            self.assertEqual(len(loaded_fdr.tracked_bias), len(self.fdr.tracked_bias))
            for loaded_df, original_df in zip(loaded_fdr.tracked_bias, self.fdr.tracked_bias):
                pd.testing.assert_frame_equal(loaded_df, original_df, check_freq=False)


class TestFaultDetectionResultSaveLoadMultiIndex(unittest.TestCase):
    """Test save/load with MultiIndex (device_id, timestamp)."""

    def setUp(self):
        devices = ["turbine_A", "turbine_B"]
        timestamps = pd.date_range("2023-01-01", periods=5, freq="h")
        multi_idx = pd.MultiIndex.from_product(
            [devices, timestamps], names=["device_id", "timestamp"]
        )
        n = len(multi_idx)

        self.predicted_anomalies = pd.Series(
            [False, True, False, True, False] * 2, index=multi_idx, name="anomaly"
        )
        self.reconstruction = pd.DataFrame({
            "sensor_1": np.arange(n, dtype=float),
            "sensor_2": np.arange(n, dtype=float) + 10,
        }, index=multi_idx)
        self.recon_error = pd.DataFrame({
            "sensor_1": np.random.rand(n),
            "sensor_2": np.random.rand(n),
        }, index=multi_idx)
        self.anomaly_score = pd.Series(
            np.random.rand(n), index=multi_idx, name="score"
        )
        self.bias_data = pd.DataFrame({
            "sensor_1": np.random.rand(n),
            "sensor_2": np.random.rand(n),
        }, index=multi_idx)

        self.fdr = FaultDetectionResult(
            predicted_anomalies=self.predicted_anomalies,
            reconstruction=self.reconstruction,
            recon_error=self.recon_error,
            anomaly_score=self.anomaly_score,
            bias_data=self.bias_data,
        )

    def test_save_and_load_roundtrip_multiindex(self):
        """Test that MultiIndex is correctly preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.fdr.save(tmp_path)
            loaded = FaultDetectionResult.load(tmp_path)

            # Check index type is preserved
            self.assertIsInstance(loaded.predicted_anomalies.index, pd.MultiIndex)
            self.assertEqual(loaded.predicted_anomalies.index.nlevels, 2)
            self.assertEqual(list(loaded.predicted_anomalies.index.names), ["device_id", "timestamp"])

            # Check values
            pd.testing.assert_series_equal(
                loaded.predicted_anomalies, self.predicted_anomalies, check_freq=False
            )
            pd.testing.assert_frame_equal(
                loaded.reconstruction, self.reconstruction, check_freq=False
            )
            pd.testing.assert_frame_equal(
                loaded.recon_error, self.recon_error, check_freq=False
            )
            pd.testing.assert_series_equal(
                loaded.anomaly_score, self.anomaly_score, check_freq=False
            )
            pd.testing.assert_frame_equal(
                loaded.bias_data, self.bias_data, check_freq=False
            )

    def test_multiindex_meta_content(self):
        """Test that saved metadata correctly describes the MultiIndex."""
        import json

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.fdr.save(tmp_path)

            with open(tmp_path / "_index_meta.json") as f:
                meta = json.load(f)

            self.assertTrue(meta["is_multiindex"])
            self.assertEqual(meta["n_levels"], 2)
            self.assertEqual(meta["names"], ["device_id", "timestamp"])
            self.assertEqual(meta["datetime_levels"], [1])

    def test_multiindex_datetime_level_parsed(self):
        """Test that the datetime level is correctly parsed as DatetimeIndex on load."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.fdr.save(tmp_path)
            loaded = FaultDetectionResult.load(tmp_path)

            # The timestamp level should be datetime
            ts_level = loaded.predicted_anomalies.index.get_level_values("timestamp")
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(ts_level))

    def test_multiindex_group_level_preserved(self):
        """Test that the grouping level values are correctly preserved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.fdr.save(tmp_path)
            loaded = FaultDetectionResult.load(tmp_path)

            device_level = loaded.predicted_anomalies.index.get_level_values("device_id")
            self.assertEqual(set(device_level), {"turbine_A", "turbine_B"})

    def test_multiindex_boolean_dtype_preserved(self):
        """Test that predicted_anomalies remain boolean after round-trip."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.fdr.save(tmp_path)
            loaded = FaultDetectionResult.load(tmp_path)

            self.assertEqual(loaded.predicted_anomalies.dtype, bool)

    def test_multiindex_without_optional_fields(self):
        """Test MultiIndex save/load with no optional fields."""
        fdr_minimal = FaultDetectionResult(
            predicted_anomalies=self.predicted_anomalies,
            reconstruction=self.reconstruction,
            recon_error=self.recon_error,
            anomaly_score=self.anomaly_score,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fdr_minimal.save(tmp_path)
            loaded = FaultDetectionResult.load(tmp_path)

            self.assertIsInstance(loaded.reconstruction.index, pd.MultiIndex)
            self.assertIsNone(loaded.bias_data)
            self.assertIsNone(loaded.arcana_losses)
            self.assertIsNone(loaded.tracked_bias)


class TestFaultDetectionResultCriticality(unittest.TestCase):
    """Test the alarm/criticality derivation on FaultDetectionResult."""

    def _make_result(self, anomalies):
        return FaultDetectionResult(
            predicted_anomalies=anomalies,
            reconstruction=pd.DataFrame(index=anomalies.index),
            recon_error=pd.DataFrame(index=anomalies.index),
            anomaly_score=pd.Series(0.0, index=anomalies.index),
        )

    def test_criticality_returns_series(self):
        """The legacy criticality method returns only the criticality series."""
        idx = pd.date_range("2024-01-01", periods=6, freq="10min")
        anomalies = pd.Series([True, True, True, False, False, False], index=idx)

        result = self._make_result(anomalies)
        criticality = result.criticality()

        self.assertIsInstance(criticality, pd.Series)
        self.assertEqual(criticality.index.equals(idx), True)
        self.assertEqual(list(criticality), [1, 2, 3, 2, 1, 0])

    def test_criticality_supports_max_criticality(self):
        """The legacy criticality method caps the counter at max_criticality."""
        idx = pd.date_range("2024-01-01", periods=5, freq="10min")
        anomalies = pd.Series([True] * 5, index=idx)

        result = self._make_result(anomalies)
        criticality = result.criticality(max_criticality=3)

        self.assertEqual(list(criticality), [1, 2, 3, 3, 3])

    def test_criticality_alarms_returns_events_and_series(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="10min")
        anomalies = pd.Series(False, index=idx)
        anomalies.iloc[2:10] = True

        result = self._make_result(anomalies)
        alarm_events, criticality = result.criticality_alarms()

        self.assertIsInstance(alarm_events, pd.DataFrame)
        self.assertIsInstance(criticality, pd.Series)
        self.assertEqual(list(alarm_events.columns), ["start", "end", "duration"])
        self.assertEqual(criticality.index.equals(idx), True)
        self.assertEqual(alarm_events.iloc[0]["start"], idx[7])

    def test_criticality_alarms_uses_normal_idx_and_threshold(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="10min")
        anomalies = pd.Series(True, index=idx)
        normal_idx = pd.Series([True] * 5 + [False] * 5, index=idx)

        result = self._make_result(anomalies)
        alarm_events, criticality = result.criticality_alarms(
            normal_idx=normal_idx, criticality_threshold=3, alarm_reset_time_minutes=20
        )

        # max criticality = 3 + int(20 / 10) = 5; non-normal timestamps keep the value frozen
        self.assertEqual(list(criticality), [1, 2, 3, 4, 5, 5, 5, 5, 5, 5])
        self.assertEqual(alarm_events.iloc[0]["start"], idx[2])


if __name__ == "__main__":
    unittest.main()
