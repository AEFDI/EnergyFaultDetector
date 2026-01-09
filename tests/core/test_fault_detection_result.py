
import unittest
import tempfile
import pandas as pd
from pathlib import Path

from energy_fault_detector.core import FaultDetectionResult


class TestFaultDetectionResultSaveLoad(unittest.TestCase):

    def setUp(self):
        # Create sample data
        index = pd.date_range("2023-01-01", periods=5, freq="H")

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


if __name__ == "__main__":
    unittest.main()
