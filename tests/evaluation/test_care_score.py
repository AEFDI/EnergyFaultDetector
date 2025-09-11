import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from energy_fault_detector.evaluation.care_score import CAREScore


class TestCARE(unittest.TestCase):
    """Unit tests for the CARE class."""

    def setUp(self) -> None:
        """Set up a temporary directory and create a CARE instance for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.care = CAREScore()

        self.beta = 0.5
        self.test_good_prediction_df = pd.DataFrame(
            data=[
                [1, 'anomaly', 0.5, 0.869565, 0.9, 73, 8, 1, 2, "Wind Farm A"],
                [2, 'anomaly', 0.8, 0.625000, 0.5, 73, 5, 3, 3, "Wind Farm B"],
                [3, 'normal', np.nan, np.nan, 0.8, 77, 0, 1, 0, "Wind Farm A"],
                [4, 'normal', np.nan, np.nan, 0.5, 100, 0, 4, 0, "Wind Farm B"],
            ],
            columns=['event_id', 'event_label', 'weighted_score', 'f_beta_score', 'accuracy',
                     'max_criticality', 'tp', 'fp', 'fn', 'park']
        )
        self.expected_scores_good = [0.650567632850241, 0.705024154589372, 0.5]

        self.test_bad_prediction_df = pd.DataFrame(
            data=[
                [1, 'anomaly', 0.5, 0.625000, 0.7, 73, 5, 3, 3, "Wind Farm A"],
                [2, 'anomaly', 0.8, 0.731707, 0.5, 73, 6, 2, 3, "Wind Farm B"],
                [3, 'normal', np.nan, np.nan, 0.1, 77, 0, 10, 1, "Wind Farm A"],
                [4, 'normal', np.nan, np.nan, 0.0, 90, 0, 11, 0, "Wind Farm B"],
            ],
            columns=['event_id', 'event_label', 'weighted_score', 'f_beta_score', 'accuracy',
                     'max_criticality', 'tp', 'fp', 'fn', 'park']
        )
        self.expected_scores_bad = [0.05, 0.1, 0.0]

    def tearDown(self) -> None:
        """Clean up the temporary directory after tests."""
        self.temp_dir.cleanup()

    def test_initialization_with_invalid_anomaly_detection_method(self):
        """Test initialization raises ValueError for invalid anomaly detection method."""
        with self.assertRaises(ValueError) as context:
            CAREScore(anomaly_detection_method='invalid_method')
        self.assertEqual(str(context.exception), "Anomaly detection method must be either 'criticality' or 'fraction'")

    def test_evaluate_event_with_invalid_event_label(self):
        """Test evaluate_event raises ValueError for an unknown event label."""
        normal_index = pd.Series([True, True, False])
        predicted_anomalies = pd.Series([False, True, True])
        with self.assertRaises(ValueError) as context:
            self.care.evaluate_event(0, 2, 'unknown_label', normal_index, predicted_anomalies)
        self.assertEqual(str(context.exception), 'Unknown event label (should be either `anomaly` or `normal`')

    def test_evaluate_event_for_anomaly(self):
        """Test evaluate_event correctly evaluates an anomaly event."""
        normal_index = pd.Series([True, True, False])
        predicted_anomalies = pd.Series([False, True, True])
        evaluation = self.care.evaluate_event(0, 2, 'anomaly', normal_index, predicted_anomalies)
        self.assertIn('event_id', evaluation)
        self.assertEqual(evaluation['max_criticality'], 1)
        self.assertEqual(evaluation['tp'], 1)
        self.assertEqual(evaluation['fn'], 1)

    def test_evaluate_event_with_ignore_normal_index(self):
        """Test evaluate_event with ignore_normal_index set to True."""
        normal_index = pd.Series([True, True, False, True, False])
        predicted_anomalies = pd.Series([False, True, True, False, True])

        # Evaluate the event while ignoring normal index
        evaluation = self.care.evaluate_event(0, 4, 'anomaly',
                                              normal_index, predicted_anomalies,
                                              ignore_normal_index=True)

        # Check if the evaluation contains the expected keys
        self.assertIn('event_id', evaluation)
        self.assertIn('max_criticality', evaluation)
        self.assertEqual(evaluation['event_label'], 'anomaly')

        # Check that the evaluation metrics are calculated correctly
        # Adjust the expected values as necessary based on your specific logic
        self.assertEqual(evaluation['tp'], 3)
        self.assertEqual(evaluation['fn'], 2)

    def test_get_final_score_with_no_anomalies(self):
        """Test get_final_score returns 0 when no anomalies are detected."""
        self.care._evaluated_events = [{'event_id': 0, 'event_label': 'normal',
                                        'max_criticality': 50}]
        score = self.care.get_final_score()
        self.assertEqual(score, 0.0)
        self.assertEqual(self.care.calculate_reliability(), 0.0)

    def test_get_final_score_with_low_accuracy(self):
        """Test get_final_score returns 0 when no anomalies are detected."""
        self.care._evaluated_events = [{'event_id': 0, 'event_label': 'normal',
                                        'max_criticality': 73, 'accuracy': 0.5}]
        score = self.care.get_final_score()
        self.assertEqual(score, 0.5)
        self.assertEqual(self.care.calculate_avg_accuracy(), 0.5)

    def test_save_and_load_evaluated_events(self):
        """Test saving and loading evaluated events."""
        # Create a sample evaluated event
        sample_event = {'event_id': 0, 'event_label': 'anomaly', 'max_criticality': 20}
        self.care._evaluated_events.append(sample_event)

        # Save to a temporary file
        file_path = Path(self.temp_dir.name) / 'evaluated_events.csv'
        self.care.save_evaluated_events(file_path)

        # Create a new CARE instance and load the events
        new_care = CAREScore()
        new_care.load_evaluated_events(file_path)

        # Check if the loaded events match the saved events
        self.assertEqual(len(new_care.evaluated_events), 1)
        self.assertEqual(new_care.evaluated_events.iloc[0]['event_label'], 'anomaly')

    def test_final_score(self):
        self.care._evaluated_events = self.test_good_prediction_df.to_dict(orient='records')
        self.care.eventwise_f_score_beta = 0.5
        self.care.coverage_beta = 0.5
        self.care.criticality_threshold = 72

        self.assertAlmostEqual(self.care.calculate_avg_coverage(), 0.7472825)
        self.assertAlmostEqual(self.care.calculate_avg_weighted_score(), 0.65)
        self.assertAlmostEqual(self.care.calculate_reliability(), 0.55555556)
        self.assertAlmostEqual(self.care.calculate_avg_accuracy(), 0.65)

        self.assertAlmostEqual(self.care.get_final_score(), self.expected_scores_good[0])
        self.assertAlmostEqual(self.care.get_final_score([1, 3]), self.expected_scores_good[1])  # A
        self.assertAlmostEqual(self.care.get_final_score([2, 4]), self.expected_scores_good[2])  # B

        self.assertAlmostEqual(self.care.get_final_score(criticality_threshold=100), 0.5394565)

        self.care._evaluated_events = self.test_bad_prediction_df.to_dict(orient='records')
        self.care.eventwise_f_score_beta = 0.5
        self.care.coverage_beta = 0.5
        self.care.criticality_threshold = 72

        self.assertAlmostEqual(self.care.get_final_score(), self.expected_scores_bad[0])
        self.assertAlmostEqual(self.care.get_final_score([1, 3]), self.expected_scores_bad[1])  # A
        self.assertAlmostEqual(self.care.get_final_score([2, 4]), self.expected_scores_bad[2])  # B
