import unittest
from datetime import datetime, timedelta

import pandas as pd

from energy_fault_detector.utils import analysis


class TestAnalysis(unittest.TestCase):

    def make_time_index(self, n):
        start = datetime(2020, 1, 1)
        return pd.to_datetime([start + timedelta(minutes=15 * i) for i in range(n)])

    def test_create_anomaly_events_returns_correct_values(self):
        """Test if create_anomaly_events returns a DataFrame with correct values"""
        sensor_data = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]},
                                   index=pd.date_range('2022-01-01', freq='D', periods=3))
        predicted_anomalies = pd.DataFrame({'anomaly': [True, False, True]},
                                           index=sensor_data.index)
        event_data, _ = analysis.create_events(sensor_data=sensor_data,
                                               boolean_information=predicted_anomalies['anomaly'],
                                               min_event_length=1)
        expected_values = {'start': [datetime(2022, 1, 1), datetime(2022, 1, 3)],
                           'end': [datetime(2022, 1, 1), datetime(2022, 1, 3)],
                           'duration': [timedelta(days=0), timedelta(days=0)]}
        expected_result = pd.DataFrame(expected_values, index=[0, 1])
        pd.testing.assert_frame_equal(event_data, expected_result)

    def test_criticality_returns_correct_values(self):
        """Test if calculate_criticality returns a Series with correct values (Legacy Case)"""

        n = 11
        anomalies = pd.Series([True, False, True, True, True, True, True, False, False, False, False],
                              index=pd.date_range('2022-01-01', freq='D', periods=n))
        normal_idx = pd.Series([False, True, True, True, True, False, False, True, True, True, True],
                               index=pd.date_range('2022-01-01', freq='D', periods=n))
        result = analysis.calculate_criticality(anomalies, normal_idx)
        expected_values = [0, 0, 1, 2, 3, 3, 3, 2, 1, 0, 0]
        expected_result = pd.Series(expected_values, index=pd.date_range('2022-01-01', freq='D', periods=n))
        pd.testing.assert_series_equal(result, expected_result)

    def test_criticality_basic_increase_decrease(self):
        idx = self.make_time_index(6)
        anomalies = pd.Series([True, False, False, True, True, False], index=idx)
        normal = pd.Series([True] * 6, index=idx)
        result = analysis.calculate_criticality(anomalies=anomalies, normal_idx=normal, init_criticality=0,
                                                max_criticality=10)
        # Stepwise: +1 -> 1, -1 -> 0, -1 -> 0 (floor), +1 -> 1, +1 -> 2, -1 -> 1
        expected = pd.Series([1, 0, 0, 1, 2, 1], index=idx)
        pd.testing.assert_series_equal(result, expected)

    def test_criticality_with_non_normal_periods(self):
        idx = self.make_time_index(5)
        anomalies = pd.Series([False, True, True, False, True], index=idx)
        normal = pd.Series([True, False, False, True, True], index=idx)
        result = analysis.calculate_criticality(anomalies=anomalies, normal_idx=normal, init_criticality=2,
                                                max_criticality=5)
        # Steps where normal: t0 (-1) => 1; t1,t2 ignored; t3 (-1) => 0; t4 (+1) => 1
        expected = pd.Series([1, 1, 1, 0, 1], index=idx)
        pd.testing.assert_series_equal(result, expected)

    def test_criticality_bounds(self):
        idx = self.make_time_index(6)
        anomalies = pd.Series([True, True, True, True, False, False], index=idx)
        normal = pd.Series([True] * 6, index=idx)
        # Start high and cap at max, then decrease but not below 0
        result = analysis.calculate_criticality(anomalies=anomalies, normal_idx=normal, init_criticality=4,
                                                max_criticality=5)
        expected = pd.Series([5, 5, 5, 5, 4, 3], index=idx)
        pd.testing.assert_series_equal(result, expected)

    def test_criticality_length_mismatch_raises(self):
        idx1 = self.make_time_index(3)
        idx2 = self.make_time_index(4)
        anomalies = pd.Series([True, False, True], index=idx1)
        normal = pd.Series([True, True, True, True], index=idx2)
        # does not raise an error
        analysis.calculate_criticality(anomalies=anomalies, normal_idx=normal)

