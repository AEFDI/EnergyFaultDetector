import unittest

import pandas as pd

from energy_fault_detector.root_cause_analysis.arcana_utils import calculate_mean_arcana_importances


class TestArcanaUtils(unittest.TestCase):

    def test_calculate_arcana_importances(self):
        data = {
            'feature1': [0.1, 0.2, 0.3],
            'feature2': [0.4, 0.5, 0.6],
            'feature3': [0.7, 0.8, 0.9]
        }
        bias_data = pd.DataFrame(data, index=pd.date_range("2023-01-01", periods=3))

        # Sample normal_index
        normal_index = pd.Series(index=bias_data.index, data=[True, False, True])

        # Test without start and end
        importances = calculate_mean_arcana_importances(bias_data)
        relative_importances = bias_data.abs()
        sums = bias_data.abs().sum(axis=1)
        for i, sum_value in enumerate(sums):
            relative_importances.iloc[i] /= sum_value
        expected_importances = relative_importances.mean(axis=0).sort_values(ascending=True)

        pd.testing.assert_series_equal(importances, expected_importances)

        # Test with start and end
        importances = calculate_mean_arcana_importances(bias_data, start="2023-01-01", end="2023-01-02")
        expected_importances = relative_importances.loc["2023-01-01":"2023-01-02"].mean(axis=0).sort_values(
            ascending=True)
        pd.testing.assert_series_equal(importances, expected_importances)
