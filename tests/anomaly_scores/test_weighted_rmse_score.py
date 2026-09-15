from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal

from energy_fault_detector.anomaly_scores.weighted_rmse_score import WeightedRMSEScore
from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore


class TestWeightedRMSEScore(TestCase):
    def setUp(self) -> None:
        self.train_data = pd.DataFrame(
            {"feature_A": [1, 2, 3],
             "feature_B": [2, 4, 6],
             "feature_C": [4, 8, 12],
             }
        )
        self.test_data = pd.DataFrame(
                    {"feature_A": [3, 4, 1],
                     "feature_B": [2, 5, 5],
                     "feature_C": [9, 6, 5],
                     }
                )
        self.feature_weights = {"feature_A": 2.0, "feature_B": 1.0, "feature_C": 0.5}
        self.weighted_rmse_score = WeightedRMSEScore(feature_weights=self.feature_weights)
        self.rmse_score = RMSEScore()

    def test_fit(self) -> None:
        self.rmse_score.fit(self.train_data)
        self.weighted_rmse_score.fit(self.train_data)
        assert_array_equal(self.rmse_score.mean_x_, self.weighted_rmse_score.mean_x_)
        assert_array_equal(self.rmse_score.std_x_, self.weighted_rmse_score.std_x_)
        assert_array_almost_equal(self.weighted_rmse_score.mean_x_weighted_.values,
                                  np.array([4.0, 4.0, 4.0]))
        assert_array_almost_equal(self.weighted_rmse_score.std_x_weighted_.values,
                                          np.array([2.0, 2.0, 2.0]))
        self.assertTrue(self.weighted_rmse_score.fitted_)

    def test_transform(self) -> None:
        self.weighted_rmse_score.fit(self.train_data)
        weighted_score = self.weighted_rmse_score.transform(self.test_data)
        assert_array_almost_equal(weighted_score, np.array([0.829156, 1.224745, 0.777282]))
