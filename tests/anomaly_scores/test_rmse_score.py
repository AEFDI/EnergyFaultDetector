
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal

from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore


class TestRMSEScore(TestCase):
    def setUp(self) -> None:
        self.rmse_score = RMSEScore()
        self.train_data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        self.test_data = np.array([
            [1, 5, 6],
            [4, 8, 6],
        ])

    def test_fit(self) -> None:

        self.rmse_score.fit(self.train_data)
        assert_array_equal(np.array([4., 5., 6.]), self.rmse_score.mean_x_)
        assert_array_almost_equal(np.array([2.44948974, 2.44948974, 2.44948974]), self.rmse_score.std_x_)

        self.assertTrue(self.rmse_score.fitted_)

    def test_transform(self) -> None:
        self.rmse_score.fit(self.train_data)
        score = self.rmse_score.transform(self.test_data)
        assert_array_almost_equal(np.array([0.70710678, 0.70710678]), score)

    def test_transform_not_fitted(self) -> None:
        with self.assertRaises(ValueError):
            self.rmse_score.transform(self.test_data)

    def test_transform_with_zero_std_feature(self) -> None:
        """A feature with std 0 in the training data must not crash a later transform.

        The std-0 feature is left unscaled (dividing by 0 would produce inf/nan),
        while the remaining features are still standardized.
        """
        train_data = pd.DataFrame({
            "feature_A": [1.0, 2.0, 3.0],
            "feature_B": [5.0, 5.0, 5.0],  # std == 0
        })
        test_data = pd.DataFrame({
            "feature_A": [1.0, 5.0],
            "feature_B": [4.0, 6.0],
        })
        self.rmse_score.fit(train_data)
        with self.assertLogs('energy_fault_detector', level='WARNING') as captured:
            score = self.rmse_score.transform(test_data)
        self.assertTrue(np.isfinite(score).all())
        assert_array_almost_equal(score, np.array([2.95803989, 6.06217783]))
        self.assertTrue(any(
            'standard deviation of 0.0' in message for message in captured.output
        ))

    def test_scale_is_deprecated(self) -> None:
        with self.assertWarns(DeprecationWarning):
            RMSEScore(scale=False)

    def test_scale_has_no_effect(self) -> None:
        # the deprecated `scale` parameter no longer has any effect.
        with self.assertWarns(DeprecationWarning):
            rmse_deprecated = RMSEScore(scale=False)

        self.rmse_score.fit(self.train_data)
        rmse_deprecated.fit(self.train_data)

        score_default = self.rmse_score.transform(self.test_data)
        score_deprecated = rmse_deprecated.transform(self.test_data)
        assert_array_almost_equal(score_default, score_deprecated)
