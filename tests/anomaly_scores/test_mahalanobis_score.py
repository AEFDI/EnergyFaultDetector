
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.anomaly_scores.mahalanobis_score import MahalanobisScore


class TestMahalanobisScore(TestCase):
    def setUp(self) -> None:
        self.mahalanobis_score = MahalanobisScore(pca=True, scale=False)
        self.mahalanobis_score_no_pca = MahalanobisScore(pca=False, scale=False)
        self.mahalanobis_score_no_pca_scale = MahalanobisScore(pca=False, scale=True)
        self.mahalanobis_score_pca_scale = MahalanobisScore(pca=True, scale=True)
        self.train_data = pd.DataFrame([
            [1, 2, 3, 6],
            [4, 5, 6, 30],
            [7, 8, 9, 72],
            [10, 11, 12, 132]
        ])
        self.test_data = pd.DataFrame([
            [1, 5, 6, 5],
            [4, 8, 6, 20],
            [10, 2, 9, 100]
        ])

    def test_fit(self) -> None:

        self.mahalanobis_score.fit(self.train_data)

        assert_array_equal(np.array([5.5, 6.5, 7.5, 60.]), self.mahalanobis_score.mean_x_)
        self.assertIsNone(check_is_fitted(self.mahalanobis_score.pca_object))
        self.assertIsNone(check_is_fitted(self.mahalanobis_score.min_cov_det_object))

    def test_transform(self) -> None:
        self.mahalanobis_score.fit(self.train_data)
        score = self.mahalanobis_score.transform(self.test_data)
        assert_array_almost_equal(np.array([1.31064256, 0.68390328, 0.68390328]), score)

    def test_transform_no_pca(self) -> None:
        self.mahalanobis_score_no_pca.fit(self.train_data)
        score = self.mahalanobis_score_no_pca.transform(self.test_data)
        assert_array_almost_equal(np.array([5.49382716, 13.46666667, 13.46666667]), score)

    def test_transform_pca_scale(self) -> None:
        self.mahalanobis_score_pca_scale.fit(self.train_data)
        score = self.mahalanobis_score_pca_scale.transform(self.test_data)
        assert_array_almost_equal(np.array([0.720376, 0.102951, 0.102951]), score)

    def test_transform_no_pca_scale(self) -> None:
        self.mahalanobis_score_no_pca_scale.fit(self.train_data)
        score = self.mahalanobis_score_no_pca_scale.transform(self.test_data)
        assert_array_almost_equal(np.array([5.49382716, 13.46666667, 13.46666667]), score)

    def test_transform_not_fitted(self) -> None:
        with self.assertRaises(ValueError):
            self.mahalanobis_score.transform(self.test_data)
        with self.assertRaises(ValueError):
            self.mahalanobis_score_no_pca.transform(self.test_data)
