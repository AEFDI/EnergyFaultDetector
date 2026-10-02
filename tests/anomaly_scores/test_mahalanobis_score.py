
import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.covariance import LedoitWolf, OAS
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.anomaly_scores.mahalanobis_score import MahalanobisScore


class TestMahalanobisScore(TestCase):
    def setUp(self) -> None:
        self.mahalanobis_score = MahalanobisScore(pca=True)
        self.mahalanobis_score_no_pca = MahalanobisScore(pca=False)
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
        assert_array_almost_equal(np.array([3.35410197, 3.35410197, 3.35410197, 47.812132]),
                                  self.mahalanobis_score.std_x_)
        self.assertIsNone(check_is_fitted(self.mahalanobis_score.pca_object))
        self.assertIsNone(check_is_fitted(self.mahalanobis_score.min_cov_det_object))

    def test_transform(self) -> None:
        self.mahalanobis_score.fit(self.train_data)
        score = self.mahalanobis_score.transform(self.test_data)
        assert_array_almost_equal(np.array([0.720376, 0.102951, 0.102951]), score)

    def test_transform_no_pca(self) -> None:
        # Explicit MinCovDet on the mean-centred full-dimensionality errors.
        self.mahalanobis_score_no_pca.covariance_method = "min_cov_det"
        self.mahalanobis_score_no_pca.fit(self.train_data)
        score = self.mahalanobis_score_no_pca.transform(self.test_data)
        assert_array_almost_equal(np.array([5.49382716, 13.46666667, 13.46666667]), score)

    def test_transform_no_pca_auto_falls_back_to_shrinkage(self) -> None:
        # train_data has n/p = 1 (4 samples, 4 features) which is below the MinCovDet
        # stability threshold, so the clean path falls back to a shrinkage estimator.
        self.mahalanobis_score_no_pca.fit(self.train_data)
        self.assertEqual(self.mahalanobis_score_no_pca.covariance_method_, "shrinkage")
        self.assertIsInstance(self.mahalanobis_score_no_pca.cov_estimator_, OAS)
        score = self.mahalanobis_score_no_pca.transform(self.test_data)
        assert_array_almost_equal(np.array([2.22067872, 1.20833788, 1.32463100]), score)

    def test_transform_not_fitted(self) -> None:
        with self.assertRaises(ValueError):
            self.mahalanobis_score.transform(self.test_data)
        with self.assertRaises(ValueError):
            self.mahalanobis_score_no_pca.transform(self.test_data)

    def test_pca_is_deprecated(self) -> None:
        with self.assertWarns(DeprecationWarning):
            MahalanobisScore(pca=True)

    def test_no_pca_is_not_deprecated(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MahalanobisScore(pca=False)

    def test_scale_is_deprecated(self) -> None:
        with self.assertWarns(DeprecationWarning):
            MahalanobisScore(scale=False)

    def test_scale_has_no_effect(self) -> None:
        # the deprecated `scale` parameter no longer has any effect.
        with self.assertWarns(DeprecationWarning):
            mahalanobis_deprecated = MahalanobisScore(pca=True, scale=False)

        self.mahalanobis_score.fit(self.train_data)
        mahalanobis_deprecated.fit(self.train_data)

        score_default = self.mahalanobis_score.transform(self.test_data)
        score_deprecated = mahalanobis_deprecated.transform(self.test_data)
        assert_array_almost_equal(score_default, score_deprecated)

    def test_invalid_covariance_method(self) -> None:
        with self.assertRaises(ValueError):
            MahalanobisScore(pca=False, covariance_method="bogus")

    def test_invalid_shrinkage_method(self) -> None:
        with self.assertRaises(ValueError):
            MahalanobisScore(pca=False, shrinkage_method="bogus")

    def test_auto_uses_min_cov_det_when_ratio_is_sufficient(self) -> None:
        # n/p = 20 >= min_cov_det_ratio (2.0) and the covariance is well conditioned.
        rng = np.random.RandomState(0)
        data = pd.DataFrame(rng.randn(100, 5))
        score = MahalanobisScore(pca=False, covariance_method="auto")
        score.fit(data)
        self.assertEqual(score.covariance_method_, "min_cov_det")
        self.assertIs(score.cov_estimator_, score.min_cov_det_object)
        score.transform(data.head(3))

    def test_auto_uses_shrinkage_when_ratio_is_insufficient(self) -> None:
        # n/p = 1 < min_cov_det_ratio (2.0) -> shrinkage fallback.
        rng = np.random.RandomState(0)
        data = pd.DataFrame(rng.randn(8, 5))
        score = MahalanobisScore(pca=False, covariance_method="auto")
        score.fit(data)
        self.assertEqual(score.covariance_method_, "shrinkage")
        self.assertIsInstance(score.cov_estimator_, OAS)

    def test_auto_uses_min_cov_det_with_custom_ratio(self) -> None:
        # With a low enough ratio threshold, MinCovDet is used even at n/p = 1.6.
        rng = np.random.RandomState(0)
        data = pd.DataFrame(rng.randn(8, 5))
        score = MahalanobisScore(pca=False, covariance_method="auto", min_cov_det_ratio=1.5)
        score.fit(data)
        self.assertEqual(score.covariance_method_, "min_cov_det")

    def test_auto_falls_back_to_shrinkage_on_degenerate_covariance(self) -> None:
        # n/p is sufficient (10 >= 2) but the last feature is a perfect linear combination
        # of the first two, so the MinCovDet covariance is singular -> fall back to shrinkage.
        rng = np.random.RandomState(0)
        mat = np.column_stack([rng.randn(50, 3), rng.randn(50, 1)])
        mat = np.column_stack([mat, mat[:, 0] + mat[:, 1]])
        data = pd.DataFrame(mat)
        self.assertEqual(data.shape[0] / data.shape[1], 10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = MahalanobisScore(pca=False, covariance_method="auto")
            score.fit(data)
        self.assertEqual(score.covariance_method_, "shrinkage")
        self.assertIsInstance(score.cov_estimator_, OAS)

    def test_explicit_shrinkage_oas(self) -> None:
        rng = np.random.RandomState(0)
        data = pd.DataFrame(rng.randn(100, 5))
        score = MahalanobisScore(pca=False, covariance_method="shrinkage", shrinkage_method="oas")
        score.fit(data)
        self.assertEqual(score.covariance_method_, "shrinkage")
        self.assertIsInstance(score.cov_estimator_, OAS)
        result = score.transform(data.head(3))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_explicit_shrinkage_ledoit_wolf(self) -> None:
        rng = np.random.RandomState(0)
        data = pd.DataFrame(rng.randn(100, 5))
        score = MahalanobisScore(pca=False, covariance_method="shrinkage", shrinkage_method="ledoit_wolf")
        score.fit(data)
        self.assertEqual(score.covariance_method_, "shrinkage")
        self.assertIsInstance(score.cov_estimator_, LedoitWolf)
        result = score.transform(data.head(3))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_explicit_min_cov_det(self) -> None:
        rng = np.random.RandomState(0)
        data = pd.DataFrame(rng.randn(100, 5))
        score = MahalanobisScore(pca=False, covariance_method="min_cov_det")
        score.fit(data)
        self.assertEqual(score.covariance_method_, "min_cov_det")
        self.assertIs(score.cov_estimator_, score.min_cov_det_object)
        result = score.transform(data.head(3))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_shrinkage_handles_high_dimensional_data(self) -> None:
        # Shrinkage remains well defined even when p approaches / exceeds n.
        rng = np.random.RandomState(1)
        data = pd.DataFrame(rng.randn(50, 120))
        score = MahalanobisScore(pca=False, covariance_method="shrinkage", shrinkage_method="oas")
        score.fit(data)
        self.assertEqual(score.covariance_method_, "shrinkage")
        result = score.transform(data.head(5))
        self.assertTrue(np.all(np.isfinite(result)))
