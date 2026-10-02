import os
import pickle
import tempfile
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal

from energy_fault_detector.anomaly_scores.weighted_rmse_score import WeightedRMSEScore
from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore


class TestWeightedRMSEScore(TestCase):
    def setUp(self) -> None:
        self.train_data = pd.DataFrame(
            {"feature_A": [0, 0, 1],
             "feature_B": [0, 1, 0],
             "feature_C": [1, 0, 0],
             }
        )
        self.test_data = pd.DataFrame(
                    {"feature_A": [1, 0, 0, 1, 0, 1, 1],
                     "feature_B": [0, 1, 0, 1, 1, 0, 1],
                     "feature_C": [0, 0, 1, 0, 1, 1, 1],
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
        self.assertTrue(self.weighted_rmse_score.fitted_)

    def test_transform(self) -> None:
        self.weighted_rmse_score.fit(self.train_data)
        weighted_score = self.weighted_rmse_score.transform(self.test_data)
        assert_array_almost_equal(weighted_score, np.array([1.603567, 1.133893, 0.801784, 1.963961, 1.38873 , 1.792843, 2.12132]))

    def test_instantiate_without_feature_weights(self) -> None:
        """The scorer must be constructible without arguments.

        This is required by the FaultDetector load mechanism, which reconstructs each
        sub-model with `model_class()` (no arguments) before restoring the pickled state.
        Before a default value was added for `feature_weights`, this raised a TypeError.
        """
        score = WeightedRMSEScore()
        self.assertEqual(score.feature_weights, {})

    def test_feature_weights_restored_on_load(self) -> None:
        """`feature_weights` must be restored from the pickled state on load."""
        self.weighted_rmse_score.fit(self.train_data)

        loaded_dict = pickle.loads(pickle.dumps(self.weighted_rmse_score.__dict__))
        self.assertEqual(loaded_dict['feature_weights'], self.feature_weights)

    def test_save_load_via_save_load_mixin(self) -> None:
        """The SaveLoadMixin save/load path (used by FaultDetector) restores feature_weights."""
        tmp_dir = tempfile.mkdtemp()
        try:
            self.weighted_rmse_score.fit(self.train_data)
            self.weighted_rmse_score.save(tmp_dir)

            loaded_score = WeightedRMSEScore()
            loaded_score.load(tmp_dir)

            self.assertIsInstance(loaded_score, WeightedRMSEScore)
            self.assertEqual(loaded_score.feature_weights, self.feature_weights)
            self.assertTrue(loaded_score.fitted_)
        finally:
            shutil.rmtree(tmp_dir)

    def test_test_transform_with_incomplete_feature_weights(self) -> None:
        """If a feature is missing from the weights, it should be treated as weight=1.0."""
        incomplete_weights = {"feature_A": 2.0, "feature_B": 1.0}  # feature_C is missing
        weighted_rmse_score = WeightedRMSEScore(feature_weights=incomplete_weights)
        weighted_rmse_score.fit(self.train_data)
        weighted_score = weighted_rmse_score.transform(self.test_data)

        # The expected score is calculated with feature_C having weight=1.0
        expected_score = np.array([1.5, 1.06066 , 1.06066 , 1.837117, 1.5, 1.837117, 2.12132])
        assert_array_almost_equal(weighted_score, expected_score)
