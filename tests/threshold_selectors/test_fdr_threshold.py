import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.utils.validation import NotFittedError

from energy_fault_detector.threshold_selectors.fdr_threshold import FDRSelector
from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore


class TestFDRSelector(TestCase):
    def setUp(self) -> None:
        self.threshold_selector = FDRSelector(target_false_discovery_rate=0.2)

        # input
        self.train_data = pd.DataFrame(np.arange(1, 100).reshape(33, 3) / 100)
        self.normal_index = pd.Series([False]*4 + [True]*25 + [False]*4, name='normal')

        # rmse object needed for testing
        self.rmse = RMSEScore()
        self.rmse.fit(self.train_data)

        # save location
        self.save_location = 'saved_models'

    def tearDown(self) -> None:
        shutil.rmtree(self.save_location, ignore_errors=True)

    def test_init(self) -> None:
        self.assertEqual({'target_false_discovery_rate': 0.2},
                         self.threshold_selector.get_params())

    def test_fit(self) -> None:
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scores, self.normal_index)
        self.assertAlmostEqual(1.155231069, self.threshold_selector.threshold)
        self.assertEqual(self.threshold_selector.actual_false_discovery_rate_, 0.2)

    def test_failed_fit(self) -> None:
        scores = self.rmse.transform(self.train_data)
        threshold_selector = FDRSelector(target_false_discovery_rate=0.9)
        with self.assertLogs('energy_fault_detector', level='WARNING') as cm:
            threshold_selector.fit(scores, self.normal_index)
            self.assertEqual(cm.output, ['WARNING:energy_fault_detector:Could not find suitable threshold,'
                                         ' `threshold` is set to max score.'])

        self.assertAlmostEqual(threshold_selector.threshold, 1.680336101)

    def test_predict(self) -> None:
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scores, self.normal_index)
        anomalies = self.threshold_selector.predict(scores)

        assert_array_equal(anomalies,
                           [True]*5 + [False]*23 + [True]*5)

    def test_not_fitted(self) -> None:
        with self.assertRaises(NotFittedError):
            self.threshold_selector.predict(self.train_data)

    def test_save_and_load(self) -> None:
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scores, self.normal_index).save(self.save_location)
        new_ts = FDRSelector()
        new_ts.load(self.save_location)
        self.assertEqual(self.threshold_selector.target_false_discovery_rate, new_ts.target_false_discovery_rate)
        self.assertEqual(self.threshold_selector.threshold, new_ts.threshold)
        self.assertEqual(self.threshold_selector.actual_false_discovery_rate_, new_ts.actual_false_discovery_rate_)

    def test_overwrite(self) -> None:
        self.threshold_selector.save(self.save_location)
        self.assertListEqual(os.listdir(self.save_location), ['FDRSelector.pkl'])

        with self.assertRaises(FileExistsError):  # overwrite = False
            self.threshold_selector.save(self.save_location)

        # does not raise error
        self.threshold_selector.save(self.save_location, overwrite=True)
        self.assertListEqual(os.listdir(self.save_location), ['FDRSelector.pkl'])
        # save with a different file name
        self.threshold_selector.save(self.save_location, file_name='another_threshold_selector.pkl')
        self.assertListEqual(sorted(os.listdir(self.save_location)),
                             sorted(['another_threshold_selector.pkl', 'FDRSelector.pkl']))
