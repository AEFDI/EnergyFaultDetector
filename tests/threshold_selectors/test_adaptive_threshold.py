from unittest import TestCase

from datetime import datetime
import numpy as np
import pandas as pd
import keras
from numpy.testing import assert_array_equal

from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore
from energy_fault_detector.threshold_selectors.adaptive_threshold import AdaptiveThresholdSelector


class TestAdaptiveThresholdSelector(TestCase):
    def setUp(self) -> None:
        # Set the seed using keras.utils.set_random_seed. This will set:
        # 1) `numpy` seed
        # 2) backend random seed
        # 3) `python` random seed
        keras.utils.set_random_seed(42)
        self.threshold_selector = AdaptiveThresholdSelector(gamma=1., nn_size=10, nn_epochs=100, early_stopping=True,
                                                            patience=3, validation_split=0.25)

        # input
        train_data = np.array(np.arange(1, 100).reshape(33, 3) / 100)
        pred_data = np.array(np.arange(1, 100).reshape(33, 3) / 100)
        train_timestamps = pd.date_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 2), periods=len(train_data))
        pred_timestamps = pd.date_range(start=datetime(2020, 1, 2), end=datetime(2020, 1, 3), periods=len(pred_data))
        self.train_data = pd.DataFrame(data=train_data, index=train_timestamps)
        train_normal_index = np.array(33 * [True])
        self.train_normal_index = pd.Series(data=train_normal_index, index=train_timestamps)
        self.pred_data = pd.DataFrame(data=pred_data, index=pred_timestamps)
        pred_normal_index = np.array([False] * 4 + [True] * 25 + [False] * 4)
        self.pred_normal_index = pd.Series(data=pred_normal_index, index=pred_timestamps)
        self.pred_data[~self.pred_normal_index] += np.random.normal(loc=4, scale=2,
                                                                    size=self.pred_data[~self.pred_normal_index].shape)

        # rmse object needed for testing
        self.rmse = RMSEScore()
        self.rmse.fit(self.train_data)

    def test_fit(self) -> None:
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(self.train_data, scores, self.train_normal_index)
        self.assertIsNotNone(self.threshold_selector.nn_model)
        self.fitted_adaptive_threshold_selector = self.threshold_selector

    def test_predict(self) -> None:
        # expected output
        exp_anomalies = [True] * 4 + [False] * 25 + [True] * 4

        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scaled_ae_input=self.train_data,
                                    anomaly_score=scores,
                                    normal_index=self.train_normal_index
                                    )
        scores = self.rmse.transform(self.pred_data)
        anomalies, threshold = self.threshold_selector.predict(scaled_ae_input=self.pred_data, x=scores)

        assert_array_equal(anomalies, exp_anomalies)

    def test_smoothing(self):
        anomaly_scores = self.rmse.transform(self.pred_data)
        index, mean_score = self.threshold_selector._smooth_anomaly_score(anomaly_score=anomaly_scores)
        expected_index = pd.to_datetime(['2020-01-02 00:00:00', '2020-01-02 00:45:00',
                                         '2020-01-02 01:30:00', '2020-01-02 02:15:00',
                                         '2020-01-02 03:00:00', '2020-01-02 03:45:00',
                                         '2020-01-02 04:30:00', '2020-01-02 05:15:00',
                                         '2020-01-02 06:00:00', '2020-01-02 06:45:00',
                                         '2020-01-02 07:30:00', '2020-01-02 08:15:00',
                                         '2020-01-02 09:00:00', '2020-01-02 09:45:00',
                                         '2020-01-02 10:30:00', '2020-01-02 11:15:00',
                                         '2020-01-02 12:00:00', '2020-01-02 12:45:00',
                                         '2020-01-02 13:30:00', '2020-01-02 14:15:00',
                                         '2020-01-02 15:00:00', '2020-01-02 15:45:00',
                                         '2020-01-02 16:30:00', '2020-01-02 17:15:00',
                                         '2020-01-02 18:00:00', '2020-01-02 18:45:00',
                                         '2020-01-02 19:30:00', '2020-01-02 20:15:00',
                                         '2020-01-02 21:00:00', '2020-01-02 21:45:00',
                                         '2020-01-02 22:30:00', '2020-01-02 23:15:00',
                                         '2020-01-03 00:00:00'])
        expected_mean_score = pd.Series(data=np.array([14.86307, 15.97868, 17.91221, 12.197, 1.26025, 1.15523, 1.05021,
                                                       0.94519, 0.84017, 0.73515, 0.63013, 0.52511, 0.42008, 0.31506,
                                                       0.21004, 0.10502, 0.0, 0.10502, 0.21004, 0.31506, 0.42008,
                                                       0.52511, 0.63013, 0.73515, 0.84017, 0.94519, 1.05021, 1.15523,
                                                       1.26025, 10.09686, 13.11089, 16.17702, 12.81238]))
        pd.testing.assert_index_equal(left=index, right=expected_index)
        pd.testing.assert_series_equal(left=mean_score, right=expected_mean_score, atol=1e-4, check_index=False)


class TestAdaptiveThresholdSelectorMultiIndex(TestCase):
    """Tests for AdaptiveThresholdSelector with MultiIndex (device_id, timestamp) data."""

    def setUp(self) -> None:
        keras.utils.set_random_seed(42)
        self.threshold_selector = AdaptiveThresholdSelector(
            gamma=1., nn_size=10, nn_epochs=100, early_stopping=True,
            patience=3, validation_split=0.25, smoothing_parameter=1
        )
        self.threshold_selector_smoothed = AdaptiveThresholdSelector(
            gamma=1., nn_size=10, nn_epochs=100, early_stopping=True,
            patience=3, validation_split=0.25, smoothing_parameter=3
        )

        n_per_device = 33
        n_devices = 2
        devices = ['device_A', 'device_B']
        timestamps = pd.date_range(start=datetime(2020, 1, 1), periods=n_per_device, freq='1h')

        # Build MultiIndex
        multi_idx = pd.MultiIndex.from_product([devices, timestamps], names=['device_id', 'timestamp'])

        # Training data: two devices with slightly different baselines
        train_vals_a = np.arange(1, n_per_device * 3 + 1).reshape(n_per_device, 3) / 100
        train_vals_b = np.arange(1, n_per_device * 3 + 1).reshape(n_per_device, 3) / 100 + 0.5
        train_data = np.vstack([train_vals_a, train_vals_b])
        self.train_data = pd.DataFrame(data=train_data, index=multi_idx, columns=['f1', 'f2', 'f3'])

        # Normal index: all normal during training
        self.train_normal_index = pd.Series(
            data=np.ones(n_per_device * n_devices, dtype=bool), index=multi_idx
        )

        # Prediction data: inject anomalies at start/end of each device
        pred_timestamps = pd.date_range(start=datetime(2020, 1, 3), periods=n_per_device, freq='1h')
        pred_multi_idx = pd.MultiIndex.from_product(
            [devices, pred_timestamps], names=['device_id', 'timestamp']
        )
        pred_vals_a = np.arange(1, n_per_device * 3 + 1).reshape(n_per_device, 3) / 100
        pred_vals_b = np.arange(1, n_per_device * 3 + 1).reshape(n_per_device, 3) / 100 + 0.5
        pred_data = np.vstack([pred_vals_a, pred_vals_b])
        self.pred_data = pd.DataFrame(data=pred_data, index=pred_multi_idx, columns=['f1', 'f2', 'f3'])

        # Inject anomalies: first 4 and last 4 per device
        anomaly_mask = np.array([False] * 4 + [True] * 25 + [False] * 4)
        self.pred_normal_index = pd.Series(
            data=np.tile(anomaly_mask, n_devices), index=pred_multi_idx
        )
        np.random.seed(42)
        self.pred_data[~self.pred_normal_index] += np.random.normal(
            loc=4, scale=2, size=self.pred_data[~self.pred_normal_index].shape
        )

        # RMSE scorer
        self.rmse = RMSEScore()
        self.rmse.fit(self.train_data)

    def test_fit_resolves_groupby_level(self) -> None:
        """Test that fit correctly detects the groupby level from MultiIndex."""
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(
            scaled_ae_input=self.train_data,
            anomaly_score=scores,
            normal_index=self.train_normal_index
        )
        # Should resolve to level 0 ('device_id') or 'device_id'
        self.assertIsNotNone(self.threshold_selector.groupby_level_)
        self.assertIsNotNone(self.threshold_selector.nn_model)

    def test_fit_with_no_grouping(self) -> None:
        """Test that groupby_level=None disables grouping even with MultiIndex."""
        selector = AdaptiveThresholdSelector(
            gamma=1., nn_size=10, nn_epochs=50, early_stopping=True,
            patience=3, validation_split=0.25, groupby_level=None
        )
        scores = self.rmse.transform(self.train_data)
        selector.fit(
            scaled_ae_input=self.train_data,
            anomaly_score=scores,
            normal_index=self.train_normal_index
        )
        self.assertIsNone(selector.groupby_level_)

    def test_predict_multiindex(self) -> None:
        """Test that predict works with MultiIndex data and returns correct shape."""
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(
            scaled_ae_input=self.train_data,
            anomaly_score=scores,
            normal_index=self.train_normal_index
        )
        pred_scores = self.rmse.transform(self.pred_data)
        anomalies, threshold = self.threshold_selector.predict(
            scaled_ae_input=self.pred_data, x=pred_scores
        )
        self.assertEqual(len(anomalies), len(self.pred_data))
        self.assertEqual(len(threshold), len(self.pred_data))

    def test_predict_detects_injected_anomalies(self) -> None:
        """Test that injected anomalies at device boundaries are detected."""
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(
            scaled_ae_input=self.train_data,
            anomaly_score=scores,
            normal_index=self.train_normal_index
        )
        pred_scores = self.rmse.transform(self.pred_data)
        anomalies, _ = self.threshold_selector.predict(
            scaled_ae_input=self.pred_data, x=pred_scores
        )
        # Check that at least some of the injected anomalies are detected
        anomalies_series = pd.Series(anomalies, index=self.pred_data.index)
        anomalous_mask = ~self.pred_normal_index
        # At least half of the truly anomalous points should be detected
        detected_among_anomalous = anomalies_series[anomalous_mask].sum()
        self.assertGreater(detected_among_anomalous, anomalous_mask.sum() * 0.5)

    def test_smoothing_multiindex_respects_groups(self) -> None:
        """Test that smoothing does not mix data across device groups."""
        # Fit to resolve groupby_level_
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector_smoothed.fit(
            scaled_ae_input=self.train_data,
            anomaly_score=scores,
            normal_index=self.train_normal_index
        )

        pred_scores = self.rmse.transform(self.pred_data)
        index, mean_score = self.threshold_selector_smoothed._smooth_anomaly_score(pred_scores)

        # With smoothing_parameter=3 and 33 samples per device, 2 devices:
        # Each device produces ceil(33/3) = 11 segments
        # Total segments should be 11 * 2 = 22
        expected_segments = (33 // 3 + (1 if 33 % 3 else 0)) * 2  # 11 * 2 = 22
        self.assertEqual(len(mean_score), expected_segments)
        self.assertEqual(len(index), expected_segments)

    def test_smoothing_multiindex_no_cross_device_mixing(self) -> None:
        """Verify that smoothed scores per device match independently smoothed results."""
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector_smoothed.fit(
            scaled_ae_input=self.train_data,
            anomaly_score=scores,
            normal_index=self.train_normal_index
        )

        pred_scores = self.rmse.transform(self.pred_data)
        _, mean_score_multi = self.threshold_selector_smoothed._smooth_anomaly_score(pred_scores)

        # Compute expected per-device smoothing independently
        sp = self.threshold_selector_smoothed.smoothing_parameter
        expected_parts = []
        for device in ['device_A', 'device_B']:
            device_scores = pred_scores.loc[device]
            grouped_mean = device_scores.groupby(
                np.arange(len(device_scores)) // sp
            ).mean()
            expected_parts.append(grouped_mean)

        expected_combined = pd.concat(expected_parts, ignore_index=True)
        pd.testing.assert_series_equal(
            mean_score_multi.reset_index(drop=True),
            expected_combined.reset_index(drop=True),
            check_names=False
        )

    def test_smoothing_single_index_fallback(self) -> None:
        """Test that smoothing still works for single DatetimeIndex (no grouping)."""
        # Use single-device data
        timestamps = pd.date_range(start=datetime(2020, 1, 1), periods=33, freq='1h')
        single_data = pd.DataFrame(
            data=np.arange(1, 100).reshape(33, 3) / 100,
            index=timestamps, columns=['f1', 'f2', 'f3']
        )
        normal_idx = pd.Series(True, index=timestamps)

        rmse = RMSEScore()
        rmse.fit(single_data)
        scores = rmse.transform(single_data)

        selector = AdaptiveThresholdSelector(
            gamma=1., nn_size=10, nn_epochs=50, smoothing_parameter=3,
            early_stopping=True, patience=3, validation_split=0.25
        )
        selector.fit(scaled_ae_input=single_data, anomaly_score=scores, normal_index=normal_idx)

        # groupby_level_ should be None for single DatetimeIndex
        self.assertIsNone(selector.groupby_level_)

        # Smoothing should still produce correct number of segments
        index, mean_score = selector._smooth_anomaly_score(scores)
        expected_segments = 33 // 3 + (1 if 33 % 3 else 0)  # 11
        self.assertEqual(len(mean_score), expected_segments)
        self.assertEqual(len(index), expected_segments)
