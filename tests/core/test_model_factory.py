
import os
import unittest

import numpy as np

from energy_fault_detector.core.model_factory import ModelFactory, _parse_ts_freq
from energy_fault_detector.config import Config
from energy_fault_detector.autoencoders import MultilayerAutoencoder
from energy_fault_detector.anomaly_scores import MahalanobisScore
from energy_fault_detector.threshold_selectors import FDRSelector
from energy_fault_detector.data_preprocessing import DataPreprocessor

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..')


class TestModelFactory(unittest.TestCase):

    def test_model_creation(self):
        config_path = os.path.join(PROJECT_ROOT, './tests/test_data/test_config.yaml')
        conf = Config(config_path)
        model_factory = ModelFactory(conf)

        # Test for autoencoder
        autoencoder = model_factory.autoencoder
        self.assertIsInstance(autoencoder, MultilayerAutoencoder)
        self.assertListEqual(autoencoder.layers, [300])

        # Test for data preprocessor
        data_preprocessor = model_factory.data_preprocessor
        self.assertIsInstance(data_preprocessor, DataPreprocessor)

        # Test for threshold selector
        threshold_selector = model_factory.threshold_selector
        self.assertIsInstance(threshold_selector, FDRSelector)
        self.assertEqual(threshold_selector.target_false_discovery_rate, 0.8)

        # Test for anomaly score
        anomaly_score = model_factory.anomaly_score
        self.assertIsInstance(anomaly_score, MahalanobisScore)

    def test_parse_ts_freq_to_timedelta64(self):
        """The model factory parses ts_freq strings into np.timedelta64 for the SequenceDatasetBuilder."""
        from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder

        self.assertEqual(_parse_ts_freq('30s'), np.timedelta64(30, 's'))
        self.assertEqual(_parse_ts_freq('10m'), np.timedelta64(10, 'm'))

        builder = SequenceDatasetBuilder(
            sequence_length=36, ts_freq=_parse_ts_freq('30s'), stride=1
        )
        self.assertIsInstance(builder.ts_freq, np.timedelta64)
        self.assertEqual(builder.ts_freq, np.timedelta64(30, 's'))
