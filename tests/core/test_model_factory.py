
import os
import unittest

from energy_fault_detector.core.model_factory import ModelFactory
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
