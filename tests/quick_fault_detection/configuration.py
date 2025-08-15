import unittest
import numpy as np
import pandas as pd

from energy_fault_detector.quick_fault_detection.configuration import select_config


class TestQuickFaultDetectionConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)

    def test_select_config_low_feature_number(self):
        """ This test also tests the config update functions update_preprocessor_config, update_autoencoder_config and
        update_threshold_config.
        """
        train_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 8, 6], 'exclude_feature': [1, 1, 1],
                                   'angle': [0, 1, 0]})
        expected_code_size = 2
        expected_layers = [20, 10, 5]  # default value for a dataset with this low dimensions
        normal_index = pd.Series([True, False, True])
        config = select_config(train_data, normal_index,
                               status_label_confidence_percentage=0.99,
                               features_to_exclude=['exclude_feature'],
                               angles=['angle'],
                               automatic_optimization=False)

        self.assertListEqual(config['train']['data_preprocessor']['params']['angles'], ['angle'])
        self.assertListEqual(config['train']['data_preprocessor']['params']['features_to_exclude'], ['exclude_feature'])
        self.assertEqual(config['train']['autoencoder']['params']['code_size'], expected_code_size)
        self.assertListEqual(config['train']['autoencoder']['params']['layers'], expected_layers)
        self.assertEqual(config['train']['threshold_selector']['params']['quantile'], 0.99)

    def test_select_config_high_feature_number(self):
        num_features = 100
        train_data = pd.DataFrame(data=np.random.random(size=(3, num_features)),
                                  columns=[str(x) for x in range(num_features)])
        expected_code_size = 2
        expected_layers = [num_features, int(num_features/2), int(num_features/4)]
        normal_index = pd.Series([True, False, True])
        config = select_config(train_data, normal_index,
                               status_label_confidence_percentage=0.99,
                               features_to_exclude=None,
                               angles=None,
                               automatic_optimization=False)

        self.assertEqual(config['train']['autoencoder']['params']['code_size'], expected_code_size)
        self.assertListEqual(config['train']['autoencoder']['params']['layers'], expected_layers)
        self.assertEqual(config['train']['threshold_selector']['params']['quantile'], 0.99)
