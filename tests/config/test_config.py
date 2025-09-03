
import os
import shutil
import unittest

import numpy as np
from energy_fault_detector.config import Config, InvalidConfigFile

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..')


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.save_dir = 'save_conf'

    def tearDown(self) -> None:
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_init(self):
        conf = Config(os.path.join(PROJECT_ROOT, './tests/test_data/test_config.yaml'))
        self.assertDictEqual(conf.config_dict['train'], {
            'anomaly_score': {'name': 'mahalanobis',
                              'params': {'pca': True, 'pca_min_var': 0.85}},
            'data_preprocessor': {'params': {'max_nan_frac_per_col': 0.05,
                                             'imputer_strategy': 'mean',
                                             'features_to_exclude': ['feature1', 'feature2'],
                                             'include_duplicate_value_to_nan': False}},
            'autoencoder': {'name': 'MultilayerAutoencoder',
                            'verbose': 0,
                            'params': {'layers': [300],
                                       'code_size': 50,
                                       'learning_rate': 0.001,
                                       'decay_rate': 0.001,
                                       'decay_steps': 100,
                                       'batch_size': 144,
                                       'epochs': 10,
                                       'loss_name': 'mean_squared_error'}},
            'threshold_selector': {'name': 'FDR',
                                   'params': {'target_false_discovery_rate': 0.8},
                                   'fit_on_val': False},
            'data_splitter': {'train_block_size': 7, 'val_block_size': 3, 'type': 'DataSplitter'},
            'data_clipping': {'lower_percentile': 0.01, 'upper_percentile': 0.99}
        })
        self.assertDictEqual(conf.config_dict['root_cause_analysis'],
                             {'alpha': 0.8,
                              'init_x_bias': 'recon',
                              'num_iter': 200}
                             )

    def test_arcana_config(self):
        conf = Config(os.path.join(PROJECT_ROOT, './tests/test_data/test_config.yaml'))
        self.assertTrue(conf.root_cause_analysis)
        self.assertDictEqual(
            conf.arcana_params,
            {'alpha': 0.8, 'num_iter': 200, 'init_x_bias': 'recon'}
        )

        conf = Config(os.path.join(PROJECT_ROOT, './tests/test_data/test_config_no_rca.yaml'))
        self.assertFalse(conf.root_cause_analysis)

    def test_criticality_config(self):
        conf = Config(os.path.join(PROJECT_ROOT, './tests/test_data/test_criticality_config.yaml'))
        self.assertDictEqual(conf.config_dict['predict'], {
            'criticality': {'max_criticality': 144}
        })

    def test_early_stopping_val_split_config(self):
        conf = Config(os.path.join(PROJECT_ROOT, './tests/test_data/test_early_stopping_val_split_config.yaml'))
        self.assertDictEqual(conf.config_dict['train']['autoencoder']['params'], {
            'layers': [300],
            'code_size': 50,
            'learning_rate': 0.001,
            'decay_rate': 0.001,
            'decay_steps': 100,
            'batch_size': 144,
            'early_stopping': True,
            'patience': 42,
            'min_delta': 0.001,
            'epochs': 100,
            'loss_name': 'mean_squared_error'
        })
        self.assertDictEqual(conf.config_dict['train']['data_splitter'], {
            'type': 'sklearn',
            'shuffle': True,
            'validation_split': 0.25
        })

    def test_early_stopping_val_block_config(self):
        conf = Config(os.path.join(PROJECT_ROOT, './tests/test_data/test_early_stopping_val_block_config.yaml'))
        self.assertDictEqual(conf.config_dict['train']['autoencoder']['params'], {
            'layers': [300],
            'code_size': 50,
            'learning_rate': 0.001,
            'decay_rate': 0.001,
            'decay_steps': 100,
            'batch_size': 144,
            'early_stopping': True,
            'patience': 42,
            'min_delta': 0.0001,
            'epochs': 100,
            'loss_name': 'mean_squared_error'
        })
        self.assertDictEqual(conf.config_dict['train']['data_splitter'], {
            'type': 'DataSplitter',
            'train_block_size': 7,
            'val_block_size': 3
        })

    def test_bad_early_stopping_no_valconfig(self):
        with self.assertRaises(InvalidConfigFile):
            Config(os.path.join(PROJECT_ROOT, './tests/test_data/test_bad_early_stopping_config.yaml'))

    def test_verbose_config(self):
        conf = Config(os.path.join(PROJECT_ROOT, './tests/test_data/verbose_config.yaml'))
        self.assertEqual(conf.verbose, 0)
