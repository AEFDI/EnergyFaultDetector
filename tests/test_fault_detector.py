
import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import logging

import pandas as pd
import numpy as np

from energy_fault_detector.fault_detector import FaultDetector, Config
from energy_fault_detector.autoencoders import ConditionalAE

mock_autoencoder = MagicMock()
mock_data_preprocessor = MagicMock()
mock_threshold = MagicMock()
mock_score = MagicMock()

# silence logging
logger = logging.getLogger('energy_fault_detector')
logger.setLevel(logging.CRITICAL)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestFaultDetectorSaveLoad(unittest.TestCase):
    """Test saving and loading of the FaultDetector models."""

    def setUp(self) -> None:
        self.config_path = os.path.join(PROJECT_ROOT, 'tests/test_data/test_config.yaml')
        self.conf = Config(self.config_path)
        self.test_dir = tempfile.mkdtemp()

        # Create dummy sensor data for training
        np.random.seed(42)
        self.sensor_data = pd.DataFrame(data=np.random.random(size=(100, 3)), columns=['a', 'b', 'c'])
        self.normal_index = pd.Series(np.random.choice([True, False], size=100))

        self.fault_detector = FaultDetector(config=self.conf, model_directory=self.test_dir)

    def tearDown(self) -> None:
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_load_models(self):
        # Train the model
        results = self.fault_detector.fit(sensor_data=self.sensor_data, normal_index=self.normal_index)
        model_path = results.model_path

        # Create a new FaultDetector instance and load the saved models
        loaded_fault_detector = FaultDetector(config=self.conf, model_directory=self.test_dir)
        loaded_fault_detector.load_models(model_path=model_path)

        # Compare the attributes of the original and loaded models
        self.assertIsNotNone(self.fault_detector.autoencoder)
        self.assertIsNotNone(loaded_fault_detector.autoencoder)

        # Check if the parameters of the autoencoders are the same
        original_weights = self.fault_detector.autoencoder.model.get_weights()
        loaded_weights = loaded_fault_detector.autoencoder.model.get_weights()

        for original_weight, loaded_weight in zip(original_weights, loaded_weights):
            np.testing.assert_array_almost_equal(original_weight, loaded_weight)

        self.assertIsNotNone(self.fault_detector.threshold_selector)
        self.assertIsNotNone(loaded_fault_detector.threshold_selector)

        # Check if the parameters of the threshold selectors are the same
        original_threshold_params = self.fault_detector.threshold_selector.get_params()
        loaded_threshold_params = loaded_fault_detector.threshold_selector.get_params()
        self.assertEqual(original_threshold_params, loaded_threshold_params)

        # Check the configuration
        self.assertDictEqual(self.fault_detector.config.config_dict,
                             loaded_fault_detector.config.config_dict)


@patch("energy_fault_detector.core.model_factory.Autoencoder", new=mock_autoencoder)
@patch("energy_fault_detector.core.model_factory.AnomalyScore", new=mock_score)
@patch("energy_fault_detector.core.model_factory.ThresholdSelector", new=mock_threshold)
@patch("energy_fault_detector.core.model_factory.DataPreprocessor", new=mock_data_preprocessor)
class TestFaultDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.conf = Config(os.path.join(PROJECT_ROOT, './tests/test_data/test_config.yaml'))
        self.conf.read_config()

        self.sensor_data = pd.DataFrame(data=[[1., 2., 3.],
                                              [4., 5., 6.],
                                              [7., 8., 9.]], columns=['a', 'b', 'c'])
        self.normal_index = pd.Series(data=[True, True, False])
        self.predictions = pd.DataFrame(data=[[1.1, 2., 3.],
                                              [4., 4.9, 6.],
                                              [7., 8., 9.1]], columns=['a', 'b', 'c'])
        self.recon_error = self.predictions - self.sensor_data

        self.test_model_dir = tempfile.mkdtemp('models')

        mock_autoencoder.reset_mock(return_value=True, side_effect=True)
        mock_data_preprocessor.reset_mock(return_value=True, side_effect=True)
        mock_score.reset_mock(return_value=True, side_effect=True)
        mock_threshold.reset_mock(return_value=True, side_effect=True)

    def tearDown(self) -> None:
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_model_dir)

    def _create_fault_detector(self, config: Config) -> FaultDetector:
        fault_detector = FaultDetector(config=config, model_directory=self.test_model_dir)
        fault_detector._model_factory = MagicMock()
        fault_detector.autoencoder = mock_autoencoder
        fault_detector.data_preprocessor = mock_data_preprocessor
        fault_detector.threshold_selector = mock_threshold
        fault_detector.anomaly_score = mock_score
        return fault_detector

    def test_init(self):
        fault_detector = self._create_fault_detector(self.conf)
        self.assertEqual(fault_detector.model_directory, str(self.test_model_dir))
        self.assertEqual(fault_detector.config, self.conf)

    def test_missing_config(self):
        with self.assertLogs('energy_fault_detector', level='INFO') as cm:
            ad = FaultDetector(model_directory=self.test_model_dir)
            self.assertEqual(cm.output,
                             [
                                 'INFO:energy_fault_detector:No configuration set - load models and config from path with the `load_models` method.'])

    def test_save_models(self):
        self.conf.write_config = MagicMock()
        fault_detector = self._create_fault_detector(self.conf)

        asset_id = 1
        path, dt = fault_detector.save_models(model_name=asset_id)

        model_objects = [mock_score, mock_data_preprocessor, mock_autoencoder, mock_threshold]
        names = ['anomaly_score', 'data_preprocessor', 'autoencoder', 'threshold_selector']
        for model_object, name in zip(model_objects, names):
            model_object.save.assert_called_once()
            self.assertEqual(model_object.save.call_args[0][0],
                             os.path.join(fault_detector.model_directory, dt, str(asset_id), name))

        fault_detector.config.write_config.assert_called_once()
        self.assertEqual(fault_detector.config.write_config.call_args[0][0],
                         os.path.join(fault_detector.model_directory, dt, str(asset_id), 'config.yaml'))

    def test_load_models(self):
        fault_detector = self._create_fault_detector(self.conf)
        fault_detector._load_pickled_model = MagicMock()
        fault_detector._load_pickled_model.side_effect = [mock_data_preprocessor,
                                                          mock_autoencoder,
                                                          mock_threshold,
                                                          mock_score]

        fault_detector.load_models(model_path='path_to_saved_models')
        names = ['data_preprocessor', 'autoencoder', 'threshold_selector', 'anomaly_score']
        for call_args, name in zip(fault_detector._load_pickled_model.call_args_list, names):
            self.assertEqual(call_args[1]['model_type'], name)
            self.assertEqual(call_args[1]['model_directory'], os.path.join('path_to_saved_models', name))

    def test_train(self):
        self.conf.write_config = MagicMock()
        fault_detector = self._create_fault_detector(self.conf)

        mock_data_preprocessor.transform.side_effect = [self.sensor_data[self.normal_index],
                                                        self.sensor_data]
        mock_autoencoder.get_reconstruction_error.side_effect = [self.recon_error, self.recon_error, self.recon_error]
        mock_score.transform.side_effect = [pd.Series([0.1, 0.2, 0.15])] * 2

        results = fault_detector.fit(sensor_data=self.sensor_data,
                                     normal_index=self.normal_index)

        mock_data_preprocessor.fit.assert_called_once()
        self.assertEqual(mock_data_preprocessor.transform.call_count, 2)

        mock_autoencoder.fit.assert_called_once()
        self.assertEqual(mock_autoencoder.get_reconstruction_error.call_count, 2)

        mock_score.fit.assert_called_once()
        mock_score.transform.assert_called_once()
        mock_threshold.fit.assert_called_once()
        # saved models:
        mock_score.save.assert_called_once()
        self.conf.write_config.assert_called_once()

        model_dir = os.path.join(fault_detector.model_directory, fault_detector.save_timestamps[0], 'trained_model')
        model_date = results.model_date
        self.assertIsInstance(model_date, str)
        self.assertEqual(results.model_date, fault_detector.save_timestamps[0])
        self.assertEqual(results.model_path, os.path.abspath(model_dir))

        # without saving models
        mock_data_preprocessor.transform.side_effect = [self.sensor_data[self.normal_index],
                                                        self.sensor_data]
        mock_autoencoder.get_reconstruction_error.side_effect = [self.recon_error, self.recon_error, self.recon_error]
        mock_score.transform.return_value = pd.Series([0.1, 0.2, 0.15])

        _ = fault_detector.fit(sensor_data=self.sensor_data,
                               normal_index=self.normal_index,
                               save_models=False)

        mock_score.save.asset_not_called()
        self.assertEqual(self.conf.write_config.call_count, 1)

    def test_tune(self):
        mock_data_preprocessor.transform.side_effect = [self.sensor_data[self.normal_index],
                                                        self.sensor_data]
        mock_autoencoder.get_reconstruction_error.side_effect = [self.recon_error] * 9
        mock_score.transform.side_effect = [pd.Series([0.1, 0.2, 0.15])] * 3
        mock_data_preprocessor.transform.side_effect = [self.sensor_data[self.normal_index],
                                                        self.sensor_data]
        fault_detector = self._create_fault_detector(self.conf)
        tune_results = fault_detector.tune(sensor_data=self.sensor_data, normal_index=self.normal_index,
                                           new_learning_rate=0.001, tune_epochs=1, tune_method='full',
                                           save_models=False)
        mock_autoencoder.tune.called_once()

        mock_data_preprocessor.transform.side_effect = [self.sensor_data[self.normal_index],
                                                        self.sensor_data]
        fault_detector = self._create_fault_detector(self.conf)
        tune_results = fault_detector.tune(sensor_data=self.sensor_data, normal_index=self.normal_index,
                                           new_learning_rate=0.001, tune_epochs=1, tune_method='decoder',
                                           save_models=False)
        mock_autoencoder.tune_decoder.called_once()

    @patch('energy_fault_detector.fault_detector.Arcana.find_arcana_bias')
    def test_predict(self, mock_find_arcana_bias):
        fault_detector = self._create_fault_detector(self.conf)
        fault_detector.load_models = MagicMock()  # ensures we use the mock model objects

        # set up test
        mock_data_preprocessor.transform.return_value = self.sensor_data
        mock_autoencoder.predict.return_value = self.predictions
        mock_autoencoder.get_reconstruction_error.side_effect = [self.recon_error] * 9
        mock_data_preprocessor.inverse_transform.return_value = self.predictions
        mock_score.transform.side_effect = [pd.Series([0.1, 0.2, 0.15], index=self.sensor_data.index)] * 2
        mock_threshold.predict.return_value = np.array([False, False, True])
        mock_find_arcana_bias.return_value = self.predictions

        results = fault_detector.predict(model_path='path_to_saved_models',
                                         sensor_data=self.sensor_data,
                                         root_cause_analysis=True)

        # expected results
        recon = self.predictions
        recon.index = [0, 1, 2]
        anomalies = pd.DataFrame(data=[[False], [False], [True]],
                                 columns=['anomaly'])

        pd.testing.assert_frame_equal(results.reconstruction, recon)
        pd.testing.assert_frame_equal(results.predicted_anomalies, anomalies)
        self.assertIsNotNone(results.bias_data)

        mock_find_arcana_bias.assert_called_once()
        mock_find_arcana_bias.assert_called_with(x=self.sensor_data, track_losses=False, track_bias=False)

    def test__fit_threshold(self):
        fault_detector = self._create_fault_detector(self.conf)
        fault_detector.load_models = MagicMock()  # ensures we use the mock model objects

        # set up test
        mock_data_preprocessor.transform.return_value = self.sensor_data
        mock_autoencoder.get_reconstruction_error.side_effect = [
            self.recon_error,  # all
            self.recon_error.iloc[-1:],  # validation data
        ]
        scores = pd.Series([0.1, 0.2, 0.15], index=self.sensor_data.index)
        mock_score.transform.side_effect = [
            scores,  # all
            scores.iloc[-1:],  # validation data
        ]
        fault_detector.train_val_split = MagicMock()
        fault_detector.train_val_split.return_value = (self.sensor_data.iloc[:2], self.sensor_data.iloc[-1:])

        # fit on validation data only
        fault_detector._fit_threshold(self.sensor_data, self.normal_index, None, True)
        pd.testing.assert_series_equal(
            fault_detector.threshold_selector.fit.call_args.kwargs['x'],
            scores.iloc[-1:]
        )
        pd.testing.assert_series_equal(
            fault_detector.threshold_selector.fit.call_args.kwargs['y'],
            self.normal_index.iloc[-1:]
        )


class TestFaultDetectorModelCreation(unittest.TestCase):

    def setUp(self):
        self.test_model_dir = 'fd_test_models'

    def test_models_created(self):
        config = Config(os.path.join(PROJECT_ROOT, './tests/test_data/test_conditional_ae_config.yaml'))
        model = FaultDetector(config, model_directory=self.test_model_dir)
        self.assertIsInstance(model.autoencoder, ConditionalAE)
        self.assertListEqual(model.autoencoder.conditional_features, ['feature_a', 'feature_b'])
