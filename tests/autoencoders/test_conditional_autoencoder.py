"""Conditional AE tests"""

import os.path
import shutil
from typing import Dict
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from energy_fault_detector.autoencoders.conditional_autoencoder import ConditionalAE


class TestConditionalAutoencoder(TestCase):
    def setUp(self) -> None:
        params: Dict = {
            'layers': [20],
            'code_size': 5,
            'learning_rate': 0.001,
            'decay_rate': 0.9,
            'decay_steps': 1000,
            'batch_size': 144,
            'epochs': 33,
            'loss_name': 'mean_squared_error'
        }
        self.conditional_features = ['cond1', 'cond2']
        self.autoencoder = ConditionalAE(self.conditional_features, **params)
        np.random.seed(42)
        # Create training data with conditional features
        self.train_data = pd.DataFrame(
            np.random.random(size=(1000, 10)),
            columns=[f'feature_{i}' for i in range(10)]
        )
        for cond in self.conditional_features[::-1]:
            # conditional features
            self.train_data.insert(
                loc=0, column=cond, value=np.random.random(size=(1000,))
            )

        self.fitted_autoencoder = ConditionalAE(self.conditional_features, **params).fit(
            self.train_data, verbose=0
        )
        self.test_data = pd.DataFrame(
            np.random.random(size=(10, 10)),
            columns=[f'feature_{i}' for i in range(10)]
        )
        for cond in self.conditional_features[::-1]:
            self.test_data.insert(
                loc=0, column=cond, value=np.random.random(size=(10,))
            )

        self.test_inputs = self.test_data[[f'feature_{i}' for i in range(10)]]
        self.test_conditions = self.test_data[self.conditional_features]

        self.saved_models = './test_models'

    def tearDown(self) -> None:
        """Remove saved files"""
        shutil.rmtree(self.saved_models, ignore_errors=True)

    def test_init(self) -> None:
        # test init without arguments
        _ = ConditionalAE()

        self.assertEqual(self.autoencoder.layers, [20])
        self.assertEqual(self.autoencoder.batch_size, 144)
        self.assertEqual(self.autoencoder.loss_name, 'mean_squared_error')
        self.assertEqual(self.autoencoder.conditional_features, self.conditional_features)

    def test_call(self) -> None:
        output = self.fitted_autoencoder(self.test_inputs.values, self.test_conditions.values)
        output_predict = self.fitted_autoencoder.predict(self.test_data)

        assert_array_almost_equal(output.numpy(), output_predict)

    def test_fit(self) -> None:
        self.autoencoder.fit(self.train_data, verbose=0)
        self.assertEqual(len(self.autoencoder.model.layers), 11)  # Adjust based on your model structure
        self.assertIsNotNone(self.autoencoder.model)
        self.assertEqual(len(self.autoencoder.history['loss']), 33)

    def test_fit_with_val(self) -> None:
        self.autoencoder.fit(self.train_data, x_val=self.test_data, verbose=0)
        self.assertIsNotNone(self.autoencoder.model)
        self.assertEqual(len(self.autoencoder.history['loss']), 33)

    def test_encode(self) -> None:
        self.autoencoder.fit(self.train_data, x_val=self.test_data, verbose=0)

        encoded = self.autoencoder.encode(self.test_data)
        self.assertEqual(encoded.shape, (10, 5))

    def test_predict(self) -> None:
        self.autoencoder.fit(self.train_data, verbose=0)
        output = self.autoencoder.predict(self.test_data)
        self.assertEqual(self.test_data.shape[0], output.shape[0])  # Check number of rows
        self.assertEqual(self.test_inputs.shape[1], output.shape[1])  # Check number of columns

        output = self.autoencoder.predict(self.test_data, return_conditions=True)
        self.assertEqual(self.test_data.shape[0], output.shape[0])  # Check number of rows
        self.assertEqual(self.test_data.shape[1], output.shape[1])  # Check number of columns

    def test_recon_error(self):
        self.autoencoder.fit(self.train_data, verbose=0)
        recon_error = self.autoencoder.get_reconstruction_error(self.test_data)
        self.assertEqual(self.test_data.shape[0], recon_error.shape[0])
        self.assertEqual(self.test_inputs.shape[1], recon_error.shape[1])  # columns should be input data only

    def test_predict_not_fitted(self) -> None:
        with self.assertRaises(ValueError):
            self.autoencoder.predict(self.test_data)

    def test_tune(self) -> None:
        self.autoencoder.fit(self.train_data, verbose=0)
        self.autoencoder.tune(self.train_data, tune_epochs=5, learning_rate=0.001)
        self.assertEqual(len(self.autoencoder.history['loss']), 5 + 33)

    def test_save_and_load(self) -> None:
        self.autoencoder.save(self.saved_models)

        new_model = ConditionalAE(self.conditional_features)
        with self.assertWarns(UserWarning):
            new_model.load(self.saved_models)

        self.assertEqual(new_model.layers, self.autoencoder.layers)
        self.assertEqual(new_model.code_size, self.autoencoder.code_size)
        self.assertEqual(new_model.batch_size, self.autoencoder.batch_size)
        self.assertEqual(new_model.loss_name, self.autoencoder.loss_name)
        self.assertIsNone(new_model.model)
        self.assertIsNone(new_model.history)

        self.autoencoder.fit(self.train_data, verbose=0)
        self.autoencoder.save(self.saved_models, overwrite=True)

        new_model = ConditionalAE(self.conditional_features)
        new_model.load(self.saved_models)
        self.assertEqual(new_model.layers, self.autoencoder.layers)
        self.assertEqual(new_model.code_size, self.autoencoder.code_size)
        self.assertEqual(new_model.batch_size, self.autoencoder.batch_size)
        self.assertEqual(new_model.loss_name, self.autoencoder.loss_name)
        self.assertIsNotNone(new_model.model)
        self.assertDictEqual(new_model.history, self.autoencoder.history)

        for index, layer_weights in enumerate(self.autoencoder.model.weights):
            assert_array_almost_equal(layer_weights, new_model.model.weights[index])
