
import shutil
from typing import Dict
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from energy_fault_detector.autoencoders.multilayer_autoencoder import MultilayerAutoencoder


class TestMultilayerAutoencoder(TestCase):
    def setUp(self) -> None:
        params: Dict = {
            'layers': [20],
            'code_size': 5,
            'learning_rate': 0.001,
            'decay_rate': 0.9,
            'decay_steps': 1000,
            'batch_size': 144,
            'epochs': 33,
            'loss_name': 'mean_squared_error',
            'verbose': 0
        }
        self.autoencoder = MultilayerAutoencoder(**params)
        np.random.seed(42)
        self.train_data = pd.DataFrame(
            np.random.random(size=(1000, 10))
        )
        self.fitted_autoencoder = MultilayerAutoencoder(**params).fit(self.train_data)
        self.test_data = pd.DataFrame(
            np.random.random(size=(10, 10))
        )
        self.saved_models = './test_models'

    def tearDown(self) -> None:
        """Remove saved files"""
        shutil.rmtree(self.saved_models, ignore_errors=True)

    def test_init(self) -> None:
        self.assertEqual(self.autoencoder.layers, [20])
        self.assertEqual(self.autoencoder.batch_size, 144)
        self.assertEqual(self.autoencoder.loss_name, 'mean_squared_error')

    def test_call(self) -> None:
        self.autoencoder.fit(self.train_data)
        output = self.autoencoder(self.test_data.values)
        output_predict = self.autoencoder.predict(self.test_data)
        assert_array_almost_equal(output.numpy(), output_predict)

    def test_fit(self) -> None:
        self.autoencoder.fit(self.train_data)
        self.assertEqual(len(self.autoencoder.model.layers), 8)  # input, 20, prelu, 5, prelu, 20, prelu, output
        self.assertIsNotNone(self.autoencoder.model)
        self.assertEqual(len(self.autoencoder.history['loss']), 33)

    def test_fit_with_val(self) -> None:
        self.autoencoder.fit(self.train_data, x_val=self.test_data)
        self.assertIsNotNone(self.autoencoder.model)
        self.assertEqual(len(self.autoencoder.history['loss']), 33)

    def test_encode(self) -> None:
        self.autoencoder.fit(self.train_data, x_val=self.test_data)

        encoded = self.autoencoder.encode(self.test_data)
        # note: weak test
        self.assertEqual(encoded.shape, (10, 5))

    def test_predict(self) -> None:
        self.autoencoder.fit(self.train_data)
        output = self.autoencoder.predict(self.test_data)
        self.assertEqual(self.test_data.shape, output.shape)

    def test_predict_not_fitted(self) -> None:
        with self.assertRaises(ValueError):
            self.autoencoder.predict(self.test_data)

    def test_tune(self) -> None:
        self.autoencoder.fit(self.train_data)
        self.autoencoder.tune(self.train_data, tune_epochs=5, learning_rate=0.001)
        self.assertEqual(len(self.autoencoder.history['loss']), 5 + 33)

    def test_tune_decoder(self) -> None:
        self.autoencoder.fit(self.train_data)
        encoder_weights = self.autoencoder.encoder.get_weights()
        self.autoencoder.tune_decoder(self.train_data, tune_epochs=5, learning_rate=0.001)
        encoder_weights_tuned = self.autoencoder.encoder.get_weights()
        self.assertEqual(len(self.autoencoder.history['loss']), 5 + 33)
        for w1, w2 in zip(encoder_weights, encoder_weights_tuned):
            assert_array_almost_equal(w1, w2)

    def test_save_and_load(self) -> None:

        self.autoencoder.save(self.saved_models)

        new_model = MultilayerAutoencoder()
        with self.assertWarns(UserWarning):
            new_model.load(self.saved_models)

        self.assertEqual(new_model.layers, self.autoencoder.layers)
        self.assertEqual(new_model.code_size, self.autoencoder.code_size)
        self.assertEqual(new_model.batch_size, self.autoencoder.batch_size)
        self.assertEqual(new_model.loss_name, self.autoencoder.loss_name)
        self.assertIsNone(new_model.model)
        self.assertIsNone(new_model.history)

        self.autoencoder.fit(self.train_data)
        self.autoencoder.save(self.saved_models, overwrite=True)

        new_model = MultilayerAutoencoder()
        new_model.load(self.saved_models)
        self.assertEqual(new_model.layers, self.autoencoder.layers)
        self.assertEqual(new_model.code_size, self.autoencoder.code_size)
        self.assertEqual(new_model.batch_size, self.autoencoder.batch_size)
        self.assertEqual(new_model.loss_name, self.autoencoder.loss_name)
        self.assertIsNotNone(new_model.model)
        self.assertDictEqual(new_model.history, self.autoencoder.history)

        for index, layer_weights in enumerate(self.autoencoder.model.weights):
            assert_array_almost_equal(layer_weights, new_model.model.weights[index])
