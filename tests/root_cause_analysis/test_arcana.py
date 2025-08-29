import os
from unittest import TestCase
import pickle as pkl

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
import tensorflow as tf

from energy_fault_detector.root_cause_analysis.arcana import Arcana
from energy_fault_detector.autoencoders import MultilayerAutoencoder, ConditionalAE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestArcana(TestCase):

    def setUp(self) -> None:
        self.n = 1000
        data = np.array([[
            x,
            (x + 1),
            x ** 2,
            x * 3,
            x * 2 + 5,
            np.exp(x / self.n)
        ] for x in range(self.n)])
        self.data = (data - data.mean(axis=0)) / data.std(axis=0)
        time_index = pd.date_range(start="01-01-2022", periods=self.n, freq='10min')
        self.data_frame = pd.DataFrame(index=time_index, data=self.data)  # find_arcana_bias expects a pandas Dataframe

        self.ml_ae = MultilayerAutoencoder()
        input_dim = self.data.shape[1]
        ml_input_layer = Input(shape=(input_dim,))
        encoded = Dense(10, input_shape=(input_dim,), activation="linear")(ml_input_layer)
        decoded = Dense(input_dim, activation="linear")(encoded)
        ml_model = Model(inputs=ml_input_layer, outputs=decoded)
        ml_model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_absolute_error'])
        ml_model.load_weights(os.path.join(PROJECT_ROOT, 'test_data/ml_model_weights.h5'))
        self.ml_ae.model = ml_model
        self.ml_ae.history = 0  # ml_ae._is_fitted() will now return True so predict can be used without fit

        self.cond_ae = ConditionalAE(conditional_features=['a'])
        cond_data = self.data_frame.copy()
        cond_data['a'] = [1]*500 + [2]*500
        self.cond_data = cond_data
        self.cond_ae.fit(self.cond_data, verbose=0)

    def test_find_arcana_bias(self):
        with open(os.path.join(PROJECT_ROOT, 'test_data/arcana_bias.pkl'), 'rb') as file:
            expected_bias = pkl.load(file)

        arcana = Arcana(model=self.ml_ae, num_iter=42)
        bias, _, _ = arcana.find_arcana_bias(self.data_frame)
        assert_array_almost_equal(expected_bias, bias.values, decimal=3)
        self.assertIsInstance(bias, pd.DataFrame)

    def test_find_arcana_bias_with_history(self):
        arcana = Arcana(model=self.ml_ae, num_iter=51)  # at least 50 iterations are needed
        bias, losses, bias_history = arcana.find_arcana_bias(self.data_frame, track_losses=True, track_bias=True)
        self.assertIsInstance(bias_history, list)
        self.assertTrue(len(bias_history) == 3)  # init bias + bias of 1st iteration and bias of 50th iteration
        self.assertIsInstance(losses, pd.DataFrame)
        for loss in losses:
            self.assertTrue(len(losses[loss]) == 2)  # losses of 1st iteration and 50th iteration

    def test_decreasing_loss(self):

        for alpha in [0, 0.99]:
            arcana = Arcana(model=self.ml_ae, num_iter=1, alpha=alpha, init_x_bias='recon')
            bias = arcana.initialize_x_bias(self.data.astype('float32'))
            last_loss = 1e8
            x = tf.Variable(self.data, dtype=tf.float32)
            bias = tf.Variable(bias, dtype=tf.float32)
            for _ in range(5):
                bias, losses, _ = arcana.update_x_bias(x=x, x_bias=bias)
                self.assertLess(losses[0].numpy(), last_loss)
                last_loss = losses[0].numpy()

    def test_init_bias(self):
        arcana = Arcana(model=self.ml_ae, num_iter=42, init_x_bias='weightedB', alpha=0.6)
        bias_expected = 0.6 * (self.ml_ae.predict(self.data) - self.data)
        bias = arcana.initialize_x_bias(self.data)
        assert_array_almost_equal(bias_expected, bias)

        arcana = Arcana(model=self.ml_ae, num_iter=42, init_x_bias='weightedA', alpha=0.6)
        bias_expected = 0.4 * (self.ml_ae.predict(self.data) - self.data)
        bias = arcana.initialize_x_bias(self.data)
        assert_array_almost_equal(bias_expected, bias)

        arcana = Arcana(model=self.ml_ae, num_iter=42, init_x_bias='recon', alpha=0.6)
        bias_expected = self.ml_ae.predict(self.data) - self.data
        bias = arcana.initialize_x_bias(self.data)
        assert_array_almost_equal(bias_expected, bias)

        arcana = Arcana(model=self.ml_ae, num_iter=42, init_x_bias='zero', alpha=0.6)
        bias_expected = 0 * self.data
        bias = arcana.initialize_x_bias(self.data)
        assert_array_almost_equal(bias_expected, bias)

    def test_draw_samples(self):
        arcana = Arcana(model=self.ml_ae, num_iter=42, max_sample_threshold=self.n)
        selection = arcana.draw_samples(x=self.data)
        self.assertTupleEqual(self.data.shape, self.data[selection].shape)

        arcana.max_sample_threshold = self.n - 1
        selection = arcana.draw_samples(x=self.data)
        self.assertTupleEqual(self.data[:-1].shape, self.data[selection].shape)

    def test_conditional_ae(self):
        arcana = Arcana(model=self.cond_ae, num_iter=10, init_x_bias='recon', max_sample_threshold=100)
        inputs = self.cond_data.drop(['a'], axis=1)
        conditions = self.cond_data[['a']]

        # Test initialization
        bias_init = arcana.initialize_x_bias(inputs.values, conditions.values)
        self.assertTupleEqual(inputs.shape, bias_init.numpy().shape)

        # Test one update
        inputs = tf.Variable(inputs.values, dtype=tf.float32)
        bias = tf.Variable(bias_init, dtype=tf.float32)
        conditions = tf.constant(conditions, dtype=tf.float32)
        bias, _, _ = arcana.update_x_bias(x=inputs, x_bias=bias, conditions=conditions)
        self.assertTupleEqual(inputs.numpy().shape, bias.numpy().shape)

        bias, _, _ = arcana.update_x_bias(x=inputs, x_bias=bias, conditions=conditions)
        bias, _, _ = arcana.update_x_bias(x=inputs, x_bias=bias, conditions=conditions)

        # Test 10 iterations
        arcana = Arcana(model=self.cond_ae, num_iter=10, init_x_bias='recon')
        x_bias, tracked_losses, tracked_bias_dfs = arcana.find_arcana_bias(self.cond_data, track_bias=True)
        self.assertTupleEqual(inputs.numpy().shape, x_bias.shape)
        self.assertTupleEqual((0, 3), tracked_losses.shape)
        self.assertEqual(len(tracked_bias_dfs), 2)  # init bias + first iteration
