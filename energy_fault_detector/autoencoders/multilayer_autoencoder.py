"""Feedforward dense autoencoder."""

from typing import List, Union, Optional

import numpy as np
import pandas as pd

# pylint: disable=E0401,E0611
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Dense, PReLU, Input

from energy_fault_detector.core.autoencoder import Autoencoder

DataType = Union[np.ndarray, pd.DataFrame]


class MultilayerAutoencoder(Autoencoder):
    """Multilayer symmetric autoencoder with Dense layers.

    Args:
        layers: list of integers indicating the size (# units) of the layers in both the encoder and in
            the decoder (reversed order in this case). Default [200]
        code_size: number of units of the encoded layer (bottleneck layer).
            (number of features to compress the input features to). Default 10.
        kernel_initializer: initializer to use in each layer. Default he_normal.
        act: activation function to use, prelu, relu, ... Defaults to prelu.
        last_act: activation function for last layer, prelu, relu, sigmoid, linear... Defaults to linear.
        ae_kwargs: Training-related parameters (learning_rate, batch_size, epochs, loss_name, early_stopping, etc.)
            are accepted as keyword arguments and forwarded to Autoencoder.__init__.

    Attributes:
        model: keras Model object - the autoencoder network.
        encoder: keras Model object - encoder network of the autoencoder.
        history: dictionary with the losses and metrics for each epoch.

    Configuration example:

    .. code-block:: text

        train:
          autoencoder:
            name: MultilayerAutoencoder
            params:
              layers: [200]
              code_size: 40
              learning_rate: 0.001
              batch_size: 128,
              epochs: 15
              loss_name: mse
    """

    def __init__(self,
                 layers: Optional[List[int]] = None,
                 code_size: int = 10,
                 kernel_initializer: str = "he_normal",
                 act: str = "prelu",
                 last_act: str = "linear",
                 **ae_kwargs,
                 ):

        super().__init__(**ae_kwargs)

        self.layers: List[int] = [200] if layers is None else layers
        self.code_size: int = code_size
        self.kernel_initializer: str = kernel_initializer
        self.act: str = act.lower()
        self.last_act: str = last_act.lower()

    def create_model(self, input_dimension: int, **kwargs) -> KerasModel:
        """Compile a symmetric autoencoder model in Keras.

        Args:
            input_dimension: number of features in input data.

        Returns:
            A Keras model.
        """

        inputs = Input(shape=(input_dimension,))

        first_hidden_layer = self.layers[0]
        layer = Dense(first_hidden_layer, kernel_initializer=self.kernel_initializer,
                      activation=None if self.act == 'prelu' else self.act)(inputs)
        if self.act == 'prelu':
            layer = PReLU()(layer)
        for layer_size in self.layers[1:]:
            layer = Dense(layer_size, kernel_initializer=self.kernel_initializer,
                          activation=None if self.act == 'prelu' else self.act)(layer)
            if self.act == 'prelu':
                layer = PReLU()(layer)

        # encoding
        layer = Dense(self.code_size, kernel_initializer=self.kernel_initializer,
                      activation=None if self.act == 'prelu' else self.act,
                      name='encoded' if self.act != 'prelu' else None)(layer)
        if self.act == 'prelu':
            layer = PReLU(name='encoded' if self.act == 'prelu' else None)(layer)

        for layer_size in self.layers[::-1]:
            layer = Dense(layer_size, kernel_initializer=self.kernel_initializer,
                          activation=None if self.act == 'prelu' else self.act)(layer)
            if self.act == 'prelu':
                layer = PReLU()(layer)

        layer = Dense(input_dimension, kernel_initializer=self.kernel_initializer,
                      activation=None if self.last_act == 'prelu' else self.last_act)(layer)
        if self.last_act == 'prelu':
            layer = PReLU()(layer)

        self.model = KerasModel(inputs=inputs, outputs=layer)
        self.encoder = KerasModel(self.model.input, self.model.get_layer("encoded").output, name='encoder')
        return self.model

