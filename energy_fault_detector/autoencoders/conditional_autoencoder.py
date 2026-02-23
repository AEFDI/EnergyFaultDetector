"""Conditional autoencoder implementation (deterministic)."""

from typing import List, Optional

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Dense, PReLU, Input, Concatenate

from energy_fault_detector.core.autoencoder import Autoencoder


class ConditionalAE(Autoencoder):
    """Conditional symmetric autoencoder. Same as the MultilayerAutoencoder, where we use certain features in the
    input as conditions. These are concatenated to the input of both the encoder and decoder.

    NOTE: If the input of the fit, tune or predict method is a numpy array or a tensorflow tensor, we assume that the
        first couple of columns are the conditions.

    Args:
        conditional_features: list of feature names that are used as conditions. The conditional feature vector is
            concatenated to the input of both the encoder and decoder.
        layers: list of integers indicating the size (# units) of the layers in both the encoder and in
            the decoder (reversed order in this case). Default [200]
        code_size: number of units of the encoded layer (bottleneck layer).
            (number of features to compress the input features to). Default 10.
                kernel_initializer: initializer to use in each layer. Default he_normal.
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
            name: ConditionalAutoencoder
            params:
              layers: [200]
              code_size: 40
              learning_rate: 0.001
              batch_size: 128,
              epochs: 15
              loss_name: mse
              conditional_features:
               - condition1
               - condition2
    """

    def __init__(self,
                 conditional_features: Optional[List[str]] = None,
                 layers: Optional[List[int]] = None,
                 code_size: int = 10,
                 kernel_initializer: str = "he_normal",
                 act: str = "prelu",
                 last_act: str = "linear",
                 **ae_kwargs,
                 ):

        super().__init__(conditional_features=conditional_features, **ae_kwargs)

        self.layers: List[int] = [200] if layers is None else layers
        self.code_size: int = code_size
        self.kernel_initializer: str = kernel_initializer
        self.act: str = act
        self.last_act: str = last_act

    def create_model(self, input_dimension: int, condition_dimension: int, **kwargs) -> KerasModel:
        """Compile a symmetric dense autoencoder.

        Args:
            input_dimension: number of features in input data.
            condition_dimension: number of features in conditional data.

        Returns:
            A Keras model.
        """

        inputs = Input(shape=(input_dimension,))
        conditions = Input(shape=(condition_dimension,))
        combined_input = Concatenate()([inputs, conditions])

        first_hidden_layer = self.layers[0]
        layer = Dense(first_hidden_layer, kernel_initializer=self.kernel_initializer,
                      activation=None if self.act=='prelu' else self.act)(combined_input)
        if self.act == 'prelu':
            layer = PReLU()(layer)
        for layer_size in self.layers[1:]:
            layer = Dense(layer_size, kernel_initializer=self.kernel_initializer,
                          activation=None if self.act=='prelu' else self.act)(layer)
            if self.act == 'prelu':
                layer = PReLU()(layer)

        # encoding
        layer = Dense(self.code_size, kernel_initializer=self.kernel_initializer,
                      activation=None if self.act=='prelu' else self.act,
                      name='encoded' if self.act != 'prelu' else None)(layer)
        if self.act == 'prelu':
            layer = PReLU(name='encoded' if self.act == 'prelu' else None)(layer)

        # Create encoder
        self.encoder = KerasModel(inputs=[inputs, conditions], outputs=layer, name='encoder')

        # decoding
        layer = Concatenate()([layer, conditions])  # add conditions to decoder input
        for layer_size in self.layers[::-1]:
            layer = Dense(layer_size, kernel_initializer=self.kernel_initializer,
                          activation=None if self.act == 'prelu' else self.act)(layer)
            if self.act == 'prelu':
                layer = PReLU()(layer)

        layer = Dense(input_dimension, kernel_initializer=self.kernel_initializer,
                      activation=None if self.last_act == 'prelu' else self.last_act)(layer)
        if self.last_act == 'prelu':
            layer = PReLU()(layer)

        # Create AE
        self.model = KerasModel(inputs=[inputs, conditions], outputs=layer)

        return self.model
