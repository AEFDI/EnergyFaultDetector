"""Conditional autoencoder implementation (deterministic)."""

from typing import List

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Dense, PReLU, Input, Concatenate

from energy_fault_detector.core import Autoencoder


class ConditionalAE(Autoencoder):
    """Conditional symmetric autoencoder. Same as the MultilayerAutoencoder, where we use certain features in the
    input as conditions. These are concatenated to the input of both the encoder and decoder.

    NOTE: If the input of the fit, tune or predict method is a numpy array or a tensorflow tensor, we assume that the
        first couple of columns are the conditions.

    Args:
        layers: list of integers indicating the size (# units) of the layers in both the encoder and in
            the decoder (reversed order in this case). Default [200]
        code_size: number of units of the encoded layer (bottleneck layer).
            (number of features to compress the input features to). Default 10.
        learning_rate: learning rate of the adam optimizer. Default 0.001
        batch_size: number of samples per batch. Default 128
        epochs: number of epochs to run. Default 10
        loss_name: name of loss metric to use. Default mean_squared_error
        metrics: list of additional metrics to track. Default [mean_absolute_error].
        act: activation function to use, prelu, relu, ... Defaults to prelu.
        last_act: activation function for last layer, prelu, relu, sigmoid, linear... Defaults to linear.
        kernel_initializer: initializer to use in each layer. Default he_normal.
        early_stopping: Whether to use EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5,
            restore_best_weights=True). Cannot be used if there is no validation data. In that case, add a callback
            directly via the fit method.
        decay_rate: learning rate decay. Optional. If not defined, a fixed learning rate is used.
        decay_steps: number of steps to decay learning rate over. Optional.
        patience: parameter for early stopping. If early stopping is used the training will end if more than
            patience epochs in a row have not shown an improved loss. (Default is 3)
        min_delta: parameter of the early stopping callback. If the losses of an epoch and the next epoch differ
            by less than min_delta, they are considered equal (i.e. no improvement).
        noise: float value that determines the influence of the noise term on the training input. High values mean
            highly noisy input. 0 means no noise at all. Default 0. If noise >0 is used validation metrics will not be
            affected by it. Thus training loss and validation loss can differ depending on the magnitude of noise.

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

    def __init__(self, conditional_features: List[str] = None, layers: List[int] = None, code_size: int = 10,
                 learning_rate: float = 0.001, batch_size: int = 128, epochs: int = 10,
                 loss_name: str = 'mean_squared_error', metrics: List[str] = None,
                 kernel_initializer: str = 'he_normal', act: str = 'prelu',
                 last_act: str = 'linear', early_stopping: bool = False, decay_rate: float = None,
                 decay_steps: int = None, patience: int = 3, min_delta: float = 1e-4,
                 noise: float = 0.):

        super().__init__(conditional_features=conditional_features, learning_rate=learning_rate, batch_size=batch_size,
                         epochs=epochs, loss_name=loss_name, metrics=metrics, decay_rate=decay_rate,
                         decay_steps=decay_steps, early_stopping=early_stopping, patience=patience, min_delta=min_delta,
                         noise=noise)

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
