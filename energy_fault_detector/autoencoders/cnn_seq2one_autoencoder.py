from __future__ import annotations

from typing import List, Optional, Tuple
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Dropout,
    Dense,
    Concatenate,
    Flatten,
    Lambda,
    MaxPooling1D,
)
from tensorflow.keras.models import Model as KerasModel

from energy_fault_detector.autoencoders.seq2one_autoencoder import Seq2OneAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder

logger = logging.getLogger('energy_fault_detector')


class CNNSeq2OneAutoencoder(Seq2OneAutoencoder):
    """CNN-based seq2one autoencoder.

    This model takes a sequence of length ``sequence_length`` and predicts/reconstructs
    the main features of the **last timestep** in that sequence. Optionally, per-timestep
    conditional features can be used as inputs.

    It is designed as a *short-context* model: the Conv1D stack captures local temporal dynamics,
    while global seasonality (e.g. daily/weekly patterns) is expected to be provided via
    conditional features such as time-of-day and day-of-week encodings.

    How this Seq2One Architecture works:
    The model will first use a stack of 1D-Convolutional layers to encode sequence data with dimensional reduction.
    After that, a series of MaxPooling-Layers followed by 1D convolutions is used to summarize the encoded sequences
    while subsequently reducing the number of timestamps per sequence. Once the encoded data is compact enough, the
    sample will be flattened and the prediction of the last last timestamp of the original input series is done by
    applying a stack of Dense layers.

    Input:
        (batch_size, sequence_length, n_main_features) [+ conditional features]
    Output:
        (batch_size, n_main_features)  (last timestep reconstruction)

    Args:
        sequence_builder: SequenceDatasetBuilder instance used to create the sequence datasets.
        layers: List of integers indicating the number of filters of the convolutional layers in the encoder.
            Defaults to [128, 64, 32] if None.
        decoder_layers: List of integers indicating the number of units in the layers of the decoder.
            If not provided, the decoder units will mirror the encoder filters in reverse order.
        code_size: Size of the latent representation (encoded vector).
        kernel_size: Specifies the length of the 1D convolution window. Default: 3.
        strides: Stride length of the 1D convolution window. Default: 1.
        dropout_rate: Dropout rate applied after each convolutional layer. Default: 0.
        conditional_features: Optional list of column names treated as conditional features. This will concatenate
            the conditions to the main inputs before feeding them to the encoder.
        max_encoding_size: Maximum size of the encoded vector. If the encoded vector is larger, it will be reduced
            by a factor of 2 (using max pooling) until it reaches this size. Default: 1000.
        ae_kwargs: Training-related parameters (learning_rate, batch_size, epochs, loss_name, early_stopping, etc.)
            are accepted as keyword arguments and forwarded to Autoencoder.__init__.

    Configuration example:

    .. code-block:: text

        train:
          autoencoder:
            name: CNNSeq2OneAutoencoder
            params:
              layers: [64, 64, 32]
              code_size: 8
              kernel_size: 3
              max_encoding_size: 1000
              sequence_builder:
                sequence_length: 36
                ts_freq: "5m"
                stride: 6
              conditional_features:
                - hour_of_day_sine
                - hour_of_day_cosine
                - day_of_week_sine
                - day_of_week_cosine
    """

    def __init__(
        self,
        sequence_builder: SequenceDatasetBuilder = None,
        layers: Optional[List[int]] = None,
        decoder_layers: Optional[List[int]] = None,
        code_size: int = 32,
        kernel_size: int = 3,
        strides: int = 1,
        dropout_rate: float = 0.0,
        max_encoding_size: int = 1000,
        **ae_kwargs,
    ):
        """Initialize a CNN-based seq2one autoencoder."""

        self.layers = layers or [128, 64, 32]
        self.decoder_layers = decoder_layers or [32, 64]
        self.code_size = code_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate
        self.max_encoding_size = max_encoding_size

        super().__init__(sequence_builder=sequence_builder, **ae_kwargs)

    def create_model(self, input_dimension: Tuple[int, int], condition_dimension: Optional[int] = None,
                     **kwargs) -> KerasModel:
        """Create the underlying CNN seq2one autoencoder model.

        Args:
            input_dimension: Tuple ``(sequence_length, n_main_features)`` for the main inputs.
            condition_dimension: Number of conditional features per timestep, or None.
            **kwargs: Additional keyword arguments (unused, for extensibility).

        Returns:
            The created Keras model.
        """
        sequence_length, n_main_features = input_dimension
        n_conditional_features = condition_dimension or 0

        # Main input (features to be reconstructed at last timestep)
        main_input = Input(shape=(sequence_length, n_main_features), name="main_input")
        encoder_input = main_input

        # Optional conditional input
        conditional_input = None
        if n_conditional_features > 0:
            conditional_input = Input(shape=(sequence_length, n_conditional_features), name="cond_input")
            encoder_input = Concatenate(axis=-1)([main_input, conditional_input])

        # Encoder: Conv Stack
        x = encoder_input
        for i, n_filters in enumerate(self.layers):
            x = Conv1D(
                filters=n_filters,
                kernel_size=self.kernel_size,
                padding="same",
                strides=self.strides,
                activation="relu",
            )(x)
            x = BatchNormalization()(x)
            if self.dropout_rate > 0:
                x = Dropout(rate=self.dropout_rate)(x)

        x = Conv1D(filters=self.code_size, kernel_size=self.kernel_size, strides=1, padding="same")(x)
        if self.code_size * self.sequence_builder.sequence_length > self.max_encoding_size:
            factor = self.code_size * self.sequence_builder.sequence_length / self.max_encoding_size
            for i in range(int(np.ceil(np.log2(factor)))):
                # Reduce until encoding size is below max_encoding_size
                x = MaxPooling1D(pool_size=2)(x)
                x = Conv1D(filters=self.code_size, kernel_size=self.kernel_size, strides=1, padding="same")(x)

        encoded = Flatten(name='encoding')(x)

        # Encoder model for latent representation
        if conditional_input is not None:
            self.encoder = tf.keras.Model(
                inputs=[main_input, conditional_input],
                outputs=encoded,
                name="encoder",
            )
        else:
            self.encoder = tf.keras.Model(
                inputs=main_input,
                outputs=encoded,
                name="encoder",
            )

        # Decoder: latent (+ last conditionas) -> reconstruction
        # Use the actual encoded dimension for the decoder input
        latent_dim = int(encoded.shape[-1])
        latent_input = Input(shape=(latent_dim,), name="latent_input")

        if n_conditional_features > 0:
            # Decoder gets the last timestep of the conditional features as context
            cond_last_input = Input(shape=(n_conditional_features,), name="cond_last_input")
            z = Concatenate(name="dec_concat")([latent_input, cond_last_input])
        else:
            cond_last_input = None
            z = latent_input

        z = Dense(self.decoder_layers[0], activation="relu")(z)
        for layer_size in self.decoder_layers[1:]:
            z = Dense(layer_size, activation="relu")(z)
        reconstruction = Dense(units=n_main_features, name="reconstruction")(z)

        # Stand-alone decoder model
        if conditional_input is not None:
            self.decoder = tf.keras.Model(
                inputs=[latent_input, cond_last_input],
                outputs=reconstruction,
                name="decoder",
            )
        else:
            self.decoder = tf.keras.Model(
                inputs=latent_input,
                outputs=reconstruction,
                name="decoder",
            )

        # Full AE
        if cond_last_input is not None:
            encoded = self.encoder(inputs=[main_input, conditional_input])
            cond_last = Lambda(lambda c: c[:, -1, :], name="cond_last")(conditional_input)
            decoded = self.decoder([encoded, cond_last])
            self.model = tf.keras.Model(
                inputs=[main_input, conditional_input],
                outputs=decoded,
                name="cnn_seq2one_autoencoder",
            )
        else:
            encoded = self.encoder(main_input)
            decoded = self.decoder(encoded)
            self.model = tf.keras.Model(
                inputs=main_input,
                outputs=decoded,
                name="cnn_seq2one_autoencoder",
            )

        return self.model
