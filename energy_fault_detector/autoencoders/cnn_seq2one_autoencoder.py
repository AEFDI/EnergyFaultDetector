from __future__ import annotations

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Conv1DTranspose,
    BatchNormalization,
    Dropout,
    Dense,
    Concatenate,
    Flatten,
    Reshape,
)
from tensorflow.keras.models import Model as KerasModel

from energy_fault_detector.autoencoders.seq2one_autoencoder import Seq2OneAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class CNNSeq2OneAutoencoder(Seq2OneAutoencoder):
    """CNN-based seq2one autoencoder.

    This model takes a sequence of length ``sequence_length`` and predicts/reconstructs
    the main features of the **last timestep** in that sequence. Optionally, per-timestep
    conditional features can be used as inputs.

    Input:
        (batch_size, sequence_length, n_main_features) [+ conditional features]
    Output:
        (batch_size, n_main_features)  (last timestep reconstruction)

    Args:
            sequence_builder: SequenceDatasetBuilder instance used to create the sequence datasets.
            layers: List of integers indicating the number of filters of the convolutional layers in the encoder.
                Defaults to [128, 64, 32] if None.
            kernel_size: Specifies the length of the 1D convolution window.
            strides: Stride length of the 1D convolution window.
            dropout_rate: Dropout rate applied after each convolutional layer.
            conditional_features: Optional list of column names treated as conditional features. This will concatenate
                the conditions to the main inputs before feeding them to the encoder.
            ae_kwargs: Training-related parameters (learning_rate, batch_size, epochs, loss_name, early_stopping, etc.)
                are accepted as keyword arguments and forwarded to Autoencoder.__init__.

    Configuration example:

    .. code-block:: text

        train:
          autoencoder:
            name: CNNSeq2OneAutoencoder
            params:
              filters: [100, 50, 25]
              kernel_size: 5
              sequence_builder:
                sequence_length: 10
                ts_freq: "10m"
                overlap: 9
    """

    def __init__(
        self,
        sequence_builder: SequenceDatasetBuilder = None,
        layers: Optional[List[int]] = None,
        code_size: int = 32,
        kernel_size: int = 3,
        strides: int = 1,
        dropout_rate: float = 0.0,
        **ae_kwargs,
    ):
        """Initialize a CNN-based seq2one autoencoder."""

        self.layers = layers or [128, 64, 32]
        self.code_size = code_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate

        super().__init__(sequence_builder=sequence_builder, **ae_kwargs)

    def create_model(
        self,
        input_dimension: Tuple[int, int],
        condition_dimension: Optional[int] = None,
        **kwargs,
    ) -> KerasModel:
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
        main_input = Input(
            shape=(sequence_length, n_main_features),
            name="main_input",
        )
        encoder_input = main_input

        # Optional conditional input
        conditional_input = None
        if n_conditional_features > 0:
            conditional_input = Input(
                shape=(sequence_length, n_conditional_features),
                name="cond_input",
            )
            encoder_input = Concatenate(axis=-1)([main_input, conditional_input])

        # Encoder - Conv Stack
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

        # Store the shape to reshape back in the decoder
        # x.shape is (batch, sequence_length / strides^num_layers, filters)
        conv_shape = x.shape[1:]  # (T_enc, F_enc)
        flat_dim = int(conv_shape[0] * conv_shape[1])
        flat = Flatten()(x)
        encoded = Dense(self.code_size, activation="relu", name="encoded")(flat)

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

        # Decoder: latent -> reconstruction
        latent_input = Input(shape=(self.code_size,), name="latent_input")
        z = Dense(flat_dim, activation="relu")(latent_input)
        z = Reshape(conv_shape)(z)

        # Decoder - Conv Transpose Stack
        y = z
        for n_filters in self.layers[-2::-1] + [n_main_features]:
            y = Conv1DTranspose(
                filters=n_filters,
                kernel_size=self.kernel_size,
                padding="same",
                strides=self.strides,
                activation="relu",
            )(y)
            y = BatchNormalization()(y)
            if self.dropout_rate > 0:
                y = Dropout(rate=self.dropout_rate)(y)

        y = Flatten()(y)
        reconstruction = Dense(units=n_main_features, name="reconstruction")(y)

        # Stand-alone decoder model
        self.decoder = tf.keras.Model(
            inputs=latent_input,
            outputs=reconstruction,
            name="decoder",
        )

        if conditional_input is not None:
            self.model = tf.keras.Model(
                inputs=[main_input, conditional_input],
                outputs=self.decoder(self.encoder(inputs=[main_input, conditional_input])),
                name="cnn_seq2one_autoencoder",
            )
        else:
            self.model = tf.keras.Model(
                inputs=main_input,
                outputs=self.decoder(self.encoder(inputs=main_input)),
                name="cnn_seq2one_autoencoder",
            )

        return self.model
