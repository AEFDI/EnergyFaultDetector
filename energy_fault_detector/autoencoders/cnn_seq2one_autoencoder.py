from __future__ import annotations

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Dropout,
    Dense,
    Concatenate,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
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
            decoder_layers: List of integers indicating the number of units in the layers of the decoder.
                If not provided, the number of units in the decoder will be the same as the number of filters in the
                encoder, in reverse order.
            kernel_size: Specifies the length of the 1D convolution window. Default: 3.
            strides: Stride length of the 1D convolution window. Default: 1.
            dropout_rate: Dropout rate applied after each convolutional layer. Default: 0.
            conditional_features: Optional list of column names treated as conditional features. This will concatenate
                the conditions to the main inputs before feeding them to the encoder.
            pooling: Pooling strategy to use for the encoder. Either 'average' or 'max'. Default: 'average'.
            ae_kwargs: Training-related parameters (learning_rate, batch_size, epochs, loss_name, early_stopping, etc.)
                are accepted as keyword arguments and forwarded to Autoencoder.__init__.

    Configuration example:

    .. code-block:: text

        train:
          autoencoder:
            name: CNNSeq2OneAutoencoder
            params:
              layers: [100, 50, 25]
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
        decoder_layers: Optional[List[int]] = None,
        code_size: int = 32,
        kernel_size: int = 3,
        strides: int = 1,
        dropout_rate: float = 0.0,
        pooling: str = 'average',
        **ae_kwargs,
    ):
        """Initialize a CNN-based seq2one autoencoder."""

        self.layers = layers or [128, 64, 32]
        self.decoder_layers = decoder_layers or list(reversed(self.layers))
        self.code_size = code_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate
        if pooling not in ['average', 'max']:
            raise ValueError(f"pooling must be either 'average' or 'max', not {pooling}")
        self.pooling = pooling

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

        # Pool + dense for encoding
        if self.pooling == 'max':
            x = GlobalMaxPooling1D()(x)
        else:
            x = GlobalAveragePooling1D()(x)
        encoded = Dense(self.code_size, activation="relu", name="encoded")(x)

        # Encoder model for latent representation
        inputs = [main_input]
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
        # TODO: tbd - do we add the conditional input again, like in the dense AE?
        #       In that case we need to reshape the conditional inputs to 1D again
        latent_input = Input(shape=(self.code_size,), name="latent_input")
        z = Dense(self.decoder_layers[0], activation="relu")(latent_input)
        for layer_size in self.decoder_layers[1:]:
            z = Dense(layer_size, activation="relu")(z)
        reconstruction = Dense(units=n_main_features, name="reconstruction")(z)

        # Stand-alone decoder model
        self.decoder = tf.keras.Model(
            inputs=latent_input,
            outputs=reconstruction,
            name="decoder",
        )

        if conditional_input is not None:
            encoded = self.encoder(inputs=[main_input, conditional_input])
        else:
            encoded = self.encoder(inputs=main_input)

        decoded = self.decoder(encoded)

        if conditional_input is not None:
            self.model = tf.keras.Model(
                inputs=[main_input, conditional_input],
                outputs=decoded,
                name="cnn_seq2one_autoencoder",
            )
        else:
            self.model = tf.keras.Model(
                inputs=main_input,
                outputs=decoded,
                name="cnn_seq2one_autoencoder",
            )

        return self.model
