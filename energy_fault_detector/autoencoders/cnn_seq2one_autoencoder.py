from __future__ import annotations

from typing import List, Optional, Tuple
import logging

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Dropout,
    Dense,
    Concatenate,
    Flatten,
    Lambda
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

    It is designed as a *short-context* model: the Conv1D stack around the last timestep
    captures local temporal dynamics, while global seasonality (e.g. daily/weekly patterns)
    is expected to be provided via **conditional features** such as time-of-day and
    day-of-week encodings.

    For data with strong daily/weekly seasonality (e.g. heat pumps, wind turbines),
    it is highly recommended to provide time-of-day and day-of-week features as
    ``conditional_features``. The CNN encoder then focuses on local dynamics, while
    the conditional features provide global seasonal context to the decoder.

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
        encoder_aggregation: How to aggregate encoder time dimension into a vector.
            Options:
              - 'last': use features at the last timestep of the Conv stack (default, short-context).
              - 'flatten': Flatten all timesteps and channels, then Dense(code_size).
                This gives full-window context but scales poorly for long sequences.
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
              encoder_aggregation: "last"
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
        encoder_aggregation: str = 'last',
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

        if encoder_aggregation not in ("last", "flatten"):
            raise ValueError(f"encoder_aggregation must be 'last' or 'flatten', got {encoder_aggregation!r}")
        self.encoder_aggregation = encoder_aggregation

        if sequence_builder is not None:
            rf = 1 + (self.kernel_size - 1) * len(self.layers)
            if rf < sequence_builder.sequence_length and self.encoder_aggregation == 'last':
                logger.warning(
                    "CNNSeq2OneAutoencoder: receptive field (%d timesteps) is smaller than "
                    "sequence_length (%d). Only the last ~%d timesteps effectively influence "
                    "the last-timestep representation when using encoder_aggregation='last'. "
                    "Consider reducing sequence_length, increasing kernel_size / number of "
                    "layers, or using encoder_aggregation='flatten' for short sequences.",
                    rf, sequence_builder.sequence_length, rf
                )

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

        # Encoder: Aggregate over time dimension
        if self.encoder_aggregation == "last":
            # short-context: last timestep only (receptive field limited by conv stack)
            x_agg = Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)
        else:  # "flatten"
            # full-window context: flatten all timesteps and channels, then Dense(code_size)
            x_agg = Flatten(name="flatten")(x)

        encoded = Dense(self.code_size, activation="relu", name="encoded")(x_agg)

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
        latent_input = Input(shape=(self.code_size,), name="latent_input")

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
            self.model = tf.keras.Model(
                inputs=main_input,
                outputs=decoded,
                name="cnn_seq2one_autoencoder",
            )

        return self.model
