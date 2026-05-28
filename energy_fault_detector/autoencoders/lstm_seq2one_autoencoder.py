from __future__ import annotations

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
    Dense,
    Concatenate,
    RepeatVector,
    Lambda,
)
from tensorflow.keras.models import Model as KerasModel

from energy_fault_detector.autoencoders.seq2one_autoencoder import Seq2OneAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class LSTMSeq2OneAutoencoder(Seq2OneAutoencoder):
    """LSTM-based seq2one autoencoder.

    This model takes a sequence of length ``sequence_length`` and predicts/reconstructs
    the main features of the **last timestep** in that sequence. Optionally, per-timestep
    conditional features can be used as inputs.

    The encoder uses a stack of LSTM layers, while the decoder is an MLP with dense layers.
    The last encoder layer reduces the sequences to the last timestep, which is the fed to the
    MLP decoder.

    Input:
        (batch_size, sequence_length, n_main_features) [+ conditional features]
    Output:
        (batch_size, n_main_features)  (last timestep reconstruction)

    Args:
        sequence_builder: SequenceDatasetBuilder instance used to create the sequence datasets.
        layers: List with the number of LSTM units per encoder layer. Defaults to [128, 64, 32] if None.
        code_size: Size of the latent representation (encoded vector). Default: 32.
        decoder_layers: List of integers indicating the number of units in the layers of the decoder.
            If not provided, defaults to [32, 64].
        dropout_rate: Dropout rate applied after each LSTM layer.
        regularization: L2 regularization strength for the first encoder LSTM layer.
        conditional_features: Optional list of column names treated as conditional features. This will concatenate
            the conditions to the main inputs before feeding them to the encoder.
        ae_kwargs: Training-related parameters (learning_rate, batch_size, epochs, loss_name, early_stopping, etc.)
            are accepted as keyword arguments and forwarded to Autoencoder.__init__.

    Configuration example:

    .. code-block:: text

        train:
          autoencoder:
            name: LSTMSeq2OneAutoencoder
            params:
              layers: [100, 50, 25]
              code_size: 10
              regularization: 0.01
              sequence_builder:
                sequence_length: 10
                ts_freq: "10m"
                overlap: 9
    """

    def __init__(
        self,
            sequence_builder: Optional[SequenceDatasetBuilder] = None,
            layers: Optional[List[int]] = None,
            decoder_layers: Optional[List[int]] = None,
            code_size: int = 32,
            dropout_rate: float = 0.0,
            regularization: float = 0.01,
            **ae_kwargs,
    ):
        """Initialize an LSTM-based seq2one autoencoder."""

        self.layers = layers or [128, 64, 32]
        self.decoder_layers = decoder_layers or [32, 64]
        self.code_size = code_size
        self.dropout_rate = dropout_rate
        self.regularization = regularization

        super().__init__(sequence_builder=sequence_builder, **ae_kwargs)

    def create_model(
        self,
        input_dimension: Tuple[int, int],
        condition_dimension: Optional[int] = None,
        **kwargs,
    ) -> KerasModel:
        """Create the underlying LSTM seq2one autoencoder model.

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

        # Encoder:
        first_hidden_units = self.layers[0]
        encoder_output = LSTM(
            units=first_hidden_units,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(self.regularization),
        )(encoder_input)
        encoder_output = Dropout(rate=self.dropout_rate)(encoder_output)

        for layer_size in self.layers[1:]:
            encoder_output = (
                LSTM(
                units=layer_size,
                return_sequences=True,
            )(encoder_output))
            encoder_output = Dropout(rate=self.dropout_rate)(encoder_output)

        # Reduce to code_size
        encoded = LSTM(
            units=self.code_size,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(self.regularization),
        )(encoder_output)

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

        # Decoder: latent (+ last conditions) -> reconstruction
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

        if conditional_input is not None:
            enc = self.encoder(inputs=[main_input, conditional_input])
            cond_last = Lambda(lambda c: c[:, -1, :], name="cond_last")(conditional_input)
            decoded = self.decoder([enc, cond_last])
            self.model = tf.keras.Model(
                inputs=[main_input, conditional_input],
                outputs=decoded,
            )
        else:
            enc = self.encoder(main_input)
            decoded = self.decoder(enc)
            self.model = tf.keras.Model(
                inputs=main_input,
                outputs=decoded,
            )

        return self.model
