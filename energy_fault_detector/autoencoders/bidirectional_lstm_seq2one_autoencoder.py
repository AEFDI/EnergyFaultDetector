from __future__ import annotations

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Input,
    LSTM,
    Lambda,
    RepeatVector,
)
from tensorflow.keras.models import Model as KerasModel

from energy_fault_detector.autoencoders.seq2one_autoencoder import Seq2OneAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class BidirectionalLSTMSeq2OneAutoencoder(Seq2OneAutoencoder):
    """Bidirectional LSTM-based seq2one autoencoder.

    This model consumes a fixed-length time window and reconstructs the
    main features of the last timestep in that window. Conditional
    features can be provided per timestep and are concatenated to the
    encoder inputs without being reconstructed.
    """

    def __init__(
        self,
        sequence_builder: Optional[SequenceDatasetBuilder] = None,
        layers: Optional[List[int]] = None,
        dropout_rate: float = 0.0,
        regularization: float = 0.01,
        stateful: bool = False,
        merge_mode: str = "sum",
        **ae_kwargs,
    ):
        """Initialize a bidirectional LSTM-based seq2one autoencoder."""

        if merge_mode not in {"concat", "sum", "ave", "mul"}:
            raise ValueError(
                "merge_mode must be one of {'concat', 'sum', 'ave', 'mul'}."
            )

        self.layers = layers or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.stateful = stateful
        self.merge_mode = merge_mode

        super().__init__(sequence_builder=sequence_builder, **ae_kwargs)

    def create_model(
        self,
        input_dimension: Tuple[int, int],
        condition_dimension: Optional[int] = None,
        **kwargs,
    ) -> KerasModel:
        """Create the underlying bidirectional LSTM seq2one autoencoder model."""

        sequence_length, n_main_features = input_dimension
        n_conditional_features = condition_dimension or 0

        main_input = Input(
            shape=(sequence_length, n_main_features),
            name="main_input",
        )
        encoder_input = main_input

        conditional_input = None
        if n_conditional_features > 0:
            conditional_input = Input(
                shape=(sequence_length, n_conditional_features),
                name="cond_input",
            )
            encoder_input = Concatenate(axis=-1)([main_input, conditional_input])

        first_hidden_units = self.layers[0]
        encoder_output = Bidirectional(
            layer=LSTM(
                units=first_hidden_units,
                return_sequences=True,
                stateful=self.stateful,
                kernel_regularizer=regularizers.l2(self.regularization),
            ),
            merge_mode=self.merge_mode,
            name="encoder_bilstm_0",
        )(encoder_input)
        encoder_output = Dropout(rate=self.dropout_rate)(encoder_output)

        for layer_index, layer_size in enumerate(self.layers[1:-1], start=1):
            encoder_output = Bidirectional(
                layer=LSTM(
                    units=layer_size,
                    return_sequences=True,
                    stateful=self.stateful,
                ),
                merge_mode=self.merge_mode,
                name=f"encoder_bilstm_{layer_index}",
            )(encoder_output)
            encoder_output = Dropout(rate=self.dropout_rate)(encoder_output)

        encoded = Bidirectional(
            layer=LSTM(
                units=self.layers[-1],
                return_sequences=False,
                stateful=self.stateful,
            ),
            merge_mode=self.merge_mode,
            name="encoded",
        )(encoder_output)
        encoded = Dropout(rate=self.dropout_rate)(encoded)

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

        decoder_output = RepeatVector(n=sequence_length)(encoded)
        for layer_index, layer_size in enumerate(reversed(self.layers)):
            decoder_output = LSTM(
                units=layer_size,
                return_sequences=True,
                stateful=self.stateful,
                name=f"decoder_lstm_{layer_index}",
            )(decoder_output)
            decoder_output = Dropout(rate=self.dropout_rate)(decoder_output)

        last_timestep = Lambda(lambda t: t[:, -1, :], name="last_timestep")(decoder_output)
        reconstruction = Dense(
            units=n_main_features,
            name="reconstruction",
        )(last_timestep)

        if conditional_input is not None:
            self.model = tf.keras.Model(
                inputs=[main_input, conditional_input],
                outputs=reconstruction,
                name="bidirectional_lstm_seq2one_autoencoder",
            )
        else:
            self.model = tf.keras.Model(
                inputs=main_input,
                outputs=reconstruction,
                name="bidirectional_lstm_seq2one_autoencoder",
            )

        return self.model
