from __future__ import annotations

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
    RepeatVector,
    Dense,
    TimeDistributed,
    Concatenate,
)
from tensorflow.keras.models import Model as KerasModel

from .seq2seq_autoencoder import Seq2SeqAutoencoder
from ..data_splitting.sequence_dataset import SequenceDatasetBuilder


class LSTMSeqAutoencoder(Seq2SeqAutoencoder):
    """LSTM-based sequence autoencoder.

    This model reconstructs only the main features (all columns that are not listed in
    ``conditional_features``). It can optionally use per-timestep conditional features
    as additional inputs to the encoder/decoder.

    The input to the model is a sequence of length ``sequence_length`` with shape
    ``(batch_size, sequence_length, n_main_features)`` (plus conditional features if used).
    The output is a sequence of the same length and feature dimension for the main features.
    """

    def __init__(
        self,
        sequence_builder: SequenceDatasetBuilder = None,
        layers: Optional[List[int]] = None,
        dropout_rate: float = 0.0,
        regularization: float = 0.01,
        stateful: bool = False,
        conditional_features: Optional[List[str]] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 10,
        loss_name: str = "mean_squared_error",
        metrics: Optional[List[str]] = None,
        decay_rate: float = None,
        decay_steps: float = None,
        early_stopping: bool = False,
        patience: int = 3,
        min_delta: float = 1e-4,
        noise: float = 0.0,
    ) -> None:
        """Initialize an LSTM-based sequence autoencoder.

        Args:
            sequence_builder: SequenceDatasetBuilder instance used to create the sequence datasets.
            layers: List with the number of LSTM units per layer in the encoder. The decoder
                mirrors these layers in reverse order. Defaults to [128, 64, 32] if None.
            dropout_rate: Dropout rate applied after each LSTM layer.
            regularization: L2 regularization strength for the first encoder LSTM layer.
            stateful: Whether to use stateful LSTMs.
            conditional_features: Optional list of column names treated as conditional features.
            learning_rate: Initial learning rate for the optimizer.
            batch_size: Batch size during training.
            epochs: Number of epochs for initial training.
            loss_name: Loss function name passed to ``model.compile``.
            metrics: Additional metrics to track during training.
            decay_rate: Exponential decay rate for the learning rate (optional).
            decay_steps: Number of steps over which to apply learning rate decay (optional).
            early_stopping: If True, enable EarlyStopping in the base Autoencoder.
            patience: Patience for EarlyStopping (number of epochs without improvement).
            min_delta: Minimum change in monitored quantity for EarlyStopping to qualify as an improvement.
            noise: Standard deviation of Gaussian noise applied to inputs during training (denoising AE).
        """
        self.layers = layers or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.stateful = stateful

        metrics = metrics or ["mean_absolute_error"]

        super().__init__(
            sequence_builder=sequence_builder,
            conditional_features=conditional_features,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            loss_name=loss_name,
            metrics=metrics,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            noise=noise,
        )

    def create_model(
        self,
        input_dimension: Tuple[int, int],
        condition_dimension: Optional[int] = None,
        **kwargs,
    ) -> KerasModel:
        """Create the underlying LSTM autoencoder model.

        Args:
            input_dimension: Tuple ``(sequence_length, n_main_features)`` describing the
                main-feature input shape.
            condition_dimension: Number of conditional features (per timestep) to use as
                additional inputs, or None if no conditions are used.
            **kwargs: Additional keyword arguments (currently unused, kept for extensibility).

        Returns:
            The created Keras model.
        """
        sequence_length, n_main_features = input_dimension
        n_conditional_features = condition_dimension or 0

        # Main input (features to be reconstructed)
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

        # Encoder
        first_hidden_units = self.layers[0]
        encoder_output = LSTM(
            units=first_hidden_units,
            return_sequences=True,
            stateful=self.stateful,
            kernel_regularizer=regularizers.l2(self.regularization),
        )(encoder_input)
        encoder_output = Dropout(rate=self.dropout_rate)(encoder_output)

        for layer_size in self.layers[1:-1]:
            encoder_output = LSTM(
                units=layer_size,
                return_sequences=True,
                stateful=self.stateful,
            )(encoder_output)
            encoder_output = Dropout(rate=self.dropout_rate)(encoder_output)

        encoded = LSTM(
            units=self.layers[-1],
            name="encoded",
            return_sequences=False,
            stateful=self.stateful,
        )(encoder_output)
        encoded = Dropout(rate=self.dropout_rate)(encoded)

        # Encoder model (for latent representation)
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

        # Decoder
        decoder_output = RepeatVector(sequence_length)(encoded)
        for layer_size in reversed(self.layers):
            decoder_output = LSTM(
                units=layer_size,
                return_sequences=True,
                stateful=self.stateful,
            )(decoder_output)
            decoder_output = Dropout(rate=self.dropout_rate)(decoder_output)

        reconstruction = TimeDistributed(
            Dense(units=n_main_features),
            name="reconstruction",
        )(decoder_output)

        if conditional_input is not None:
            self.model = tf.keras.Model(
                inputs=[main_input, conditional_input],
                outputs=reconstruction,
            )
        else:
            self.model = tf.keras.Model(
                inputs=main_input,
                outputs=reconstruction,
            )

        return self.model
