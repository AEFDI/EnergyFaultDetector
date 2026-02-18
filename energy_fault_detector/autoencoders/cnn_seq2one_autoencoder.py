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
    Flatten,
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
    """

    def __init__(
        self,
        sequence_builder: SequenceDatasetBuilder = None,
        filters: Optional[List[int]] = None,
        kernel_size: int = 3,
        strides: int = 1,
        dropout_rate: float = 0.0,
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
        """Initialize a CNN-based seq2one autoencoder.

        Args:
            sequence_builder: SequenceDatasetBuilder instance used to create the sequence datasets.
            filters: List of integers indicating the number of filters of the convolutional layers in the encoder.
                Defaults to [128, 64, 32] if None.
            kernel_size: Specifies the length of the 1D convolution window.
            strides: Stride length of the 1D convolution window.
            dropout_rate: Dropout rate applied after each convolutional layer.
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
        self.filters = filters or [128, 64, 32]
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate

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

        # Encoder
        x = encoder_input
        for i, n_filters in enumerate(self.filters):
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
            
            # The last layer of the encoder should be named "encoded"
            if i == len(self.filters) - 1:
                # We apply Flatten if we want a single vector for the whole sequence
                # or we just use the last temporal output if it's already one.
                # Since it's seq2one, we probably want to flatten or global pool
                # to get a latent vector that represents the whole sequence.
                # However, the user's CNNAutoencoder (seq2seq) didn't flatten.
                # For seq2one, we MUST get to a fixed-size vector.
                encoded = Flatten(name="encoded")(x)
            
        # Encoder model for latent representation
        inputs = [main_input]
        if conditional_input is not None:
            inputs.append(conditional_input)

        self.encoder = tf.keras.Model(
            inputs=inputs,
            outputs=encoded,
            name="encoder",
        )

        # Decoder: map encoded vector directly to last-timestep features
        reconstruction = Dense(
            units=n_main_features,
            name="reconstruction",
        )(encoded)

        self.model = tf.keras.Model(
            inputs=inputs,
            outputs=reconstruction,
            name="cnn_seq2one_autoencoder",
        )

        return self.model
