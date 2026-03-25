from __future__ import annotations

from typing import List, Optional, Tuple
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Conv1DTranspose,
    BatchNormalization,
    Dropout,
    Dense,
    Concatenate,
)
from tensorflow.keras.models import Model as KerasModel

from .seq2seq_autoencoder import SeqAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class CNNAutoencoder(SeqAutoencoder):
    """CNN-based Seq2Seq autoencoder

    Args:
        sequence_builder: SequenceDatasetBuilder instance used to create the sequence datasets.
        layers: List of integers indicating the number of filters of the convolutional layers in the encoder.
            Defaults to [128, 64, 32] if None. The last number of filters is effectively the code size / latent
            dimension of the autoencoder.
        kernel_size: Specifies the length of the 1D convolution window. Default: 3.
        strides: Stride length of the 1D convolution window. Default: 1.
        dropout_rate: Dropout rate applied after each convolutional layer. Default: 0.
        conditional_features: Optional list of column names treated as conditional features. This will concatenate
            the conditions to the main inputs before feeding them to the encoder.
        ae_kwargs: Training-related parameters (learning_rate, batch_size, epochs, loss_name, early_stopping, etc.)
            are accepted as keyword arguments and forwarded to Autoencoder.__init__.

    Attributes:
        model: keras Model object.
        history: dictionary with the losses for each epoch.

    Configuration example:

    .. code-block:: text

        train:
          autoencoder:
            name: CNNAutoencoder
            time_series_sampler:
              name: TimeSeriesRandomSampler
              params:
                sequence_length: 36
                ts_freq: np.timedelta64(10, 'm')
                pad_incomplete: false
                pad_value: 0
            params:
              sequence_length: 12
              dropout_rate: 0.0
              filters: [128, 64, 32]
              learning_rate: 0.001
              batch_size: 128,
              epochs: 15
              loss_name: mse

    """

    def __init__(self, sequence_builder: SequenceDatasetBuilder = None, layers: Optional[List[int]] = None,
                 kernel_size: int = 3, strides: int = 1, dropout_rate: float = 0.0, **ae_kwargs):

        self.layers = layers or [128, 64, 32]
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate

        super().__init__(sequence_builder=sequence_builder, **ae_kwargs)

    def create_model(self, input_dimension: Tuple[int, int, int], **kwargs) -> KerasModel:
        """Create ConvAE Model

        Args:
            input_dimension: tuple (n_samples, sequence_len, n_features)
        """

        inputs = Input(shape=(input_dimension[1], input_dimension[2]), name="main_input")

        # Encoder
        x = inputs
        for nf in self.layers:
            x = Conv1D(
                filters=nf,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding="same",
                activation="relu",
            )(x)
            x = BatchNormalization()(x)
            if self.dropout_rate > 0:
                x = Dropout(rate=self.dropout_rate)(x)

        encoded = BatchNormalization(name="encoded")(x)

        # Decoder
        x = encoded
        for nf in self.layers[::-1]:
            x = Conv1DTranspose(
                filters=nf,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding="same",
                activation="relu",
            )(x)
            x = BatchNormalization()(x)
            if self.dropout_rate > 0:
                x = Dropout(rate=self.dropout_rate)(x)

        outputs = Conv1DTranspose(
            filters=input_dimension[2],
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            name="reconstruction",
        )(x)

        self.model = KerasModel(inputs=inputs, outputs=outputs)
        return self.model
