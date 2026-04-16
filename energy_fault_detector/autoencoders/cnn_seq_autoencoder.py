from __future__ import annotations

from typing import List, Optional, Tuple

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Conv1DTranspose,
    BatchNormalization,
    Dropout,
    Concatenate
)
from tensorflow.keras.models import Model as KerasModel

from .seq2seq_autoencoder import Seq2SeqAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class CNNAutoencoder(Seq2SeqAutoencoder):
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

    def create_model(self, input_dimension: Tuple[int, int], condition_dimension: Optional[int] = None,
                     **kwargs) -> KerasModel:
        """Create ConvAE Model.

        Args:
            input_dimension: Tuple ``(sequence_length, n_main_features)``.
            condition_dimension: Number of conditional features per timestep, or None.
        """

        sequence_length, n_main_features = input_dimension
        n_conditional_features = condition_dimension or 0

        # Main input
        main_input = Input(shape=(sequence_length, n_main_features), name="main_input")
        encoder_input = main_input

        # Optional conditional input — concatenated channel-wise before encoder
        conditional_input = None
        if n_conditional_features > 0:
            conditional_input = Input(
                shape=(sequence_length, n_conditional_features),
                name="cond_input",
            )
            encoder_input = Concatenate(axis=-1)([main_input, conditional_input])

        # Encoder
        x = encoder_input
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

        # Encoder model (for latent representations)
        encoder_inputs = [main_input, conditional_input] if conditional_input is not None else main_input
        self.encoder = KerasModel(inputs=encoder_inputs, outputs=encoded, name="encoder")

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
            filters=n_main_features,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            name="reconstruction",
        )(x)

        model_inputs = [main_input, conditional_input] if conditional_input is not None else main_input
        self.model = KerasModel(inputs=model_inputs, outputs=outputs)
        return self.model
