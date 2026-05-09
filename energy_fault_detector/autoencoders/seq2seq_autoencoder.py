from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback

from .seq2one_autoencoder import Seq2OneAutoencoder
from ..data_splitting.sequence_dataset import SequenceDatasetBuilder
from energy_fault_detector.autoencoders._sequence_utils import sequences_to_dataframe


# TODO: fix class hierarchy
class Seq2SeqAutoencoder(Seq2OneAutoencoder):
    """Base class for sequence autoencoders (e.g. LSTM, CNN) on time-series data.

    This class works directly with Pandas DataFrames that have a DatetimeIndex. It:

      * builds sequence datasets via ``SequenceDatasetBuilder``,
      * trains on 3D sequences (optionally with per-timestep conditional features),
      * reconstructs the input sequence (of the main (non-conditional) features).
      * flattens the reconstructed sequences into a DataFrame with one row per timestamp.

    If ``conditional_features`` is provided:

      * main features = all columns minus those in ``conditional_features``,
      * model input: (sequence_main, sequence_cond),
      * model output: sequence_main.

    """

    def __init__(
        self,
        sequence_builder: SequenceDatasetBuilder = None,
        **ae_kwargs
    ):
        """Initialize a sequence autoencoder.

        Args:
            sequence_builder: SequenceDatasetBuilder instance used to create sliding-window datasets.
            **ae_kwargs: Additional keyword arguments passed to ``Autoencoder.__init__``.
        """
        super().__init__(**ae_kwargs)

        self.is_seq2one: bool = False
        self.sequence_builder = sequence_builder
        self.window_timestamps_: Optional[np.ndarray] = None

    def _predict(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Reconstruct main features from input time series.

        Args:
            x: Input data as a DataFrame with DatetimeIndex.
            **kwargs: Additional keyword arguments passed to ``model.predict``.

        Returns:
            DataFrame containing reconstructed main features, indexed by timestamp.
        """
        self._check_sequence_builder()
        dataset, window_timestamps = self.sequence_builder.build_sliding_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=False,
            predict_mode=True,
        )
        self.window_timestamps_ = window_timestamps

        predictions = self.model.predict(dataset, **kwargs)

        if self.conditional_features:
            main_columns = [c for c in x.columns if c not in self.conditional_features]
        else:
            main_columns = list(x.columns)

        reconstruction = sequences_to_dataframe(
            sequences=predictions,
            timestamps=window_timestamps,
            columns=main_columns,
        )
        return reconstruction

    def encode(self, x: pd.DataFrame, conditions: pd.DataFrame = None) -> np.ndarray:
        """Encode input time series into the latent space.

        Args:
            x: Input data as a DataFrame with DatetimeIndex.
            conditions: Optional DataFrame with conditional features. Currently not used. TODO: needed?

        Returns:
            NumPy array with latent representations for each sequence window.
        """
        self._check_sequence_builder()
        dataset, _ = self.sequence_builder.build_sliding_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=False,
            predict_mode=True,
        )
        return self.encoder.predict(dataset)

    def get_reconstruction_error(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute reconstruction error for main features.

        The error is defined as reconstructed_main_features - original_main_features, aligned by timestamp.

        Args:
            x: Input data as a DataFrame with DatetimeIndex.
            **kwargs: Additional keyword arguments passed to ``_predict``.

        Returns:
            DataFrame containing reconstruction errors for each main feature and timestamp.
        """
        reconstruction = self._predict(x, **kwargs)

        if self.conditional_features:
            main_columns = [c for c in x.columns if c not in self.conditional_features]
        else:
            main_columns = list(x.columns)

        x_main = x[main_columns]
        x_main = x_main.loc[reconstruction.index]

        error = reconstruction - x_main
        return error

    def _fit_internal(
        self,
        x: pd.DataFrame,
        x_val: Optional[pd.DataFrame],
        total_epochs: int,
        initial_epoch: int = 0,
        learning_rate: Optional[float] = None,
        **kwargs,
    ) -> "Seq2SeqAutoencoder":
        """Internal training helper working with tf.data.Datasets.

        Args:
            x: Training data as a DataFrame with DatetimeIndex.
            x_val: Optional validation data as a DataFrame with DatetimeIndex.
            total_epochs: Total number of epochs to run.
            initial_epoch: Epoch number to start from (used for tuning).
            learning_rate: Optional new learning rate to use for training.
            **kwargs: Additional keyword arguments passed to ``model.fit``.

        Returns:
            The trained (or tuned) ``Seq2SeqAutoencoder`` instance.
        """
        if learning_rate is not None:
            self.compile_model(new_learning_rate=learning_rate)
        else:
            self.compile_model()

        train_dataset, _ = self.sequence_builder.build_sliding_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=True,
        )

        val_dataset = None
        if x_val is not None:
            val_dataset, _ = self.sequence_builder.build_sliding_dataset(
                df=x_val,
                batch_size=self.batch_size,
                conditional_features=self.conditional_features,
                shuffle=False,
                predict_mode=True,
            )

        callbacks = list(self.callbacks)
        if "callbacks" in kwargs:
            callbacks += kwargs["callbacks"]
            kwargs.pop("callbacks")

        history = self.model.fit(
            train_dataset,
            epochs=total_epochs,
            initial_epoch=initial_epoch,
            validation_data=val_dataset,
            callbacks=callbacks,
            **kwargs,
        )
        self._extend_fit_history(history.history)
        return self

    def _ensure_model_created_from(self, x: pd.DataFrame) -> None:
        """Ensure that the underlying Keras model is created based on data shape.

        This method builds a temporary sequence dataset from ``x``, inspects one batch
        to infer (sequence_length, n_main_features) and optional condition_dimension,
        and calls ``create_model`` if ``self.model`` is not yet defined.

        Args:
            x: Example training data as a DataFrame with DatetimeIndex.
        """
        dataset, _ = self.sequence_builder.build_sliding_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=True,
        )

        # Infer shape for model creation from a single batch
        for batch in dataset.take(1):
            inputs, _ = batch
            if self.conditional_features:
                seq_main, seq_cond = inputs
                sequence_length = seq_main.shape[1]
                n_main_features = seq_main.shape[2]
                n_cond_features = seq_cond.shape[2]
                input_dim = (int(sequence_length), int(n_main_features))
                cond_dim = int(n_cond_features)
            else:
                seq_main = inputs
                sequence_length = seq_main.shape[1]
                n_main_features = seq_main.shape[2]
                input_dim = (int(sequence_length), int(n_main_features))
                cond_dim = None
            break

        if self.model is None:
            self.create_model(input_dimension=input_dim, condition_dimension=cond_dim)

    def _check_sequence_builder(self) -> None:
        """Ensure sequence_builder is set before using sequence-based methods."""
        if self.sequence_builder is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a 'sequence_builder' to be set. "
                "When training a new model, configure 'sequence_builder' under "
                "'train.autoencoder.params.sequence_builder' in the config. "
                "When loading an existing model, call 'FaultDetector.load_models' "
                "before calling fit/predict."
            )
