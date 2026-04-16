from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model as KerasModel

from ..data_splitting.sequence_dataset import SequenceDatasetBuilder
from ..core.autoencoder import Autoencoder


class Seq2OneAutoencoder(Autoencoder):
    """Base class for causal sequence-to-one (half) autoencoders.

    This class trains models that map a sequence of length ``sequence_length`` to a single
    output vector (typically corresponding to the last timestep in the window).

    It reuses the same initialization and latent handling as Seq2SeqAutoencoder, but uses
    ``SequenceDatasetBuilder.build_seq2one_dataset`` instead of ``build_sliding_dataset``.
    """

    def __init__(
        self,
        sequence_builder: SequenceDatasetBuilder = None,
        **ae_kwargs
    ) -> None:
        """Initialize a sequence-to-one autoencoder.

        Args:
            sequence_builder: SequenceDatasetBuilder used to generate windowed datasets.
            **ae_kwargs: Passed to ``Autoencoder.__init__``.
        """

        super().__init__(**ae_kwargs)
        self.sequence_builder = sequence_builder
        self.is_sequential: bool = True
        self.is_seq2one: bool = True

    # Child classes must still implement create_model, but with seq2one output shape
    def create_model(
        self,
        input_dimension: Tuple[int, int],
        condition_dimension: Optional[int] = None,
        **kwargs,
    ) -> KerasModel:
        """Create the underlying Keras model.

        Subclasses must implement this method and set ``self.model`` (and optionally ``self.encoder``).

        Args:
            input_dimension: Tuple ``(sequence_length, n_main_features)``.
            condition_dimension: Number of conditional features, or ``None`` if no conditions are used.
            **kwargs: Additional keyword arguments for model creation.

        Returns:
            The created Keras model.
        """
        raise NotImplementedError

    def fit(
        self,
        x: pd.DataFrame,
        x_val: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Seq2OneAutoencoder:
        """Fit the sequence autoencoder on time-series data.

        Args:
            x: Training data as a DataFrame with DatetimeIndex.
            x_val: Optional validation data as a DataFrame with DatetimeIndex.
            **kwargs: Additional keyword arguments passed to ``model.fit``.

        Returns:
            The fitted ``Seq2SeqAutoencoder`` instance.
        """
        self._check_sequence_builder()
        self._ensure_model_created_from(x)

        kwargs.setdefault("verbose", self.verbose)
        return self._fit_internal(
            x=x,
            x_val=x_val,
            total_epochs=self.epochs,
            initial_epoch=0,
            learning_rate=None,
            **kwargs,
        )

    def tune(
        self,
        x: pd.DataFrame,
        x_val: Optional[pd.DataFrame] = None,
        learning_rate: float = 0.001,
        tune_epochs: int = 5,
        **kwargs,
    ) -> Seq2OneAutoencoder:
        """Fine-tune the sequence autoencoder on additional data.

        This extends training for ``tune_epochs`` epochs, optionally with a new learning rate.

        Args:
            x: Training data as a DataFrame with DatetimeIndex.
            x_val: Optional validation data as a DataFrame with DatetimeIndex.
            learning_rate: Learning rate to use during tuning.
            tune_epochs: Number of additional epochs to run.
            **kwargs: Additional keyword arguments passed to ``model.fit``.

        Returns:
            The tuned ``Seq2SeqAutoencoder`` instance.
        """
        self._check_sequence_builder()

        kwargs.setdefault("verbose", self.verbose)
        return self._fit_internal(
            x=x,
            x_val=x_val,
            total_epochs=self.epochs + tune_epochs,
            initial_epoch=self.epochs,
            learning_rate=learning_rate,
            **kwargs,
        )

    def encode(self, x: pd.DataFrame, conditions: pd.DataFrame = None) -> np.ndarray:
        """Encode input time series into the latent space.
        One latent vector per window, associated with the last timestamp of each window.

        Args:
            x: Input data as a DataFrame with DatetimeIndex.
            conditions: Optional DataFrame with conditional features. Currently not used. TODO: needed?

        Returns:
            NumPy array with latent representations for each sequence window.
        """
        self._check_sequence_builder()
        dataset, _ = self.sequence_builder.build_seq2one_dataset(
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

    def create_model(
        self,
        input_dimension: Tuple[int, int],
        condition_dimension: Optional[int] = None,
        **kwargs,
    ) -> KerasModel:
        """Create the underlying Keras model.

        Subclasses must implement this method and set ``self.model`` (and optionally ``self.encoder``).

        Args:
            input_dimension: Tuple ``(sequence_length, n_main_features)``.
            condition_dimension: Number of conditional features, or ``None`` if no conditions are used.
            **kwargs: Additional keyword arguments for model creation.

        Returns:
            The created Keras model.
        """
        raise NotImplementedError

    def fit(
        self,
        x: pd.DataFrame,
        x_val: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Seq2OneAutoencoder:
        """Fit the sequence autoencoder on time-series data.

        Args:
            x: Training data as a DataFrame with DatetimeIndex.
            x_val: Optional validation data as a DataFrame with DatetimeIndex.
            **kwargs: Additional keyword arguments passed to ``model.fit``.

        Returns:
            The fitted ``Seq2SeqAutoencoder`` instance.
        """
        self._check_sequence_builder()
        self._ensure_model_created_from(x)

        kwargs.setdefault("verbose", self.verbose)
        return self._fit_internal(
            x=x,
            x_val=x_val,
            total_epochs=self.epochs,
            initial_epoch=0,
            learning_rate=None,
            **kwargs,
        )

    def tune(
        self,
        x: pd.DataFrame,
        x_val: Optional[pd.DataFrame] = None,
        learning_rate: float = 0.001,
        tune_epochs: int = 5,
        **kwargs,
    ) -> Seq2OneAutoencoder:
        """Fine-tune the sequence autoencoder on additional data.

        This extends training for ``tune_epochs`` epochs, optionally with a new learning rate.

        Args:
            x: Training data as a DataFrame with DatetimeIndex.
            x_val: Optional validation data as a DataFrame with DatetimeIndex.
            learning_rate: Learning rate to use during tuning.
            tune_epochs: Number of additional epochs to run.
            **kwargs: Additional keyword arguments passed to ``model.fit``.

        Returns:
            The tuned ``Seq2SeqAutoencoder`` instance.
        """
        self._check_sequence_builder()

        kwargs.setdefault("verbose", self.verbose)
        return self._fit_internal(
            x=x,
            x_val=x_val,
            total_epochs=self.epochs + tune_epochs,
            initial_epoch=self.epochs,
            learning_rate=learning_rate,
            **kwargs,
        )

    def encode(self, x: pd.DataFrame, conditions: pd.DataFrame = None) -> np.ndarray:
        """Encode input time series into the latent space.
        One latent vector per window, associated with the last timestamp of each window.

        Args:
            x: Input data as a DataFrame with DatetimeIndex.
            conditions: Optional DataFrame with conditional features. Currently not used. TODO: needed?

        Returns:
            NumPy array with latent representations for each sequence window.
        """
        self._check_sequence_builder()
        dataset, _ = self.sequence_builder.build_seq2one_dataset(
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
    ) -> Seq2OneAutoencoder:
        """Internal training helper using seq2one dataset (sequence -> last timestep).

        Args:
            x: Training data as a DataFrame with DatetimeIndex.
            x_val: Optional validation data as a DataFrame with DatetimeIndex.
            total_epochs: Total number of epochs to run.
            initial_epoch: Epoch number to start from (used for tuning).
            learning_rate: Optional new learning rate to use for training.
            **kwargs: Additional keyword arguments passed to ``model.fit``.

        Returns:
            The trained (or tuned) ``Seq2OneAutoencoder`` instance.
        """
        self._check_sequence_builder()
        if learning_rate is not None:
            self.compile_model(new_learning_rate=learning_rate)
        else:
            self.compile_model()

        train_dataset, _ = self.sequence_builder.build_seq2one_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=True,
        )

        val_dataset = None
        if x_val is not None:
            val_dataset, _ = self.sequence_builder.build_seq2one_dataset(
                df=x_val,
                batch_size=self.batch_size,
                conditional_features=self.conditional_features,
                shuffle=False,
                predict_mode=True,
            )

        callbacks: List[Callback] = list(self.callbacks)
        if "callbacks" in kwargs:
            callbacks += kwargs["callbacks"]
            kwargs.pop("callbacks")

        kwargs.setdefault("verbose", self.verbose)
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

    def _predict(self, x: pd.DataFrame, return_conditions: bool = False, **kwargs) -> pd.DataFrame:
        """Predict the last timestep per window and return a 2D DataFrame.

        For each window, the model outputs a single vector (reconstruction for the last timestep).
        The resulting DataFrame has one row per window, indexed by the last timestamp of that window.
        """
        self._check_sequence_builder()
        dataset, window_timestamps = self.sequence_builder.build_seq2one_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=False,
            predict_mode=True,
        )
        self.window_timestamps_ = window_timestamps

        kwargs.setdefault("verbose", self.verbose)
        predictions = self.model.predict(dataset, **kwargs)  # (N, F_main)

        if self.conditional_features:
            main_columns = [c for c in x.columns if c not in self.conditional_features]
        else:
            main_columns = list(x.columns)

        target_timestamps = window_timestamps[:, -1]
        target_index = self._timestamps_to_index(target_timestamps, x.index)

        reconstruction = pd.DataFrame(
            data=predictions,
            index=target_index,
            columns=main_columns,
        )
        if return_conditions and self.conditional_features:
            cond_at_targets = x[self.conditional_features].loc[reconstruction.index]
            return cond_at_targets.join(reconstruction)

        return reconstruction

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
                "When loading an existing model, call 'FaultDetector.load_models' before calling fit/predict."
            )

    @staticmethod
    def _timestamps_to_index(target_timestamps: np.ndarray, ref_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Convert numpy datetime64 timestamps to a DatetimeIndex compatible with ref_index.

        - If ref_index is tz-naive: return tz-naive index.
        - If ref_index is tz-aware: interpret target_timestamps as UTC instants
          (as produced by ref_index.values) and convert to ref_index.tz.
        """

        idx = pd.DatetimeIndex(target_timestamps)
        if isinstance(ref_index, pd.DatetimeIndex) and ref_index.tz is not None and idx.tz is None:
            idx = idx.tz_localize("UTC").tz_convert(ref_index.tz)
        return idx
