"""Base class for all sequence (windowed) autoencoders.

Note: not part of .core since it depends on SequenceDatasetBuilder.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from energy_fault_detector.core.autoencoder import Autoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class SequenceAutoencoder(Autoencoder):
    """Abstract base for sequence autoencoders (seq2one and seq2seq).

    Subclasses must implement:
        - ``create_model``
        - ``_build_dataset``
        - ``_assemble_predictions``
    """

    def __init__(
        self,
        sequence_builder: Optional[SequenceDatasetBuilder] = None,
        **ae_kwargs,
    ) -> None:
        super().__init__(**ae_kwargs)
        self.sequence_builder = sequence_builder
        self.window_timestamps_: Optional[np.ndarray] = None

    @abstractmethod
    def _build_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int,
        conditional_features: Optional[List[str]],
        shuffle: bool,
        predict_mode: bool = False,
    ) -> Tuple[tf.data.Dataset, np.ndarray]:
        """Build the appropriate tf.data.Dataset.

        Returns:
            Tuple of (dataset, window_timestamps).
        """

    @abstractmethod
    def _assemble_predictions(
        self,
        predictions: np.ndarray,
        window_timestamps: np.ndarray,
        columns: List[str],
        ref_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Convert raw model output into an aligned DataFrame."""

    def fit(
        self,
        x: pd.DataFrame,
        x_val: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> SequenceAutoencoder:
        """Fit the sequence autoencoder on time-series data.

        Args:
            x: Training data (DataFrame with DatetimeIndex).
            x_val: Optional validation data.
            **kwargs: Passed to ``model.fit``.

        Returns:
            The fitted instance.
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
    ) -> SequenceAutoencoder:
        """Fine-tune the model for additional epochs.

        Args:
            x: Training data.
            x_val: Optional validation data.
            learning_rate: Learning rate for tuning.
            tune_epochs: Number of additional epochs.
            **kwargs: Passed to ``model.fit``.

        Returns:
            The tuned instance.
        """
        self._check_sequence_builder()
        kwargs.setdefault("verbose", self.verbose)
        return self._fit_internal(
            x=x,
            x_val=x_val,
            total_epochs=self.epochs + tune_epochs,
            initial_epoch=self.epochs_completed,
            learning_rate=learning_rate,
            **kwargs,
        )

    def tune_decoder(
        self,
        x: pd.DataFrame,
        x_val: Optional[pd.DataFrame] = None,
        learning_rate: float = 0.001,
        tune_epochs: int = 5,
        **kwargs,
    ) -> SequenceAutoencoder:
        """Tune decoder only — encoder weights are frozen."""
        if self.encoder is None:
            raise ValueError(
                "Encoder was not created. Ensure `create_model` sets `self.encoder`."
            )
        self.encoder.trainable = False
        self.tune(x=x, x_val=x_val, learning_rate=learning_rate, tune_epochs=tune_epochs, **kwargs)
        self.encoder.trainable = True
        return self

    def encode(self, x: pd.DataFrame, conditions: pd.DataFrame = None, **kwargs) -> np.ndarray:
        """Encode input time series into the latent space.

        Args:
            x: Input data (DataFrame with DatetimeIndex).
            conditions: Unused (kept for API compatibility).

        Returns:
            Latent representations as NumPy array.
        """
        self._check_sequence_builder()
        dataset, _ = self._build_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=False,
            predict_mode=True,
        )
        kwargs.setdefault("verbose", self.verbose)
        return self.encoder.predict(dataset, **kwargs)

    def predict(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Predict/reconstruct input data.

        Args:
            x: Input data (DataFrame with DatetimeIndex).

        Returns:
            Reconstructed DataFrame.
        """
        if not self._is_fitted():
            raise ValueError(f"{self.__class__.__name__} must be fitted first!")
        kwargs.setdefault("verbose", self.verbose)
        return self._predict(x, **kwargs)

    def get_reconstruction_error(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute reconstruction error (predicted - actual) for main features.

        Args:
            x: Input data (DataFrame with DatetimeIndex).

        Returns:
            DataFrame with reconstruction errors.
        """
        reconstruction = self._predict(x, **kwargs)
        main_columns = self._get_main_columns(x)
        x_main = x[main_columns].loc[reconstruction.index]
        return reconstruction - x_main

    def _fit_internal(
        self,
        x: pd.DataFrame,
        x_val: Optional[pd.DataFrame],
        total_epochs: int,
        initial_epoch: int = 0,
        learning_rate: Optional[float] = None,
        **kwargs,
    ) -> SequenceAutoencoder:
        """Unified training loop for all sequence autoencoders."""
        if learning_rate is not None:
            self.compile_model(new_learning_rate=learning_rate)
        else:
            self.compile_model()

        train_shuffle = True
        if hasattr(self, 'stateful'):
            if self.stateful:
                train_shuffle = False

        train_dataset, _ = self._build_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=train_shuffle,
        )

        # Apply noise to training data
        if self.noise > 0:
            train_dataset = train_dataset.map(self._add_noise_to_batch)

        val_dataset = None
        if x_val is not None:
            val_dataset, _ = self._build_dataset(
                df=x_val,
                batch_size=self.batch_size,
                conditional_features=self.conditional_features,
                shuffle=False,
                predict_mode=True,
            )

        callbacks: List[Callback] = list(self.callbacks)
        if "callbacks" in kwargs:
            callbacks += kwargs.pop("callbacks")

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

    def _predict(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run model prediction and reassemble into a DataFrame."""
        self._check_sequence_builder()
        dataset, window_timestamps = self._build_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=False,
            predict_mode=True,
        )
        self.window_timestamps_ = window_timestamps

        kwargs.setdefault("verbose", self.verbose)
        predictions = self.model.predict(dataset, **kwargs)

        main_columns = self._get_main_columns(x)
        return self._assemble_predictions(
            predictions=predictions,
            window_timestamps=window_timestamps,
            columns=main_columns,
            ref_index=x.index,
        )

    def _ensure_model_created_from(self, x: pd.DataFrame) -> None:
        """Infer input shape and create model if not yet built."""
        if self.model is not None:
            return

        dataset, _ = self._build_dataset(
            df=x,
            batch_size=self.batch_size,
            conditional_features=self.conditional_features,
            shuffle=False,
        )

        for batch in dataset.take(1):
            inputs, _ = batch
            if self.conditional_features:
                seq_main, seq_cond = inputs
                input_dim = (int(seq_main.shape[1]), int(seq_main.shape[2]))
                cond_dim = int(seq_cond.shape[2])
            else:
                seq_main = inputs
                input_dim = (int(seq_main.shape[1]), int(seq_main.shape[2]))
                cond_dim = None
            break

        self.create_model(input_dimension=input_dim, condition_dimension=cond_dim)

    def _check_sequence_builder(self) -> None:
        """Validate that sequence_builder is available."""
        if self.sequence_builder is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a 'sequence_builder'. "
                "Configure 'train.autoencoder.params.sequence_builder' in the config, "
                "or call 'FaultDetector.load_models' before fit/predict."
            )

    def _get_main_columns(self, x: pd.DataFrame) -> List[str]:
        """Get non-conditional column names."""
        if self.conditional_features:
            return [c for c in x.columns if c not in self.conditional_features]
        return list(x.columns)

    def _add_noise_to_batch(self, inputs, targets):
        """Map function to add Gaussian noise to training inputs."""
        if isinstance(inputs, tuple):
            # (main_seq, cond_seq) — only add noise to main
            main, cond = inputs
            noisy_main = main + tf.random.normal(tf.shape(main), stddev=self.noise)
            return (noisy_main, cond), targets
        else:
            noisy = inputs + tf.random.normal(tf.shape(inputs), stddev=self.noise)
            return noisy, targets

    @staticmethod
    def _timestamps_to_index(
        target_timestamps: np.ndarray, ref_index: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """Convert numpy datetime64 array to DatetimeIndex matching ref_index timezone."""
        idx = pd.DatetimeIndex(target_timestamps)
        if (
            isinstance(ref_index, pd.DatetimeIndex)
            and ref_index.tz is not None
            and idx.tz is None
        ):
            idx = idx.tz_localize("UTC").tz_convert(ref_index.tz)
        return idx
