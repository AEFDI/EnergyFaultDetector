from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model as KerasModel

from energy_fault_detector.autoencoders.seq2seq_autoencoder import Seq2SeqAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class Seq2OneAutoencoder(Seq2SeqAutoencoder):
    """Base class for causal sequence-to-one autoencoders.

    This class trains models that map a sequence of length ``sequence_length`` to a single
    output vector (typically corresponding to the last timestep in the window).

    It reuses the same initialization and latent handling as Seq2SeqAutoencoder, but uses
    ``SequenceDatasetBuilder.build_seq2one_dataset`` instead of ``build_sliding_dataset``.
    """

    def __init__(
        self,
        sequence_builder: SequenceDatasetBuilder = None,
        conditional_features: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Initialize a sequence-to-one autoencoder.

        Args:
            sequence_builder: SequenceDatasetBuilder used to generate windowed datasets.
            conditional_features: Optional list of column names to treat as conditional features.
            **kwargs: Passed to ``Autoencoder.__init__`` via ``Seq2SeqAutoencoder``.
        """
        super().__init__(
            sequence_builder=sequence_builder,
            conditional_features=conditional_features,
            **kwargs,
        )

    # Child classes must still implement create_model, but with seq2one output shape
    def create_model(
        self,
        input_dimension: Tuple[int, int],
        condition_dimension: Optional[int] = None,
        **kwargs,
    ) -> KerasModel:
        """Create the underlying Keras model (sequence -> single timestep)."""
        raise NotImplementedError

    def _fit_internal(
        self,
        x: pd.DataFrame,
        x_val: Optional[pd.DataFrame],
        total_epochs: int,
        initial_epoch: int = 0,
        learning_rate: Optional[float] = None,
        **kwargs,
    ) -> "Seq2OneAutoencoder":
        """Internal training helper using seq2one dataset (sequence -> last timestep)."""
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
            )

        callbacks: List[Callback] = list(self.callbacks)
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

    def _predict(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        )
        self.window_timestamps_ = window_timestamps

        predictions = self.model.predict(dataset, **kwargs)  # (N, F_main)

        if self.conditional_features:
            main_columns = [c for c in x.columns if c not in self.conditional_features]
        else:
            main_columns = list(x.columns)

        target_timestamps = window_timestamps[:, -1]

        reconstruction = pd.DataFrame(
            data=predictions,
            index=target_timestamps,
            columns=main_columns,
        )
        return reconstruction
