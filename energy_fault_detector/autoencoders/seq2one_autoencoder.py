"""Seq2One (causal half-autoencoder) base class."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from energy_fault_detector.autoencoders.sequence_autoencoder import SequenceAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class Seq2OneAutoencoder(SequenceAutoencoder):
    """Sequence-to-one autoencoder: maps a window to a single output (last timestep).

    Subclasses must implement ``create_model``.

    This class trains models that map a sequence of length ``sequence_length`` to a single
    output vector (typically corresponding to the last timestep in the window).

    It reuses the same initialization and latent handling as Seq2SeqAutoencoder, but uses
    ``SequenceDatasetBuilder.build_seq2one_dataset`` instead of ``build_sliding_dataset``.
    """

    def __init__(self, sequence_builder: Optional[SequenceDatasetBuilder] = None, **ae_kwargs):
        super().__init__(sequence_builder=sequence_builder, **ae_kwargs)

    def _build_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int,
        conditional_features: Optional[List[str]],
        shuffle: bool,
        predict_mode: bool = False,
    ) -> Tuple[tf.data.Dataset, np.ndarray]:
        """Build seq2one dataset (input: full window, target: last timestep)."""
        return self.sequence_builder.build_seq2one_dataset(
            df=df,
            batch_size=batch_size,
            conditional_features=conditional_features,
            shuffle=shuffle,
            predict_mode=predict_mode,
        )

    def _assemble_predictions(
        self,
        predictions: np.ndarray,
        window_timestamps: np.ndarray,
        columns: List[str],
        ref_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """One prediction per window → indexed by last timestamp."""
        target_timestamps = window_timestamps[:, -1]
        target_index = self._timestamps_to_index(target_timestamps, ref_index)
        return pd.DataFrame(
            data=predictions,
            index=target_index,
            columns=columns,
        )
