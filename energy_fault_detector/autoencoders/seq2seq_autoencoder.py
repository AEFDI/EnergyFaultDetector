"""Seq2Seq autoencoder base class."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from energy_fault_detector.autoencoders.sequence_autoencoder import SequenceAutoencoder
from energy_fault_detector.autoencoders._sequence_utils import sequences_to_dataframe
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class Seq2SeqAutoencoder(SequenceAutoencoder):
    """Sequence-to-sequence autoencoder: reconstructs the full input window.

    Subclasses must implement ``create_model``.

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
        """Build sliding-window dataset (input = target = full window)."""
        return self.sequence_builder.build_sliding_dataset(
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
        """Flatten overlapping windows into a single DataFrame (averaging overlaps)."""
        return sequences_to_dataframe(
            sequences=predictions,
            timestamps=window_timestamps,
            columns=columns,
        )
