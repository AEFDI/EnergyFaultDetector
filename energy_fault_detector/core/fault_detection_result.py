
import os
from typing import Optional, List
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class FaultDetectionResult:
    """Class to encapsulate results from the fault detection process."""

    predicted_anomalies: pd.DataFrame
    """DataFrame with a column 'anomaly' (bool)."""

    reconstruction: pd.DataFrame
    """DataFrame with reconstruction of the sensor data with timestamp as index."""

    recon_error: pd.DataFrame
    """DataFrame with reconstruction errors."""

    anomaly_score: pd.DataFrame
    """DataFrame with anomaly scores for each timestamp."""

    bias_data: Optional[pd.DataFrame] = None
    """DataFrame with ARCANA results with timestamp as index. None if ARCANA was not run."""

    arcana_losses: Optional[pd.DataFrame] = None
    """DataFrame containing recorded values for all losses in ARCANA. None if ARCANA was not run."""

    tracked_bias: Optional[List[pd.DataFrame]] = None
    """List of DataFrames containing the ARCANA bias every 50th iteration. None if ARCANA was not run."""

    def save(self, directory: str, **kwargs) -> None:
        """Saves the results to CSV files in the specified directory.

        Args:
            directory (str): The directory where the CSV files will be saved.
            kwargs: other keywords args for `pd.DataFrame.to_csv`
        """
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Save each DataFrame as a CSV file
        self.predicted_anomalies.to_csv(os.path.join(directory, 'predicted_anomalies.csv'), **kwargs)
        self.reconstruction.to_csv(os.path.join(directory, 'reconstruction.csv'), **kwargs)
        self.recon_error.to_csv(os.path.join(directory, 'reconstruction_errors.csv'), **kwargs)
        self.anomaly_score.to_csv(os.path.join(directory, 'anomaly_scores.csv'), **kwargs)

        if self.bias_data is not None:
            self.bias_data.to_csv(os.path.join(directory, 'bias_data.csv'), **kwargs)

        if self.arcana_losses is not None:
            self.arcana_losses.to_csv(os.path.join(directory, 'arcana_losses.csv'), **kwargs)

        if self.tracked_bias is not None and len(self.tracked_bias) > 0:
            for idx, bias_df in enumerate(self.tracked_bias):
                bias_df.to_csv(os.path.join(directory, f'tracked_bias_{idx}.csv'), **kwargs)


@dataclass
class ModelMetadata:
    """Class to encapsulate metadata about the FaultDetector model."""

    model_date: str
    model_path: str
    train_recon_error: np.ndarray
    val_recon_error: Optional[np.ndarray] = None
