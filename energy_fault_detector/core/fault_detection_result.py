
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from ..utils.analysis import calculate_criticality


@dataclass
class FaultDetectionResult:
    """Class to encapsulate results from the fault detection process."""

    predicted_anomalies: pd.Series
    """Series with a predicted anomalies (bool)."""

    reconstruction: pd.DataFrame
    """DataFrame with reconstruction of the input data."""

    recon_error: pd.DataFrame
    """DataFrame with reconstruction errors."""

    anomaly_score: pd.Series
    """Series with predicted anomaly scores."""

    bias_data: Optional[pd.DataFrame] = None
    """DataFrame with ARCANA results (ARCANA bias). None if ARCANA was not run."""

    arcana_losses: Optional[pd.DataFrame] = None
    """DataFrame containing recorded values for all losses in ARCANA. None if ARCANA was not run.
       Empty if losses were not tracked."""

    tracked_bias: Optional[List[pd.DataFrame]] = None
    """List of DataFrames containing the ARCANA bias every 50th iteration. None if ARCANA was not run.
       Empty if bias was not tracked."""

    def criticality(self, normal_idx: pd.Series | None, init_criticality: int = 0, max_criticality: int = 1000
                    ) -> pd.Series:
        """Criticality based on the predicted anomalies.

        Args:
            normal_idx (pd.Series, optional): A pandas Series with boolean values indicating normal operation, indexed
                by timestamp. Ignored if None.
            init_criticality (int, optional): The initial criticality value. Defaults to 0.
            max_criticality (int, optional): The maximum criticality value. Defaults to 1000.

        """
        return calculate_criticality(self.predicted_anomalies, normal_idx, init_criticality, max_criticality)

    def save(self, directory: str | Path, **kwargs) -> None:
        """Saves the results to CSV files in the specified directory.

        Args:
            directory (str): The directory where the CSV files will be saved.
            kwargs: other keywords args for `pd.DataFrame.to_csv` (i.e. sep=',')
        """
        # Ensure the directory exists
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # Save each DataFrame as a CSV file
        self.predicted_anomalies.to_csv(directory / 'predicted_anomalies.csv', **kwargs)
        self.reconstruction.to_csv(directory / 'reconstruction.csv', **kwargs)
        self.recon_error.to_csv(directory / 'reconstruction_errors.csv', **kwargs)
        self.anomaly_score.to_csv(directory / 'anomaly_scores.csv', **kwargs)

        if self.bias_data is not None:
            self.bias_data.to_csv(directory / 'bias_data.csv', **kwargs)

        if self.arcana_losses is not None:
            self.arcana_losses.to_csv(directory / 'arcana_losses.csv', **kwargs)

        if self.tracked_bias is not None and len(self.tracked_bias) > 0:
            for idx, bias_df in enumerate(self.tracked_bias):
                bias_df.to_csv(directory / f'tracked_bias_{idx}.csv', **kwargs)

    @classmethod
    def load(cls, directory: str | Path, **kwargs) -> "FaultDetectionResult":
        """Loads the results from CSV files in the specified directory.

        Args:
            directory (str | Path): The directory where the CSV files are stored.
            kwargs: other keywords args for `pd.read_csv` (e.g., sep=',')

        Returns:
            FaultDetectionResult: The loaded result object.
        """
        directory = Path(directory)

        # Default pandas loading arguments to ensure indices are restored correctly
        params = {'index_col': 0, 'parse_dates': True}
        params.update(kwargs)

        # Load mandatory fields
        predicted_anomalies = pd.read_csv(directory / 'predicted_anomalies.csv', **params).iloc[:, 0]
        # Ensure predicted_anomalies is explicitly a Series and boolean
        predicted_anomalies = predicted_anomalies.astype(bool)

        reconstruction = pd.read_csv(directory / 'reconstruction.csv', **params)
        recon_error = pd.read_csv(directory / 'reconstruction_errors.csv', **params)
        anomaly_score = pd.read_csv(directory / 'anomaly_scores.csv', **params).iloc[:, 0]

        # Load optional fields if they exist
        bias_data = None
        if (directory / 'bias_data.csv').exists():
            bias_data = pd.read_csv(directory / 'bias_data.csv', **params)

        arcana_losses = None
        if (directory / 'arcana_losses.csv').exists():
            arcana_losses = pd.read_csv(directory / 'arcana_losses.csv', **params)

        tracked_bias = None
        tracked_files = sorted(directory.glob('tracked_bias_*.csv'))
        if tracked_files:
            tracked_bias = [pd.read_csv(f, **params) for f in tracked_files]

        return cls(
            predicted_anomalies=predicted_anomalies,
            reconstruction=reconstruction,
            recon_error=recon_error,
            anomaly_score=anomaly_score,
            bias_data=bias_data,
            arcana_losses=arcana_losses,
            tracked_bias=tracked_bias
        )


@dataclass
class ModelMetadata:
    """Class to encapsulate metadata about the FaultDetector model."""

    model_date: str
    model_path: str | Path
    train_recon_error: np.ndarray
    val_recon_error: Optional[np.ndarray] = None
