
import logging
from typing import Union, Tuple

import numpy as np
import pandas as pd

# pylint: disable=E0401,E0611

from energy_fault_detector.core.threshold_selector import ThresholdSelector

logger = logging.getLogger('energy_fault_detector')
DataType = Union[np.ndarray, pd.DataFrame]
Array1D = Union[np.ndarray, pd.Series]


class AdaptiveThresholdSelector(ThresholdSelector):
    """ Adaptive threshold calculation based on NN regression and mutual information.

    Args:
        gamma (float): Determines the sensitivity; added to the SVR model output (the expected anomaly score).
        nn_size (int): NN hyperparameter determining the size of the hidden layer of the NN.
        nn_epochs (int): NN hyperparameter for the number of epochs during training.
        nn_learning_rate (float): NN hyperparameter for the learning rate of the optimizer during training.
        nn_batch_size (int): Number of samples per gradient update.
        smoothing_parameter (int): Parameter for score smoothing; determines the length of segments in the
            smoothing function. A value of 1 practically disables smoothing (default is 1).
        early_stopping (bool): If True, the early stopping callback will be used in the fit method.
        patience (int): Parameter for early stopping. If early stopping is used, training will end if more than
            `patience` epochs in a row have not shown an improved loss. (default is 3)
        verbose (int): Determines the amount of console output during training:
            0=silent, 1=progress bar, 2=one line per epoch.


    Configuration example:

    .. code-block:: text

        train:
          threshold_selector:
            name: AdaptiveThresholdSelector
            params:
              gamma: 0.2
              nn_size: 10
              nn_epochs: 100
              nn_learning_rate: 0.001
              nn_batch_size: 128
              smoothing_parameter: 3
              early_stopping: True
              patience: 3
              verbose: 0
    """

    def __init__(self, gamma: float = 0.2, nn_size: int = 10, nn_epochs: int = 300, nn_learning_rate: float = 0.001,
                 nn_batch_size: int = 128, smoothing_parameter: int = 1, early_stopping: bool = True, patience: int = 3,
                 validation_split: float = 0.25, verbose: int = 0):

        try:
            # lazy import to avoid tensorflow imports if not needed
            from energy_fault_detector.threshold_selectors._adaptive_threshold_model import RegressionNN
        except ImportError:
            logger.error("Could not import RegressionNN. Please install tensorflow.")
            raise

        super().__init__()
        self.gamma = gamma
        self.nn_epochs = nn_epochs
        self.validation_split = validation_split
        self.nn_model = RegressionNN(size=nn_size, learning_rate=nn_learning_rate, batch_size=nn_batch_size,
                                     early_stopping=early_stopping, patience=patience,
                                     validation_split=self.validation_split)
        self.smoothing_parameter = smoothing_parameter
        self.verbose = verbose

    # noinspection PyMethodOverriding
    def fit(self, scaled_ae_input: DataType, anomaly_score: Array1D, normal_index: pd.Series = None
            ) -> 'AdaptiveThresholdSelector':
        """Trains an NN model with the autoencoder input as input and the corresponding anomaly_score as targets.

        Args:
            scaled_ae_input (DataType): Standardized sensor data (autoencoder input).
            anomaly_score (Array1D): Anomaly scores based on deviations of the autoencoder.
            normal_index (pd.Series, optional): Labels indicating whether each sample is normal (True) or anomalous
                (False). Optional; if not provided, assumes all data represents normal behavior.

        Returns:
            AdaptiveThresholdSelector: The instance of this class after fitting the model.
        """

        if isinstance(anomaly_score, pd.Series) or isinstance(anomaly_score, pd.DataFrame):
            anomaly_score = anomaly_score.sort_index()
        else:
            anomaly_score = pd.Series(data=anomaly_score, index=normal_index.index)
        if isinstance(normal_index, pd.Series):
            normal_index = normal_index.sort_index()
        if isinstance(scaled_ae_input, pd.DataFrame):
            scaled_ae_input = scaled_ae_input.sort_index()
        else:
            scaled_ae_input = pd.DataFrame(data=scaled_ae_input, index=normal_index.index)

        if normal_index is not None:
            scaled_ae_input = scaled_ae_input[normal_index].copy()
            anomaly_score = anomaly_score[normal_index].copy()
        if self.smoothing_parameter == 1:
            self.nn_model.fit(x=scaled_ae_input.values, y=anomaly_score.values, epochs=self.nn_epochs,
                              verbose=self.verbose)
        else:
            index, smooth_score = self._smooth_anomaly_score(anomaly_score)
            self.nn_model.fit(x=scaled_ae_input.loc[index].values, y=smooth_score.values, epochs=self.nn_epochs)
        return self

    # noinspection PyMethodOverriding
    def predict(self, x: Array1D, scaled_ae_input: DataType) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts the status (normal or anomalous) of each sample based on the trained NN model.

        Args:
            x (Array1D): Anomaly scores based on deviations of the autoencoder.
            scaled_ae_input (DataType): Standardized sensor data (autoencoder input).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing a boolean array indicating the predicted status of each
                sample and the corresponding adaptive thresholds.
        """
        self.threshold = (self.nn_model.predict(x=scaled_ae_input, verbose=self.verbose) + self.gamma).reshape(-1)
        return x > self.threshold, self.threshold

    def _smooth_anomaly_score(self, anomaly_score: Union[pd.DataFrame, pd.Series]) -> tuple:
        """ This function divides anomaly_score into segments of length self.smoothing_parameter. It then computes
        the average for each segment and returns the timeseries of averages where the first timestamp of each segment is
        used to represent the segment.

        Args:
            anomaly_score (Union[pd.DataFrame, pd.Series]): A time series of anomaly scores.

        returns:
            index: pandas DataFrame index containing every first timestamp of each segment.
            mean_anomaly_score: pandas Series containing the average of each segment with an integer index.
        """
        index = anomaly_score.index[::self.smoothing_parameter].copy()
        mean_anomaly_score = anomaly_score.groupby(np.arange(len(anomaly_score)) // self.smoothing_parameter).mean()
        return index, mean_anomaly_score
