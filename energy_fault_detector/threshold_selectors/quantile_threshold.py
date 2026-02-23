
from typing import Union

import numpy as np
import pandas as pd

from energy_fault_detector.core.threshold_selector import ThresholdSelector

Array1D = Union[np.ndarray, pd.Series]


class QuantileThresholdSelector(ThresholdSelector):
    """Find a threshold by defining a specified quantile of the given anomaly scores.

    Args:
        quantile (float): The quantile of the scores to be computed. Defaults to 0.99.

    Attributes:
        threshold (float): Scores above the threshold are classified as anomalies, while scores below are classified as
            normal.

    Example Configuration:

    .. code-block:: text

        train:
          threshold_selector:
            name: QuantileThresholdSelector
            params:
              quantile: 0.99
    """

    def __init__(self, quantile: float = 0.99):
        super().__init__()

        self.quantile = quantile

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: Array1D, y: pd.Series = None) -> 'QuantileThresholdSelector':
        """
        Sets the threshold to the chosen quantile of the provided anomaly scores.

        Args:
            x (Array1D): Array containing calculated anomaly scores.
            y (pd.Series, optional): Labels indicating whether each sample is normal (True) or anomalous (False).
                Optional; if not provided, it is assumed that all data represents normal behavior.

        Returns:
            QuantileThresholdSelector: The instance of this class after setting the threshold.

        Notes:
            UserWarning: If a suitable threshold cannot be found, the threshold is set to the maximum score.
        """

        if isinstance(x, pd.Series):
            x = x.sort_index().values
        if isinstance(y, pd.Series):
            y = y.sort_index().values

        if y is not None:
            x_ = x[y]
        else:
            x_ = x

        self.threshold = float(np.quantile(x_, self.quantile))

        if self.threshold is None:
            import warnings
            warnings.warn('Could not find suitable threshold, `threshold` is set to max score.', UserWarning)
            self.threshold = float(np.sort(x)[-1])
        return self
