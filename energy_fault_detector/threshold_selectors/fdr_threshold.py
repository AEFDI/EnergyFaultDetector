
import logging
from typing import Union

import numpy as np
import pandas as pd

from energy_fault_detector.core.threshold_selector import ThresholdSelector

logger = logging.getLogger('energy_fault_detector')
Array1D = Union[np.ndarray, pd.Series]


class FDRSelector(ThresholdSelector):
    """Find a threshold given a target false discovery rate (FDR).

    Args:
        target_false_discovery_rate (float): The target FDR to fit the threshold to. Defaults to 0.2.

    Attributes:
        threshold (float): Scores above the threshold are classified as anomalies, while scores below are classified as
            normal.
        actual_false_discovery_rate_ (float): The actual FDR (the nearest threshold to the target) after fitting.

    Example Configuration:

    .. code-block:: yaml

        train:
          threshold_selector:
            name: FDRSelector
            params:
              target_false_discovery_rate: 0.2
    """

    def __init__(self, target_false_discovery_rate: float = 0.2):
        super().__init__()

        self.target_false_discovery_rate: float = target_false_discovery_rate

    def fit(self, x: Array1D, y: pd.Series = None) -> 'FDRSelector':
        """Finds a threshold given the specified false discovery rate.

        Args:
            x (Array1D): Array with calculated anomaly scores.
            y (pd.Series, optional): Labels indicating whether each sample is normal (True) or anomalous (False).
                Required for FDR threshold calculation.

        Returns:
            FDRSelector: The instance of this class after fitting the threshold.
        """

        if isinstance(x, pd.Series):
            x = x.sort_index().values
            y = y.sort_index().values

        anomalies = ~y  # anomaly := inverse of normal indices

        self._fdr_selector(x, anomalies)
        return self

    # pylint: disable=attribute-defined-outside-init
    def _fdr_selector(self, scores: np.ndarray, anomalies: np.ndarray):
        """Sets the threshold based on the specified false discovery rate.
        This method uses the unique sorted scores as possible thresholds and loops through
        them to find the correct threshold. The threshold is determined when the false
        discovery rate for a certain threshold exceeds the desired target.

        Note: This function may take a long time to execute if the list of possible thresholds
            is long and the target false discovery rate is low, as the list is sorted from low to high.

        Args:
            scores (np.ndarray): Scores to compare against `self.threshold`.
            anomalies (np.ndarray): Array indicating whether each sample is an actual anomaly.
        """

        if all(anomalies) or all(~anomalies):
            logger.warning('Cannot set suitable threshold, all values are either anomalous or normal,'
                           ' `threshold` is set to max score.')
            self.threshold = float(np.unique(np.sort(scores))[-1])
            return

        possible_thresholds = np.unique(np.sort(scores))

        for threshold in possible_thresholds:
            # detected anomalies:
            detected_anomaly = scores > threshold
            # false positive: detected, not an anomaly
            fp = np.sum(detected_anomaly * (~anomalies))
            # true positive: detected and is an anomaly
            tp = np.sum(detected_anomaly * anomalies)
            # False-Positive-Rate
            # FPR = FP/(TN+FP)
            if (fp+tp) == 0:
                fdr = 1.0
            else:
                fdr = fp/(fp+tp)
            if fdr >= self.target_false_discovery_rate:
                self.actual_false_discovery_rate_ = fdr
                self.threshold = float(threshold)
            else:
                # threshold found, stop
                break

        if self.threshold is None:
            logger.warning('Could not find suitable threshold, `threshold` is set to max score.')
            self.threshold = float(possible_thresholds[-1])
