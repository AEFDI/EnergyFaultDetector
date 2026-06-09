
import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from energy_fault_detector.core.threshold_selector import ThresholdSelector

logger = logging.getLogger('energy_fault_detector')

Array1D = Union[np.ndarray, pd.Series]


class FbetaSelector(ThresholdSelector):
    """Find a threshold via searching for the optimal fbeta score amongst all options.

    Args:
        beta (float): beta weights the importance between precision and recall in the fbeta score. For example
            beta=2.0 means recall is twice as important as precision and beta=0.5 means the opposite.
            Defaults to 0.5.
        eps (float): small number, to ensure the optimal threshold is just below the anomaly score resulting in the optimal
            fbeta score.
        quantile (float): optional parameter that can introduce score smoothing. This parameter specifies a quantile which
            is used during self.fit to neglect all normal anomaly-scores that are greater than the quantile of all
            normal anomaly-scores. It must be a float between 0 and 1, where 1 practically disables the score smoothing.

    Attributes:
        threshold (float): scores above the threshold is classified as anomaly, below is classified as normal.

    Configuration example:

    .. code-block:: text

        train:
          threshold_selector:
            name: FbetaSelector
            params:
              beta: 0.5
              eps: 0.000001
              quantile: 1.

    """

    def __init__(self, beta: float = 0.5, eps: float = 1e-6, quantile: float = 1.):
        super().__init__()

        self.beta = beta
        self.eps = eps
        if quantile > 1 or quantile < 0:
            msg = f'Quantile has to be a float between 0 and 1, not {quantile}. Score smoothing will be disabled '
            msg += f'now by setting quantile=1.'
            logger.warning(msg)
            self.quantile = 1.
        self.quantile = quantile
        self.threshold = None

    def fit(self, x: Array1D, y: pd.Series = None) -> 'FbetaSelector':
        """Selects a threshold based on fbeta-scores.

        Args:
            x (Array1D): numpy array or pandas Series with calculated anomaly scores
            y (pd.Series): series of labels indicating whether sample is normal (True) or anomalous (False)
                Required for F_beta Threshold!
        """

        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.sort_index().values
        if isinstance(y, pd.Series):
            y = y.sort_index().values

        anomalies = ~y  # anomaly := inverse of normal indices
        selection = self.mark_normal_outliers(anomaly_score=x, normal_index=y)
        self._f_beta_selector(x[selection], anomalies=anomalies[selection])
        return self

    # pylint: disable=attribute-defined-outside-init
    def _f_beta_selector(self, scores: np.ndarray, anomalies: np.ndarray) -> None:
        """Sets the threshold such that the F-beta score is maximized for given scores (probabilities)
        and true anomalies.

        If for some reason the precision/recall curve could not be calculated the threshold
        is set to the maximum score.

        Args:
            scores (np.ndarray): scores to compare to ``self.threshold``.
            anomalies (np.ndarray): whether the sample is an actual anomaly.
        """

        if all(anomalies) or all(~anomalies):
            logger.warning('Cannot set suitable threshold, all values are either anomalous or normal,'
                           ' `threshold` is set to max score.')
            self.threshold = float(np.unique(np.sort(scores))[-1])
            return

        try:
            precision, recall, thresholds = precision_recall_curve(y_true=anomalies, y_score=scores)
            f_scores = np.zeros_like(precision)  # Initialize f_scores to zero
            # Calculate F-scores, avoiding division by zero
            for i in range(len(precision)):
                if precision[i] == 0 or recall[i] == 0:
                    f_scores[i] = 0  # Set F-score to 0 if precision or recall are 0 (ill-defined)
                else:
                    f_scores[i] = (
                            (1 + self.beta ** 2) * precision[i] * recall[i]
                            / (self.beta ** 2 * precision[i] + recall[i])
                    )

            max_score_idx = np.argmax(np.nan_to_num(f_scores))
            self.threshold = float(thresholds[max_score_idx] - self.eps)
            logger.info('Best F_%s score: %.3f, threshold: %.2f', self.beta, f_scores[max_score_idx], self.threshold)
        except Exception as e:
            logger.warning('Could not find suitable threshold, `threshold` is set to max score: %s.', str(e))
            self.threshold = float(np.unique(np.sort(scores))[-1])

    def mark_normal_outliers(self, anomaly_score: np.ndarray, normal_index: np.ndarray) -> np.array:
        """ Marks all elements of anomaly_score that are normal according to normal_index which have an anomaly_score
        that is above the 99% quantile of all normal anomaly_scores

        Args:
            anomaly_score (np.ndarray): array containing a time series of anomaly scores.
            normal_index (np.ndarray): boolean values indicating whether an element of anomaly_score is normal or not

        Notes:
            If precision/recall cannot be calculated, the threshold is set to the maximum score.

        Returns:
            selection (np.array): Boolean array which is true for all samples that are either normal and below quantile
                or not normal.
        """

        quantile = np.quantile(anomaly_score[normal_index], q=self.quantile)
        all_scores_compared = anomaly_score < quantile
        selection = np.logical_or(all_scores_compared, ~normal_index)
        return selection
