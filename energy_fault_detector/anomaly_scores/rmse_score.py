
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.anomaly_score import AnomalyScore

DataType = Union[pd.DataFrame, np.ndarray]

# Sentinel used to detect whether the deprecated `scale` parameter was passed explicitly.
_DEPRECATED = object()


class RMSEScore(AnomalyScore):
    """Calculate the RMSE of given reconstruction errors.

    .. deprecated::
        The ``scale`` parameter no longer has any effect and is only accepted for backwards compatibility.
        It will be removed in a future version.

    Configuration example:

    .. code-block:: yaml

        train:
          anomaly_score:
            name: rmse

    """

    def __init__(self, scale: bool = _DEPRECATED, **kwargs):

        super().__init__(**kwargs)
        if scale is not _DEPRECATED:
            warnings.warn(
                "The 'scale' parameter of RMSEScore is deprecated and no longer has any effect. "
                "It will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
        # `scale` is kept as an attribute for backwards compatibility but no longer influences the behaviour.
        self.scale = True

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: DataType, y: Optional[pd.Series] = None) -> 'RMSEScore':
        """Calculate standard deviation and mean on training data

        Args:
            x: numpy 2d array with differences between prediction and actual sensor values
            y (optional): not used, labels indicating whether sample is normal (True) or anomalous (False).
        """
        # fitted attributes need trailing underscore - and are not initialized
        self.std_x_: np.array = np.std(x, axis=0)
        self.mean_x_: np.array = np.mean(x, axis=0)

        self.fitted_ = True  # nothing to fit
        return self

    def standardize(self, x: DataType):
        """Standardization of the reconstruction error in x"""

        check_is_fitted(self)
        x_ = x.copy()
        if np.all(self.std_x_ > 0):
            x_ = (x - self.mean_x_) / self.std_x_
        else:
            x_ = x - self.mean_x_
        # replace possible inf values with 0
        x_[np.isinf(x_)] = 0
        return x_

    def transform(self, x: DataType) -> pd.Series:
        """Calculate the RMSE based on the deviation matrix.

        Args:
            x: numpy 2d array or pandas Dataframe with differences between prediction and actual sensor values

        Returns:
            RMSE for each sample.
        """
        x_ = self.standardize(x)

        scores = np.sqrt(np.mean(x_ ** 2, axis=1))
        if isinstance(x, (pd.DataFrame, pd.Series)):
            scores = pd.Series(scores, index=x.index)

        return scores
