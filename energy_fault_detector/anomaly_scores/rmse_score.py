
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.anomaly_score import AnomalyScore

DataType = Union[pd.DataFrame, np.ndarray]


class RMSEScore(AnomalyScore):
    """Calculate the RMSE of given reconstruction errors.

    Attributes:
        scale: If True, mean and std of the training/fit reconstruction errors will be used to standardize recon errors
            during transform. Default: True

    Configuration example:

    .. code-block:: yaml

        train:
          anomaly_score:
            name: rmse
            params:
              scale: false

    """

    def __init__(self, scale: bool = True, **kwargs):

        super().__init__(**kwargs)
        self.scale = scale

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: DataType, y: Optional[pd.Series] = None) -> 'RMSEScore':
        """Calculate standard deviation and mean on training data

        Args:
            x: numpy 2d array with differences between prediction and actual sensor values
            y (optional): not used, labels indicating whether sample is normal (True) or anomalous (False).
        """
        if not hasattr(self, 'scale'):
            # backwards compatibility, add missing attribute and set to True (as that was the standard behaviour)
            self.scale = True

        if self.scale:
            # fitted attributes need trailing underscore - and are not initialized
            self.std_x_: np.array = np.std(x, axis=0)
            self.mean_x_: np.array = np.mean(x, axis=0)

        self.fitted_ = True  # nothing to fit
        return self

    def transform(self, x: DataType) -> pd.Series:
        """Calculate the RMSE based on the deviation matrix.

        Args:
            x: numpy 2d array or pandas Dataframe with differences between prediction and actual sensor values

        Returns:
            RMSE for each sample.
        """
        if not hasattr(self, 'scale'):
            # backwards compatibility, add missing attribute and set to True (as that was the standard behaviour)
            self.scale = True

        check_is_fitted(self)

        x_ = x
        if self.scale:
            # standardization of the reconstruction error in X
            if np.all(self.std_x_ > 0):
                x_ = (x - self.mean_x_) / self.std_x_
            else:
                x_ = x - self.mean_x_
            # replace possible inf values with 0
            x_[np.isinf(x_)] = 0

        scores = np.sqrt(np.mean(x_ ** 2, axis=1))
        if isinstance(x, (pd.DataFrame, pd.Series)):
            scores = pd.Series(scores, index=x.index)

        return scores
