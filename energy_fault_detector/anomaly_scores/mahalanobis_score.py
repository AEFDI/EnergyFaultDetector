
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.anomaly_score import AnomalyScore

DataType = Union[pd.DataFrame, np.ndarray]


class MahalanobisScore(AnomalyScore):
    """Calculate mahalanobis scores using sklearn MinCovDet und optionally a PCA to accelerate calculations.

    Args:
        pca: boolean to indicate whether PCA should be done before determining the covariance. Default true.
        pca_min_var: parameter for PCA, variance to keep. Default 0.9
        mcd_support_fraction: parameter for Minimum Covariance Determinant estimation. Default 0.9
        scale: If True, std of the training/fit reconstruction errors will be used to scale recon
            errors before applying MinCovDet. Default: False


    Configuration example:

    .. code-block:: text

        train:
          anomaly_score:
            name: mahalanobis
            params:
              pca: True
              pca_min_var: 0.9
              mcd_support_fraction: 0.9
              scale: False
    """

    def __init__(self, pca: bool = True, pca_min_var: float = 0.9, mcd_support_fraction: float = 0.9,
                 scale: bool = False):
        super().__init__()

        self.pca: bool = pca
        self.pca_min_var = pca_min_var
        self.mcd_support_fraction = mcd_support_fraction
        self.scale = scale

        # fitted attributes need trailing underscore
        self.pca_object: PCA = PCA(n_components=self.pca_min_var)
        self.min_cov_det_object: MinCovDet = MinCovDet(support_fraction=self.mcd_support_fraction, assume_centered=True)

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: DataType, y: Optional[pd.Series] = None) -> 'MahalanobisScore':
        """Fit MinCovDet object to determine Mahalanobis distance.

        Args:
            x: numpy 2d array or pandas DataFrame with differences between prediction and actual sensor values.
            y (optional): not used, labels indicating whether sample is normal (True) or anomalous (False).
        """
        self.mean_x_: np.array = np.mean(x, axis=0)
        if self.scale:
            self.std_x_: np.array = np.std(x, axis=0)
            # standardization of the reconstruction error in X
            if np.all(self.std_x_ > 0):
                scaled_x = (x - self.mean_x_) / self.std_x_
            else:
                scaled_x = x - self.mean_x_
            # replace possible inf values with 0
            scaled_x[np.isinf(scaled_x)] = 0
        else:
            scaled_x = x - self.mean_x_

        # Covariance estimation
        if self.pca:
            self.pca_object.fit(scaled_x)
            pca_result = self.pca_object.transform(scaled_x)
        else:
            pca_result = scaled_x

        self.min_cov_det_object.fit(pca_result)

        return self

    def transform(self, x: DataType) -> pd.Series:
        """Calculate Mahalanobis distance from x.

        Args:
            x: numpy 2d array or pandas Dataframe with differences between prediction and actual sensor values

        Returns:
            Mahalanobis distance for each sample. Output is a pandas Series if input was a pandas DataFrame
        """
        check_is_fitted(self)
        check_is_fitted(self.min_cov_det_object)
        if self.scale:
            # standardization of the reconstruction error in X
            if np.all(self.std_x_ > 0):
                scaled_x = (x - self.mean_x_) / self.std_x_
            else:
                scaled_x = x - self.mean_x_
                # replace possible inf values with 0
            scaled_x[np.isinf(scaled_x)] = 0
        else:
            scaled_x = x - self.mean_x_

        if self.pca:
            pca_result = self.pca_object.transform(scaled_x)
        else:
            pca_result = scaled_x

        scores = self.min_cov_det_object.mahalanobis(pca_result)
        if isinstance(x, (pd.DataFrame, pd.Series)):
            scores = pd.Series(scores, index=x.index)

        return scores
