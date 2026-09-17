
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, MinCovDet, OAS
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.anomaly_score import AnomalyScore

DataType = Union[pd.DataFrame, np.ndarray]

# Sentinel used to detect whether the deprecated `scale` parameter was passed explicitly.
_DEPRECATED = object()

# Supported covariance estimation methods for the full dimensional covariance estimation (pca=False) path.
_COVARIANCE_METHODS = ("auto", "min_cov_det", "shrinkage")
_SHRINKAGE_METHODS = ("oas", "ledoit_wolf")


class MahalanobisScore(AnomalyScore):
    """Calculate Mahalanobis scores on the reconstruction errors.

    This class implements the mahalanobis norm calculation based on multiple variants of covariance 
    matrix estimation. The cases are defined by the user-set parameter ``pca`` and the relation of 
    the number of samples ``n`` and the number of features ``p``:
        1. (``pca=False``) and n>=min_cov_det_ratio * p: Estimates the covariance of the 
        mean-centred reconstruction errors using the MinCovDet algorithm and returns the 
        Mahalanobis distance of each sample. 
        2. (``pca=False``) and n<min_cov_det_ratio * p: Estimates the covariance of the 
        mean-centred reconstruction errors using shrinkage based algorithms like Ledoit-Wolf or OAS.
        The estimator can be specified via the parameter ``covariance_method`` and automatically 
        falls back to a shrinkage estimator (Ledoit-Wolf/OAS) when MinCovDet would become unstable.
        3. (``pca=True``): A PCA is carried out on  standardized reconstruction errors to reduce the 
        dimensionality. MinCovDet is then used to estimate the Covariance matrix in order to compute 
        the Mahalanobis distance of each sample. This method is deprecated because it does not consider
        all correlations in the reconstruction errors which might lead to less useful anomaly scores.

    .. deprecated::
        The ``pca=True`` path is deprecated and will be replaced by full covariance matrix estimators.
        It is kept for backwards compatibility; use ``pca=False`` for the direct
        full-dimensionality Mahalanobis distance. The ``scale`` parameter is deprecated and no
        longer has any effect.

    Args:
        pca: deprecated. If True, standardize, reduce dimensionality via PCA and estimate the
            covariance with MinCovDet (legacy behaviour). If False, estimate the
            covariance on the full-dimensionality mean-centred errors. Default ``True``.
        scale: deprecated, no longer has any effect.
        pca_min_var: parameter for PCA, variance to keep. Default 0.9
        mcd_support_fraction: parameter for Minimum Covariance Determinant estimation. Default 0.9
        covariance_method: which covariance estimator to use in the (``pca=False``) path.
            One of:
              - ``"auto"`` (default): use MinCovDet when ``n / p >= min_cov_det_ratio`` and the
                resulting covariance is stable (full rank / well conditioned), otherwise fall back
                to a shrinkage estimator (``shrinkage_method``).
              - ``"min_cov_det"``: always use MinCovDet. May raise or be unstable when ``n`` is
                not sufficiently larger than ``p``.
              - ``"shrinkage"``: always use a shrinkage estimator (``shrinkage_method``).
        min_cov_det_ratio: minimum sample-to-feature ratio ``n / p`` required (in ``"auto"`` mode)
            for MinCovDet to be used instead of shrinkage. Default ``2.0``.
        shrinkage_method: which shrinkage estimator to use. One of ``"oas"`` (default) or
            ``"ledoit_wolf"``.

    Configuration example:

    .. code-block:: yaml

        train:
          anomaly_score:
            name: mahalanobis
            params:
              pca: False
              covariance_method: auto
              min_cov_det_ratio: 2.0
              shrinkage_method: oas
    """

    def __init__(self, pca: bool = True, pca_min_var: float = 0.9, mcd_support_fraction: float = 0.9,
                 scale: bool = _DEPRECATED, covariance_method: str = "auto",
                 min_cov_det_ratio: float = 2.0, shrinkage_method: str = "oas"):
        super().__init__()

        self.pca: bool = pca
        self.pca_min_var = pca_min_var
        self.mcd_support_fraction = mcd_support_fraction
        self.covariance_method = covariance_method
        self.min_cov_det_ratio = min_cov_det_ratio
        self.shrinkage_method = shrinkage_method

        if pca:
            warnings.warn(
                "The PCA-based MahalanobisScore is deprecated and will be replaced by the full-dimensionality Mahalanobis "
                "norm implementation. Use pca=False for the direct full-dimensionality Mahalanobis norm.",
                DeprecationWarning,
                stacklevel=2,
            )
        if scale is not _DEPRECATED:
            warnings.warn(
                "The 'scale' parameter of MahalanobisScore is deprecated and no longer has any effect. "
                "It will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
        # `scale` is kept as an attribute for backwards compatibility but no longer influences the behaviour.
        self.scale = True

        if covariance_method not in _COVARIANCE_METHODS:
            raise ValueError(
                f"Invalid covariance_method '{covariance_method}'. "
                f"Supported values are: {', '.join(_COVARIANCE_METHODS)}."
            )
        if shrinkage_method not in _SHRINKAGE_METHODS:
            raise ValueError(
                f"Invalid shrinkage_method '{shrinkage_method}'. "
                f"Supported values are: {', '.join(_SHRINKAGE_METHODS)}."
            )

        # fitted attributes need trailing underscore
        self.pca_object: PCA = PCA(n_components=self.pca_min_var)
        self.min_cov_det_object: MinCovDet = MinCovDet(support_fraction=self.mcd_support_fraction,
                                                       assume_centered=True)

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: DataType, y: Optional[pd.Series] = None) -> 'MahalanobisScore':
        """Fit the covariance estimator to determine Mahalanobis distance.

        Args:
            x: numpy 2d array or pandas DataFrame with differences between prediction and actual sensor values.
            y (optional): not used, labels indicating whether sample is normal (True) or anomalous (False).
        """
        self.mean_x_: np.array = np.mean(x, axis=0)

        if self.pca:
            # Deprecated PCA-based path: standardize and reduce dimensionality before covariance estimation.
            self.std_x_: np.array = np.std(x, axis=0)
            scaled_x = self.standardize(x)
            self.pca_object.fit(scaled_x)
            pca_result = self.pca_object.transform(scaled_x)
            self.min_cov_det_object.fit(pca_result)
            self.cov_estimator_ = self.min_cov_det_object
            self.covariance_method_ = "min_cov_det"
        else:
            # Full Mahalanobis norm: full-dimensionality mean-centred errors.
            centered = x - self.mean_x_
            self.cov_estimator_, self.covariance_method_ = self._fit_covariance(centered)

        return self

    def _fit_covariance(self, x_centered: np.ndarray):
        """Fit the covariance estimator on the mean-centred errors and return (estimator, method).

        In ``auto`` mode MinCovDet is used when the sample-to-feature ratio is sufficient and the
        fitted covariance is stable, otherwise a shrinkage estimator is used.
        """
        n, p = x_centered.shape

        method = self.covariance_method
        if method == "auto":
            if n >= self.min_cov_det_ratio * p:
                method = "min_cov_det"
            else:
                method = "shrinkage"

        if method == "min_cov_det":
            try:
                self.min_cov_det_object.fit(x_centered)
            except (ValueError, np.linalg.LinAlgError):
                if self.covariance_method == "auto":
                    method = "shrinkage"
                else:
                    raise
            else:
                # Even with a sufficient n/p ratio, correlated features can produce a
                # singular / ill-conditioned covariance in auto mode -> fall back to shrinkage.
                if self.covariance_method == "auto" and self._is_degenerate_np(x_centered.shape):
                    method = "shrinkage"
                else:
                    return self.min_cov_det_object, method

        # shrinkage path
        estimator_cls = OAS if self.shrinkage_method == "oas" else LedoitWolf
        estimator = estimator_cls(assume_centered=True)
        estimator.fit(x_centered)
        return estimator, method

    def _is_degenerate_np(self, shape) -> bool:
        """Return True if the fitted MinCovDet covariance is singular or ill-conditioned."""
        cov = getattr(self.min_cov_det_object, "covariance_", None)
        p = shape[1]
        if cov is None:
            return True
        try:
            rank = np.linalg.matrix_rank(cov)
            cond = np.linalg.cond(cov)
        except np.linalg.LinAlgError:
            return True
        return rank < p or not np.isfinite(cond) or cond > self._MAX_CONDITION

    _MAX_CONDITION = 1e8

    def standardize(self, x: DataType):
        """Standardization of the reconstruction error in x (only used in the deprecated PCA path)."""

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
        """Calculate Mahalanobis distance from x.

        Args:
            x: numpy 2d array or pandas Dataframe with differences between prediction and actual sensor values

        Returns:
            Mahalanobis distance for each sample. Output is a pandas Series if input was a pandas DataFrame
        """
        check_is_fitted(self)
        if self.pca:
            check_is_fitted(self.min_cov_det_object)
            scaled_x = self.standardize(x)
            pca_result = self.pca_object.transform(scaled_x)
            scores = self.cov_estimator_.mahalanobis(pca_result)
        else:
            centered = x - self.mean_x_
            scores = self.cov_estimator_.mahalanobis(centered)

        if isinstance(x, (pd.DataFrame, pd.Series)):
            scores = pd.Series(scores, index=x.index)

        return scores
