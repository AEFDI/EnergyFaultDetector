"""Clip data before standardization or normalization"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer

logger = logging.getLogger('energy_fault_detector')


# noinspection PyAttributeOutsideInit
class DataClipper(DataTransformer):
    """Clip data to remove outliers.

    Args:
        lower_percentile (float): The lower percentile for clipping (default: 0.01).
        upper_percentile (float): The upper percentile for clipping (default: 0.99).
        features_to_exclude (List[str] | None): Column names that should not be clipped.
        features_to_clip (List[str] | None): Column names that should be clipped (mutually exclusive with
            features_to_exclude).

    Configuration example:

        .. code-block:: text

            train:
              data_clipping:
                lower_percentile: 0.001
                upper_percentile: 0.999
                  features_to_exclude:
                    - do_not_clip_this_feature
    """

    def __init__(self, lower_percentile: float = 0.01, upper_percentile: float = 0.99,
                 features_to_exclude: Optional[List[str]] = None, features_to_clip: Optional[List[str]] = None) -> None:

        super().__init__()
        if features_to_clip is not None and features_to_exclude is not None:
            raise ValueError('Only one of features_to_clip or features_to_exclude can be specified.')
        if not (0.0 <= lower_percentile <= 1.0) or not (0.0 <= upper_percentile <= 1.0):
            raise ValueError('Percentiles must be within [0, 1].')
        if lower_percentile >= upper_percentile:
            raise ValueError('lower_percentile must be strictly less than upper_percentile.')

        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.feature_to_exclude: Optional[List[str]] = features_to_exclude
        self.features_to_clip: Optional[List[str]] = features_to_clip

    def fit(self, x: pd.DataFrame, y: Optional[np.array] = None) -> 'DataClipper':
        """Set feature names in and out."""
        if not isinstance(x, pd.DataFrame):
            raise TypeError('DataClipper.fit expects a pandas DataFrame.')
        self.feature_names_in_ = x.columns.to_list()
        self.feature_names_out_ = x.columns.to_list()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Clips the data to remove outliers, excluding angles.

        Args:
            x (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The clipped DataFrame.
        """

        check_is_fitted(self)

        # Select feature to clip
        x_ = x.copy()
        if self.feature_to_exclude is not None:
            selected_features = [col for col in x_.columns if col not in self.feature_to_exclude]
        elif self.features_to_clip is not None:
            selected_features = [col for col in x_.columns if col in self.features_to_clip]
        else:
            # Clip all numeric columns
            selected_features = x_.columns.tolist()

        # Exclude non-numeric columns
        x_numeric = x_[selected_features].select_dtypes(include=np.number)

        if x_numeric.shape[1] == 0:
            logger.debug('DataClipper.transform: no numeric columns selected; returning input unchanged.')
            return x_

        # Clip the data using the specified percentiles
        x_clipped = x_numeric.clip(
            x_numeric.quantile(self.lower_percentile),
            x_numeric.quantile(self.upper_percentile),
            axis=1
        )

        # Update the original DataFrame with the clipped values
        x_[x_clipped.columns] = x_clipped

        return x_

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Not implemented for data clipper (not useful)"""
        return x

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns the list of feature names in the output."""
        check_is_fitted(self)
        return self.feature_names_out_
