from typing import Optional, List
import logging

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core import DataTransformer

logger = logging.getLogger('energy_fault_detector')


class LowUniqueValueFilter(DataTransformer):
    """Removes features with low unique values or high fraction of zeroes.

    Features are dropped if they have fewer unique values than `min_unique_value_count` or if the fraction of zeroes
    exceeds `max_col_zero_frac`.

    Args:
        min_unique_value_count (int): Minimum number of unique values a feature should have. Default: 2.
            If set to 2, only constant features are dropped.
        max_col_zero_frac (float): Maximum fraction of zeroes a column may contain. Default: 1.0

    Attributes:
        feature_names_in_ (list): List of column names in input.
        n_features_in_ (int): Number of columns in input.
        feature_names_out_ (list): List of column names to keep after filtering.
        columns_dropped_ (list): List of columns that were dropped during filtering.
    """

    def __init__(self, min_unique_value_count: int = 2, max_col_zero_frac: float = 1.0):
        super().__init__()

        self.min_unique_value_count: int = min_unique_value_count
        self.max_col_zero_frac: float = max_col_zero_frac

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: pd.DataFrame, y: Optional[np.array] = None,
            protected_features: Optional[List[str]] = None) -> 'LowUniqueValueFilter':
        """Fit the LowUniqueValueFilter to the data.

        This method evaluates the features based on the number of unique values and the fraction of zeroes, and
        determines which features to keep.

        Args:
            x (pd.DataFrame): The input data with features.
            y (Optional[np.array]): The target data (not used).
            protected_features: list of feature names that should never be dropped (e.g., conditional features for
                autoencoders). Warnings will be issued if these features would have been dropped otherwise.

        Returns:
            LowUniqueValueFilter: The fitted filter instance.
        """

        self.feature_names_in_ = x.columns.to_list()
        self.n_features_in_ = len(x.columns)

        protected_features = protected_features or []
        protected_lower = [f.lower() for f in protected_features]

        original_columns = x.columns
        counts = x.nunique()
        low_unique_count = counts[counts < self.min_unique_value_count].index

        # Protect features from low unique value dropping
        protected_low_unique = [col for col in low_unique_count if col.lower() in protected_lower]
        for col in protected_low_unique:
            unique_count = counts[col]
            logger.warning(f"Feature '{col}' has only {unique_count} unique value(s) (below threshold of "
                           f"{self.min_unique_value_count}), but is used as a conditional feature. "
                           f"Keeping it anyway.")

        low_unique_to_drop = [col for col in low_unique_count if col.lower() not in protected_lower]
        x = x.drop(low_unique_to_drop, axis=1)

        zero_pct_per_column = (x == 0).mean(axis=0)
        columns_to_drop = zero_pct_per_column[zero_pct_per_column > self.max_col_zero_frac].index

        # Protect features from high zero fraction dropping
        protected_high_zeros = [col for col in columns_to_drop if col.lower() in protected_lower]
        for col in protected_high_zeros:
            zero_frac = zero_pct_per_column[col]
            logger.warning(f"Feature '{col}' has {zero_frac*100:.1f}% zeros (exceeds threshold of "
                           f"{self.max_col_zero_frac*100:.1f}%), but is used as a conditional feature. "
                           f"Keeping it anyway.")

        high_zero_to_drop = [col for col in columns_to_drop if col.lower() not in protected_lower]
        x = x.drop(high_zero_to_drop, axis=1)

        self.columns_dropped_ = [col for col in original_columns if col not in x.columns]
        self.feature_names_out_ = x.columns.to_list()

        return self

    # pylint: disable=attribute-defined-outside-init
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by selecting only the features that are kept.

        Args:
            x (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data with only selected features.
        """

        check_is_fitted(self)

        x = x[self.feature_names_out_]
        return x

    # pylint: disable=attribute-defined-outside-init
    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform does nothing - since the columns dropped are not reconstructed."""
        return x

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns the list of feature names in the output."""
        check_is_fitted(self)
        return self.feature_names_out_
