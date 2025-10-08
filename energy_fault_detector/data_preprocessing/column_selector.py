
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer


class ColumnSelector(DataTransformer):
    """Class for selecting columns, using the provided list of features to exclude/drop and the fraction of NaNs.

    Args:
        max_nan_frac_per_col: maximum fraction of NaN values allowed per column. Defaults to 0.05.
            If the fraction exceeds max_nan_frac_per_col, the column is dropped.
        features_to_exclude: list of features that should be dropped. Defaults to None.

    Attributes:
        feature_names_in_: list of column names in input.
        n_features_in_: number of columns in input.
        feature_names_out_: list of column names to keep / selected.
        columns_dropped_: list of columns that were dropped.
    """

    def __init__(self, max_nan_frac_per_col: float = 0.05, features_to_exclude: List[str] = None):

        super().__init__()

        self.max_nan_frac_per_col: float = max_nan_frac_per_col
        self.features_to_exclude: List[str] = features_to_exclude if features_to_exclude is not None else []

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: pd.DataFrame, y: Optional[np.array] = None) -> 'ColumnSelector':
        """Find columns to keep for training

        Args:
            x: data to filter based on NaN fractions
            y: target variable, currently unused.
        """

        self.feature_names_in_ = x.columns.to_list()
        self.n_features_in_ = len(x.columns)

        # drop features to exclude - ignore upper/lower case
        to_drop = [col for col in x.columns if col.lower() in
                   [excluded_feature.lower() for excluded_feature in self.features_to_exclude]
                   ]
        x_transformed = x.drop(to_drop, axis=1, errors='ignore')

        # drop columns which have more than max_nan_frac_per_col relative NaN frequency
        empty_percentage = x_transformed.isnull().mean(axis=0)
        empty_cols = empty_percentage[empty_percentage >= self.max_nan_frac_per_col].index
        x_transformed = x_transformed.drop(empty_cols, axis=1)

        # select relevant numeric columns and set attribute for transform
        self.feature_names_out_ = x_transformed.columns.to_list()
        self.columns_dropped_ = x.columns.difference(x_transformed.columns).to_list()

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Drop columns from dataframe x."""

        check_is_fitted(self)

        # It is possible that new data misses some columns, which were present in training data, if this is the case
        # transformation is not possible.
        missing_columns = set(self.feature_names_out_) - set(x.columns)
        if len(missing_columns) > 0:
            missing_columns_sorted = sorted(missing_columns)
            missing_columns_str = ', '.join(missing_columns_sorted)
            raise ValueError(
                'ColumnSelector: There are columns missing in the prediction data, which were present in'
                ' the training data. Missing columns: '
                f"{missing_columns_str}. New models need to be trained!"
            )

        x = x[self.feature_names_out_]  # ensure ordering
        return x

    def inverse_transform(self, x: np.array) -> pd.DataFrame:
        """Inverse transform does nothing in case of column selector - since the columns dropped are
        not reconstructed."""
        return x

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns the list of feature names in the output."""
        check_is_fitted(self)
        return self.feature_names_out_
