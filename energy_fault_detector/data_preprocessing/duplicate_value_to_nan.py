from typing import Optional, Union, List

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core import DataTransformer
from energy_fault_detector.utils.index_utils import resolve_groupby_level


# noinspection PyAttributeOutsideInit
class DuplicateValuesToNan(DataTransformer):
    """Replaces duplicate values with NaN.

    In many data sets, zero can mean NaN, so we replace these duplicated values if they continue over
    `n_max_duplicates` times. The class can also be used for other values to replace.

    Example:

    .. code-block:: text

        value_to_replace = 0
        n_max_duplicates = 2
        Input: [0, 0, 0, 1, 2, 1, 3, 5, 1, 0, 0, 0, 0, 0, 7]
        Output: [0, 0, np.nan, 1, 2, 1, 3, 5, 1, 0, 0, np.nan, np.nan, np.nan, 7]

    Args:
        value_to_replace: The value to replace with NaN (default: 0.).
        n_max_duplicates: The maximum number of duplicates allowed before replacing with NaN (default: 144).
        features_to_exclude: List of features to not transform. Defaults to None.
            Some sensors simply do not change for a while and that is ok.

    Attributes:
        feature_names_in_: list of column names in input.
        feature_names_out_: list of columns in output.
    """

    def __init__(self, value_to_replace: float = 0., n_max_duplicates: int = 144,
                 features_to_exclude: List[str] = None, groupby_level: Optional[str] = "auto"):
        """
        Initialize the DuplicateValuesToNan transformer.

        Args:
            value_to_replace: The value to replace with NaN (default: 0.).
            n_max_duplicates: The maximum number of duplicates allowed before replacing with NaN (default: 144).
        """
        super().__init__()
        self.value_to_replace = value_to_replace
        self.n_max_duplicates = n_max_duplicates
        self.features_to_exclude: List[str] = features_to_exclude if features_to_exclude is not None else []
        self.groupby_level = groupby_level
        self.groupby_level_ = None

    def fit(self, x: Union[np.array, pd.DataFrame], y: Optional[np.array] = None) -> 'DuplicateValuesToNan':
        """
        Set feature names in and out.

        Args:
            x: The input data as a numpy array or pandas DataFrame.
            y: The target data as a numpy array (optional).

        Returns:
            self: The fitted DuplicateValuesToNan transformer.
        """
        self.feature_names_in_ = x.columns.to_list()
        self.feature_names_out_ = x.columns.to_list()
        self.groupby_level_ = resolve_groupby_level(x.index, self.groupby_level)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Replace any value that is duplicated more than `self.n_max_duplicates` with NaN.

        Args:
            x: The input data as a pandas DataFrame.

        Returns:
            The transformed data with duplicate values replaced with NaN.
        """
        check_is_fitted(self)

        to_replace: pd.DataFrame = (x == self.value_to_replace)

        for column in x.columns:
            if column in self.features_to_exclude:
                # Skip excluded features
                to_replace[column] = False
                continue

            col_series = x[column]
            if self.groupby_level_ is not None:
                shifted = col_series.groupby(level=self.groupby_level_).shift(1)
            else:
                shifted = col_series.shift(1)

            mask = col_series != shifted
            groups = mask.cumsum()
            # create counter for each group
            counter = groups.groupby(groups).cumcount()
            # update mask to replace values
            to_replace[column] = to_replace[column] & (counter >= self.n_max_duplicates)

        x_ = x.copy()
        x_[to_replace] = np.nan

        return x_

    def inverse_transform(self, x: Union[np.array, pd.DataFrame]) -> pd.DataFrame:
        """Not implemented for data replacer (not useful)"""
        return x

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns the list of feature names in the output."""
        check_is_fitted(self)
        return self.feature_names_out_
