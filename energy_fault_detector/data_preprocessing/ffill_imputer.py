from typing import List, Optional

import logging
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer

logger = logging.getLogger('energy_fault_detector')

class ForwardFillImputer(DataTransformer):

    """Impute missing values using forward fill with limit after resampling frequency.
    Duplicate rows are dropped and any remaining rows with NaN values are removed.
    """

    def __init__(self, freq: str = "1Min", ffill_limit: int = 15, categorical_features: Optional[List[str]] = None):
        super().__init__()
        self.freq = freq
        self.ffill_limit = ffill_limit

        # Attributes set during fit.
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.categorical_features = categorical_features if categorical_features else []
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.non_declared_categorical_features: List[str] = []

    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> "ForwardFillImputer":
        """Store input feature metadata required by the DataTransformer API."""
        # TODO: at this point there is no handling of all-NaN columns, should be included in fit to drop them and log a warning.

        if not isinstance(x, pd.DataFrame):
            raise TypeError("x must be a pandas DataFrame.")

        self.feature_names_in_ = x.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        self.numerical_columns = [col for col in self.feature_names_in_ if not any(feature in col for feature in self.categorical_features)] # Uses nested for loop in case categorical cols have been encoded already and contain the original categorical feature name as a substring.
        self.categorical_columns = [col for col in self.feature_names_in_ if any(feature in col for feature in self.categorical_features)] # Uses nested for loop in case categorical cols have been encoded already and contain the original categorical feature name as a substring.
        # Clean numerical columns from non_declared categorical features
        numerical_data = x.loc[:, self.numerical_columns]
        self.non_declared_categorical_features = numerical_data.select_dtypes(include='object').columns.tolist() 
        # Keep numerical and booleans
        self.numerical_columns = [col for col in self.numerical_columns if col not in self.non_declared_categorical_features]

        if self.non_declared_categorical_features:
            logger.info(f"Non-declared categorical features found in data: {self.non_declared_categorical_features}. "
                        f"They will be dropped. Consider adding them to the categorical_features list if they should be treated as categorical.")

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply impute_by_row logic: asfreq -> ffill(limit) -> drop_duplicates -> dropna."""
        check_is_fitted(self, "n_features_in_")
        feature_names_in = self.feature_names_in_
        if feature_names_in is None:
            raise ValueError("ForwardFillImputer is not fitted.")

        if not isinstance(x, pd.DataFrame):
            raise TypeError("x must be a pandas DataFrame.")

        missing_columns = [col for col in feature_names_in if col not in x.columns]
        if missing_columns:
            raise ValueError(f"Input is missing columns seen during fit: {missing_columns}")

        if not isinstance(x.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
            raise TypeError(
                "x index must be a DatetimeIndex, TimedeltaIndex, or PeriodIndex to use asfreq()."
            )
        # Harmonize data types in numerical columns to avoid issues during concatenation after one-hot encoding
        for col in self.numerical_columns:
            if col in x.columns:
                try:
                    x[col] = x[col].astype(float, errors='raise')
                except ValueError as e:
                    raise ValueError(f"Column '{col}' cannot be converted to float.") from e
        x_selected = x[self.numerical_columns + self.categorical_columns]
        df_resampled = x_selected.asfreq(self.freq)
        df_filled = df_resampled.ffill(limit=self.ffill_limit)
        df_cleaned = df_filled.drop_duplicates(keep="first")
        df_final = df_cleaned.dropna(how="any")

        self.feature_names_out_ = self.get_feature_names_out()
        return df_final[self.feature_names_out_]

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        For compatibility, this method returns the DataFrame with the original column
        order but assumes no special inversions are necessary after imputation.
        """
        check_is_fitted(self, "n_features_in_")
        return pd.DataFrame(x, columns=self.feature_names_in_)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return output feature names for downstream transformers."""
        check_is_fitted(self, "n_features_in_")
        return self.numerical_columns + self.categorical_columns
