from typing import List, Optional

import logging
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer

logger = logging.getLogger('energy_fault_detector')

class ForwardFillImputer(DataTransformer):

    """Impute missing values using forward fill with limit after resampling frequency.

    Duplicate rows are dropped and any remaining rows with NaN values are removed.

    The transformer assumes time-series data with a temporal index (DatetimeIndex, TimedeltaIndex, or PeriodIndex).
    During `fit`, it categorizes features as numerical or categorical, identifies and logs non-declared categorical features,
    and prepares internal metadata. During `transform`, it resamples the data to a uniform frequency, forward-fills missing
    values up to a specified limit, drops duplicate rows, and removes any rows still containing NaN values. This transformer 
    is useful for time-series datasets where missing values need to be imputed based on previous observations.

    Attributes:
        feature_names_in_ (List[str]): List of input feature names observed during fitting.
        feature_names_out_ (List[str]): List of output feature names (same as input features unless some were dropped).
        categorical_features (List[str]): List of user-specified categorical feature names or prefixes.
        numerical_columns (List[str]): List of identified numerical column names after filtering out non-declared categoricals.
        categorical_columns (List[str]): List of identified categorical column names (from `categorical_features`).
        non_declared_categorical_features (List[str]): List of columns detected as object-type (categorical) but not declared as such.
    """

    def __init__(self, freq: str = "1Min", ffill_limit: int = 15, categorical_features: Optional[List[str]] = None):
        """Initializes the ForwardFillImputer with specified frequency, forward-fill limit, and optional categorical features.

        Args:
            freq (str): Target resampling frequency for the time series (e.g., "1Min", "5Min", "H"). Passed to `pandas.DataFrame.asfreq()`.
                Default is "1Min".
            ffill_limit (int): Maximum number of consecutive NaN values to forward-fill after resampling.
                NaNs beyond this limit remain unchanged and the corresponding rows will be dropped later. Default is 15.
            categorical_features (Optional[List[str]]): List of column names or prefixes to treat as categorical features.
                Columns matching any of these (as substring) are treated as categorical; others as numerical.
                Non-declared object-type columns in numerical slots are automatically detected and excluded from processing.
                Default is None (interpreted as empty list).
        """
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
        """Fits the imputer to the input data by identifying feature types and storing metadata.

        Populates `numerical_columns`, `categorical_columns`, and `non_declared_categorical_features`
        based on the input DataFrame's structure and user-provided categorical feature hints.

        Args:
            x (pd.DataFrame): Input feature DataFrame. Must contain all features used during `transform`.
            y (Optional[pd.Series]): Target variable. Ignored; included for compatibility with the scikit-learn API.

        Returns:
            ForwardFillImputer: The fitted transformer instance (self), for method chaining.

        Raises:
            TypeError: If `x` is not a pandas DataFrame.
            ValueError: If internal consistency checks fail (e.g., invalid column references).
        """
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
        """Applies forward-fill imputation and cleaning pipeline to input data.

        The steps executed are:
        1. Resample the data to `self.freq` using `asfreq()`.
        2. Forward-fill missing values up to `self.ffill_limit`.
        3. Drop duplicate rows.
        4. Drop any rows still containing NaN values (`dropna(how="any")`).

        Args:
            x (pd.DataFrame): Input feature DataFrame, with the same column names as used in `fit()`.

        Returns:
            pd.DataFrame: Cleaned and imputed DataFrame, containing only the columns from `get_feature_names_out()`.

        Raises:
            TypeError: If `x` is not a pandas DataFrame, or if its index is not temporal (DatetimeIndex/TimedeltaIndex/PeriodIndex).
            ValueError: If `x` is missing columns seen during `fit()`, or if numerical columns cannot be converted to float.
            ValueError: If the imputer has not been fitted (via `check_is_fitted`).
        """
        check_is_fitted(self, "n_features_in_")

        x = x.copy()  # Avoid modifying the original DataFrame

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
        """Returns the input DataFrame with columns reordered to match the original input feature order.

        This method is included for compatibility with the `DataTransformer`/scikit-learn API.
        Since forward-fill imputation is not invertible (lost NaNs and duplicates cannot be recovered),
        only column reordering is performed—no actual value inversion occurs.

        Args:
            x (pd.DataFrame): Transformed DataFrame with columns matching `feature_names_out_`.

        Returns:
            pd.DataFrame: DataFrame with columns reordered to match `feature_names_in_` (original order).

        Raises:
            TypeError: If `x` is not a pandas DataFrame.
            ValueError: If the transformer has not been fitted (via `check_is_fitted`).
        """
        check_is_fitted(self, "n_features_in_")
        return pd.DataFrame(x, columns=self.feature_names_in_)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return output feature names for downstream transformers."""
        check_is_fitted(self, "n_features_in_")
        return self.numerical_columns + self.categorical_columns
