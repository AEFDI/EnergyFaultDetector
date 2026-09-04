from typing import List, Optional, Union

import logging
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer

logger = logging.getLogger('energy_fault_detector')


class ForwardFillImputer(DataTransformer):

    """Impute missing values using time-based forward fill with a maximum gap duration.

    Forward-fills NaN values from the last valid observation, but only when the elapsed
    time between the last valid value and the current row does not exceed ``ffill_limit``.
    Rows where the gap is too large remain NaN and are subsequently dropped.

    The transformer assumes time-series data with a temporal index (DatetimeIndex or
    TimedeltaIndex). During ``fit``, it categorizes features as numerical or categorical,
    identifies and logs non-declared categorical features, and converts ``ffill_limit``
    to a :class:`~pandas.Timedelta`. During ``transform``, it forward-fills missing values
    (invalidating fills that exceed the time threshold), drops duplicate rows, and removes
    any rows still containing NaN values.

    Dropping duplicate rows after forward-filling is necessary because forward-filling
    propagates the last valid value into consecutive NaN rows. If the underlying signal did
    not change between observations, the filled rows become exact duplicates of the source
    row. Removing these redundant rows avoids introducing identical training samples that
    carry no additional information and would otherwise bias the reconstruction error.

    Attributes:
        feature_names_in_ (List[str]): List of input feature names observed during fitting.
        feature_names_out_ (List[str]): List of output feature names (same as input features unless some were dropped).
        ffill_limit_ (pd.Timedelta): Fitted ``ffill_limit`` converted to a Timedelta.
        categorical_features (List[str]): List of user-specified categorical feature names (matched by exact name).
        numerical_columns (List[str]): List of identified numerical column names after filtering out non-declared categoricals.
        categorical_columns (List[str]): List of identified categorical column names (from ``categorical_features``).
        non_declared_categorical_features (List[str]): List of columns detected as object-type (categorical) but not declared as such.
    """

    def __init__(self, ffill_limit: Union[str, pd.Timedelta] = "15min",
                 categorical_features: Optional[List[str]] = None):
        """Initializes the ForwardFillImputer with a forward-fill time limit and optional categorical features.

        Args:
            ffill_limit (str | pd.Timedelta): Maximum time gap to forward-fill, expressed as a
                pandas-compatible Timedelta string (e.g. ``"15min"``, ``"1h"``) or a
                :class:`~pandas.Timedelta` instance.  NaNs whose elapsed time from the last valid
                observation exceeds this limit are left unfilled and the corresponding rows are
                dropped. Default is ``"15min"``.
            categorical_features (Optional[List[str]]): List of column names to treat as categorical features.
                Columns matching any of these (by exact name) are treated as categorical; others as numerical.
                Non-declared object-type columns in numerical slots are automatically detected and excluded from processing.
                Default is None (interpreted as empty list).
        """
        super().__init__()
        self.ffill_limit = ffill_limit

        # Attributes set during fit.
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.ffill_limit_: pd.Timedelta = pd.Timedelta(0)
        self.categorical_features = categorical_features if categorical_features else []
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.non_declared_categorical_features: List[str] = []

    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> "ForwardFillImputer":
        """Fits the imputer to the input data by identifying feature types and storing metadata.

        Populates ``numerical_columns``, ``categorical_columns``, and ``non_declared_categorical_features``
        based on the input DataFrame's structure and user-provided categorical feature hints.
        Converts ``ffill_limit`` to a :class:`~pandas.Timedelta` and stores it as ``ffill_limit_``.

        Args:
            x (pd.DataFrame): Input feature DataFrame. Must contain all features used during ``transform``.
            y (Optional[pd.Series]): Target variable. Ignored; included for compatibility with the scikit-learn API.

        Returns:
            ForwardFillImputer: The fitted transformer instance (self), for method chaining.

        Raises:
            TypeError: If `x` is not a pandas DataFrame.
            ValueError: If `ffill_limit` cannot be converted to a Timedelta.
        """
        if not isinstance(x, pd.DataFrame):
            raise TypeError("x must be a pandas DataFrame.")

        self.feature_names_in_ = x.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        try:
            self.ffill_limit_ = pd.Timedelta(self.ffill_limit)
        except ValueError as e:
            raise ValueError(
                f"ffill_limit must be a pandas-compatible Timedelta string or Timedelta "
                f"(e.g. '15min', '1h'), got: {self.ffill_limit!r}"
            ) from e

        self.numerical_columns = [col for col in self.feature_names_in_
                                  if col not in self.categorical_features]
        self.categorical_columns = [col for col in self.feature_names_in_
                                    if col in self.categorical_features]
        # Clean numerical columns from non_declared categorical features
        numerical_data = x.loc[:, self.numerical_columns]
        self.non_declared_categorical_features = numerical_data.select_dtypes(include='object').columns.tolist() 
        # Keep numerical and booleans
        self.numerical_columns = [col for col in self.numerical_columns if col not in self.non_declared_categorical_features]

        if self.non_declared_categorical_features:
            logger.warning(f"Non-declared categorical features found in data: {self.non_declared_categorical_features}. "
                        f"They will be dropped. Consider adding them to the categorical_features list if they should be"
                        f" treated as categorical.")

        # Drop columns that are entirely NaN — they cannot be forward-filled and would otherwise
        # cause every row to be dropped by dropna(how="any").
        all_nan_cols = [col for col in self.numerical_columns + self.categorical_columns
                        if x[col].isna().all()]
        if all_nan_cols:
            logger.warning(f"Columns containing only NaN values found: {all_nan_cols}. "
                        f"They will be dropped as they cannot be forward-filled.")
            self.numerical_columns = [col for col in self.numerical_columns if col not in all_nan_cols]
            self.categorical_columns = [col for col in self.categorical_columns if col not in all_nan_cols]

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Applies time-based forward-fill imputation and cleaning pipeline to input data.

        The steps executed are:

        1. Forward-fill all NaN values.
        2. Invalidate fills where the elapsed time from the last valid observation exceeds
           ``ffill_limit_`` (set those values back to NaN).
        3. Drop duplicate rows (exact duplicates created by forward-filling where the
           underlying signal did not change).
        4. Drop any rows still containing NaN values (``dropna(how="any")``).

        Unlike the previous resampling-based approach, the original (possibly irregular)
        timestamps are preserved — no synthetic rows are introduced.

        Args:
            x (pd.DataFrame): Input feature DataFrame, with the same column names as used in ``fit()``.

        Returns:
            pd.DataFrame: Cleaned and imputed DataFrame, containing only the columns from ``get_feature_names_out()``.

        Raises:
            TypeError: If `x` is not a pandas DataFrame, or if its index is not temporal (DatetimeIndex/TimedeltaIndex).
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

        if not isinstance(x.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            raise TypeError(
                "x index must be a DatetimeIndex or TimedeltaIndex for time-based forward fill."
            )
        # Harmonize data types in numerical columns to avoid issues during concatenation after one-hot encoding
        for col in self.numerical_columns:
            if col in x.columns:
                try:
                    x[col] = x[col].astype(float, errors='raise')
                except ValueError as e:
                    raise ValueError(f"Column '{col}' cannot be converted to float.") from e
        x_selected = x[self.numerical_columns + self.categorical_columns]

        # --- Time-based forward fill ---
        idx_series = x.index.to_series()
        df_filled = x_selected.copy()
        for col in df_filled.columns:
            col_series = df_filled[col]
            nan_mask = col_series.isna()
            if not nan_mask.any():
                continue
            filled = col_series.ffill()
            # Time elapsed since the last valid observation for each row
            last_valid = idx_series.where(col_series.notna()).ffill()
            elapsed = idx_series - last_valid
            # Invalidate fills where the gap exceeds the threshold
            invalid = nan_mask & (elapsed > self.ffill_limit_)
            filled[invalid] = np.nan
            df_filled[col] = filled

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
