"""Since the EnergyFaultDetector not yet contains the TimestampTransformer it is added here as a separate script."""

from __future__ import annotations

import logging
from typing import Optional, List, Union, Tuple, Callable
from calendar import isleap
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer

logger = logging.getLogger("energy_fault_detector")


def _second_of_minute(ind) -> np.ndarray:
    # 0..59 -> [0,1)
    return ind.second / 60.0


def _minute_of_hour(ind) -> np.ndarray:
    # 0..59 -> [0,1)
    return ind.minute / 60.0


def _hour_of_day(ind) -> np.ndarray:
    # 0..23 -> [0,1)
    return ind.hour / 24.0


def _day_of_week(ind) -> np.ndarray:
    # Monday=0..Sunday=6 -> [0,1)
    return ind.day_of_week / 7.0


def _day_of_month(ind) -> np.ndarray:
    # 1..days_in_month -> [0,1)
    # Need access to days_in_month via the Series accessor
    days_in_month = ind.days_in_month
    return ind.day / days_in_month


def _month_of_year(ind) -> np.ndarray:
    # 1..12 -> [0,1)
    return ind.month / 12.0


def _is_weekend(ind) -> np.ndarray:
    # Saturday=5, Sunday=6 -> weekend
    dow = ind.day_of_week
    if isinstance(dow, (bool, np.bool_)):
        return float(dow >= 5)
    return (dow >= 5).astype(float)


def _day_of_year(ind) -> np.ndarray:
    # 1..365/366 -> [0,1)
    # use .dayofyear and .is_leap_year from the DatetimeIndex accessor
    day_of_year = ind.dayofyear
    is_leap = ind.is_leap_year
    year_length = 365 + is_leap.astype(int)
    return day_of_year / year_length


def _year(ind) -> np.ndarray:
    # raw year (not normalized) – often useful for drift
    return ind.year.astype(float)


# Periodic logical features: we will encode them as sine & cosine
_PERIODIC_FEATURES: dict[str, Callable[[pd.DatetimeIndex], np.ndarray]] = {
    "second_of_minute": _second_of_minute,
    "minute_of_hour": _minute_of_hour,
    "hour_of_day": _hour_of_day,
    "day_of_week": _day_of_week,
    "day_of_month": _day_of_month,
    "day_of_year": _day_of_year,
    "month_of_year": _month_of_year,
}

# Non-periodic (scalar) features
_NON_PERIODIC_FEATURES: dict[str, Callable[[pd.DatetimeIndex], np.ndarray]] = {
    "is_weekend": _is_weekend,
    "year": _year,
}


class TimestampTransformer(DataTransformer):
    """A timestamp features generator for time-series data.

    This transformer adds normalized time-derived features based on a DatetimeIndex
    or a dedicated timestamp column:

        - second_of_minute   -> second / 60                in [0, 1)
        - minute_of_hour     -> minute / 60                in [0, 1)
        - hour_of_day        -> hour / 24                  in [0, 1)
        - day_of_week        -> weekday / 7                in [0, 1)
        - day_of_month       -> day / days_in_month        in [0, 1)
        - day_of_year        -> dayofyear / 365/366        in [0, 1)
        - month_of_year      -> month / 12                 in [0, 1)
        - is_weekend         -> 1 if Sat/Sun else 0
        - year               -> calendar year (float)

    This transformer is one-way; the `inverse_transform` returns the input unchanged.
    All features are normalized to the range [0, 1], except for `year`, which can be used
    to model drift. The features do not need to be scaled/normalized.

    Args:
        features (Optional[List[str]]): List of feature names to generate. Supported:
            ["second_of_minute", "minute_of_hour", "hour_of_day", "day_of_week", "day_of_month", "month_of_year",
             "is_weekend", "year"]
        timestamp_col (Optional[str]): The column name of the DataFrame containing timestamps.
            If None, the index is assumed to be the timestamp.
        groupby_level (Optional[str]): Optional index level name or position for grouping (e.g., 'device_id' or 0).
            If provided and a MultiIndex is used, timestamp features are extracted correctly per group.
            If 'auto', automatically detects the non-datetime level in a MultiIndex.
            Default: None (no grouping).

    Configuration example:

    .. code-block:: yaml

        train:
          data_preprocessor:
            steps:
              - name: standard_scaler
              - name: timestamp_transformer
                params:
                  features:
                    - hour_of_day
                    - day_of_week
                    - month_of_year

    """

    def __init__(self, features: List[str] = None, timestamp_col: Optional[str] = None,
                 groupby_level: Optional[str] = 'auto'):
        super().__init__()
        self.features: List[str] = features if features is not None else []
        self.timestamp_col: Optional[str] = timestamp_col
        self.groupby_level: Optional[str] = groupby_level
        self.groupby_level_: Optional[str] = None  # Will be set during fit

    def _resolve_index(self, x: pd.DataFrame):
        """Return the DatetimeIndex accessor to use, based on timestamp_col or the index.

        Returns:
            A pandas DatetimeIndex accessor (.dt) suitable for extracting time features.
        """
        if self.timestamp_col is not None:
            if self.timestamp_col not in x.columns:
                raise ValueError(f"TimestampTransformer: column '{self.timestamp_col}' not found in DataFrame.")
            ts = x[self.timestamp_col]
            if not pd.api.types.is_datetime64_any_dtype(ts):
                raise ValueError(f"TimestampTransformer: column '{self.timestamp_col}' must be datetime64.")
            return ts.dt

        # Handle index - could be DatetimeIndex or MultiIndex with DatetimeIndex level
        if isinstance(x.index, pd.DatetimeIndex):
            return x.index.to_series().dt
        elif isinstance(x.index, pd.MultiIndex):
            # Find the datetime level
            datetime_level_idx = None
            for i, level in enumerate(x.index.levels):
                if isinstance(level, pd.DatetimeIndex):
                    datetime_level_idx = i
                    break
            if datetime_level_idx is None:
                raise ValueError(
                    "TimestampTransformer: MultiIndex must contain at least one DatetimeIndex level "
                    "if no 'timestamp_col' is provided."
                )
            return pd.Series(x.index.get_level_values(datetime_level_idx), index=x.index).dt
        else:
            raise ValueError(
                "TimestampTransformer: DataFrame index must be a DatetimeIndex or MultiIndex with "
                "DatetimeIndex level if no 'timestamp_col' is provided."
            )

    def fit(self, x: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> 'TimestampTransformer':
        """Validate configuration and record feature names.

        Args:
            x: Input DataFrame with DatetimeIndex or a datetime64 timestamp column.
            y: Unused, present for compatibility.
        """

        _ = self._resolve_index(x)  # validate input

        # Resolve groupby_level (handles 'auto' detection)
        self.groupby_level_ = self._resolve_groupby_level(x)

        # Validate requested features
        supported_features = set(_PERIODIC_FEATURES.keys()) | set(_NON_PERIODIC_FEATURES.keys())
        unknown = [f for f in self.features if f not in supported_features]
        if unknown:
            raise ValueError(f"TimestampTransformer: unknown features {unknown}. "
                             f"Supported: {sorted(supported_features)}")

        self.feature_names_in_ = list(x.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        added_cols: List[str] = []
        for fname in self.features:
            if fname in _PERIODIC_FEATURES:
                added_cols.append(self._sine_name(fname))
                added_cols.append(self._cosine_name(fname))
            else:
                # non-periodic
                added_cols.append(fname)

        self.feature_names_added_ = added_cols
        self.feature_names_out_ = self.feature_names_in_ + self.feature_names_added_
        self._is_fitted = True
        return self

    def _resolve_groupby_level(self, x: pd.DataFrame) -> Optional[str]:
        """Resolve the groupby level from the user parameter or auto-detect.

        Args:
            x: Input DataFrame.

        Returns:
            The resolved groupby level (name or position), or None if no grouping.
        """
        if self.groupby_level is None:
            return None

        if self.groupby_level != 'auto':
            return self.groupby_level

        # Auto-detect: find non-datetime level in MultiIndex
        if not isinstance(x.index, pd.MultiIndex):
            return None  # Simple index, no grouping needed

        # Find datetime and non-datetime levels
        datetime_level_idx = None
        non_datetime_levels = []

        for i, level in enumerate(x.index.levels):
            if isinstance(level, pd.DatetimeIndex):
                datetime_level_idx = i
            else:
                non_datetime_levels.append(i)

        if datetime_level_idx is None:
            return None  # No datetime level found

        if len(non_datetime_levels) == 0:
            return None  # Only datetime levels, no grouping needed

        # Use the first non-datetime level as groupby level
        return non_datetime_levels[0]

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the timestamp features from the given dataset.

        Args:
            x (pd.DataFrame): The data to transform.

        Raises:
            NotFittedError: Raised by check_is_fitted utility function if the object has not been fitted.

        Returns:
            pd.DataFrame: Transformed dataset.
        """
        check_is_fitted(self)
        ind = self._resolve_index(x)
        x_ = x.copy()  # TODO: may problematic for large datasets

        for feature in self.features:
            if feature in _PERIODIC_FEATURES:
                phase = _PERIODIC_FEATURES[feature](ind)
                # convert to radians
                phase = 2 * np.pi * phase
                # add columns
                x_[self._sine_name(feature)] = np.sin(phase)
                x_[self._cosine_name(feature)] = np.cos(phase)
            else:
                x_[feature] = _NON_PERIODIC_FEATURES[feature](ind)

        return x_[self.feature_names_out_].copy()

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """We drop the added columns and return the original DataFrame."""
        return x[self.feature_names_in_]

    def __sklearn_is_fitted__(self) -> bool:
        """
        Check fitted status and return a Boolean value.

        Returns:
            bool: True if the transformer has been fitted, False otherwise.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns the list of feature names in the output."""
        check_is_fitted(self)
        return self.feature_names_out_

    @staticmethod
    def _sine_name(feature: str) -> str:
        """Generates the column name for the sine transformation of a feature."""
        return f"{feature}_sine"

    @staticmethod
    def _cosine_name(feature: str) -> str:
        """Generates the column name for the cosine transformation of a feature."""
        return f"{feature}_cosine"


__all__ = ["TimestampTransformer"]
