
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer


class CounterDiffTransformer(DataTransformer):
    """
    Transform monotonic counter columns into per-sample increments (default) or per-second rates (if compute_rate=True),
    handling resets/rollovers and masking long time gaps.

    It handles counter resets/rollovers and optionally masks values after large time gaps, which helps avoid misleading
    diffs/rates caused by missing data.

    Args:
        counters: List of counter column names to transform.
        compute_rate: If True, output per-second rates (increment / dt). If False (default),
            output per-sample increments.
        reset_strategy: One of {'zero', 'rollover', 'nan', 'auto'}:

            - 'zero' (default): if diff < 0, treat as reset-to-zero; increment = current_value.
            - 'rollover': if diff < 0, increment = current_value + (rollover_value - previous_value).
            - 'nan': if diff < 0, set increment to NaN.
            - 'auto': use 'rollover' if rollover_values contains the counter; otherwise 'zero'.

        rollover_values: Optional mapping counter -> known max value (used by 'rollover' or 'auto').
        small_negative_tolerance: Treat small negative diffs (``abs(diff) <= tol``) as 0 (noise). Default: 0.0.
        fill_first: One of {'nan', 'zero'}. How to fill the first sample where diff is undefined.
        keep_original: If True, keep original counters alongside new outputs. If False, drop them.
        gap_policy: One of {'mask', 'ignore'}:

            - 'mask' (default): set output to NaN for rows where time delta > threshold.
            - 'ignore': do nothing special for large gaps.

        max_gap_seconds: Explicit threshold (in seconds) for gap masking. If provided, overrides
            max_gap_factor.
        max_gap_factor: If max_gap_seconds is None, use threshold = factor * median(dt).
            Default is 3.0.

    Notes:
        - A DatetimeIndex is required if compute_rate=True or gap_policy='mask'.
        - The inverse_transform is a no-op and returns the input unchanged.

    Examples:
        - Diffs: [0, 1, 3, 0 (reset), 2] -> [NaN|0, 1, 2, 0|NaN, 2]
        - Rates: increment / dt (in seconds), with large-gap rows optionally masked to NaN.
    """

    def __init__(
        self,
        counters: List[str],
        compute_rate: bool = False,
        reset_strategy: str = "zero",
        rollover_values: Optional[Dict[str, float]] = None,
        small_negative_tolerance: float = 0.0,
        fill_first: str = "nan",
        keep_original: bool = False,
        gap_policy: str = "mask",
        max_gap_seconds: Optional[float] = None,
        max_gap_factor: float = 3.0,
    ) -> None:
        super().__init__()
        self.counters = counters or []
        self.compute_rate = compute_rate
        self.reset_strategy = reset_strategy
        self.rollover_values = rollover_values or {}
        self.small_negative_tolerance = float(small_negative_tolerance)
        self.fill_first = fill_first
        self.keep_original = keep_original
        self.gap_policy = gap_policy
        self.max_gap_seconds = max_gap_seconds
        self.max_gap_factor = float(max_gap_factor)

    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> "CounterDiffTransformer":
        """Validate inputs and compute output schema.

        This method validates the time index (when needed), stores the list of counters that are
        present in the input, and computes the output column layout such that transform() can
        reproduce the same order deterministically.

        Args:
            x: Input DataFrame. Requires a DatetimeIndex if compute_rate=True or gap_policy='mask'.
            y: Unused. Present for estimator interface compatibility.

        Returns:
            self

        Raises:
            ValueError: If a DatetimeIndex is required but missing or non-monotonic.
        """
        self.feature_names_in_ = x.columns.to_list()
        self.n_features_in_ = len(x.columns)

        # DatetimeIndex is required for rates or for gap masking
        if self.compute_rate or self.gap_policy == "mask":
            if not isinstance(x.index, pd.DatetimeIndex):
                raise ValueError(
                    "CounterDiffTransformer: DatetimeIndex required (rate or gap masking)."
                )
            if not x.index.is_monotonic_increasing:
                raise ValueError("CounterDiffTransformer: index must be monotonic increasing.")

        # Keep only counters present in the DataFrame
        self.counters_ = [c for c in self.counters if c in self.feature_names_in_]

        # Determine output suffix
        self.output_suffix_ = "_rate" if self.compute_rate else "_diff"

        # Compose output feature order
        new_cols = [f"{c}{self.output_suffix_}" for c in self.counters_]
        if self.keep_original:
            # Append new output columns after all original features
            self.feature_names_out_ = list(self.feature_names_in_) + new_cols
        else:
            # Keep non-counter features first, then the new output columns
            others = [col for col in self.feature_names_in_ if col not in self.counters_]
            self.feature_names_out_ = others + new_cols

        # Track columns dropped when keep_original is False (for introspection/testing)
        self.columns_dropped_ = [] if self.keep_original else [c for c in self.counters_]
        return self

    def _time_deltas_seconds(self, x: pd.DataFrame) -> Optional[pd.Series]:
        """Compute per-row time delta in seconds, or None if not needed.

        Returns NaN for the first row and when dt is 0 seconds (zero dt is masked to NaN to avoid
        division by zero for rate calculations).

        Args:
            x: Input DataFrame.

        Returns:
            A Series of dt seconds aligned to x.index, or None if neither rate nor masking is used.

        Raises:
            ValueError: If a DatetimeIndex is required but missing or non-monotonic.
        """
        if not (self.compute_rate or self.gap_policy == "mask"):
            return None
        if not isinstance(x.index, pd.DatetimeIndex):
            raise ValueError("CounterDiffTransformer: DatetimeIndex required for rate or gap masking.")
        if not x.index.is_monotonic_increasing:
            raise ValueError("CounterDiffTransformer: index must be monotonic increasing.")

        # Create a series of timestamps to keep the original index for alignment
        dt = pd.Series(x.index, index=x.index).diff().dt.total_seconds()
        # Prevent division by zero when computing rates
        dt = dt.mask(dt == 0, np.nan)
        return dt

    def _gap_threshold(self, dt: pd.Series) -> Optional[float]:
        """Compute the gap masking threshold in seconds, or None if masking disabled.

        Args:
            dt: Series of time deltas in seconds.

        Returns:
            Threshold in seconds or None if masking is not applicable. If max_gap_seconds is given,
            it is used; otherwise threshold = max_gap_factor * median(dt). If median is not finite
            or <= 0, returns None and masking is effectively disabled.
        """
        if self.gap_policy != "mask" or dt is None:
            return None
        if self.max_gap_seconds is not None:
            return float(self.max_gap_seconds)

        med = float(np.nanmedian(dt.values)) if len(dt) else np.nan
        if not np.isfinite(med) or med <= 0:
            return None
        return self.max_gap_factor * med

    def _compute_increment(
        self,
        s: pd.Series,
        strategy: str,
        rollover_value: Optional[float],
    ) -> pd.Series:
        """Compute per-sample increment for a counter series with reset handling.

        This applies the selected reset strategy to negative diffs and also clamps small negative
        diffs (within small_negative_tolerance) to zero to mitigate minor noise/clock skew.

        Args:
            s: Input counter Series.
            strategy: Reset strategy ('zero', 'rollover', 'nan', 'auto').
            rollover_value: Known rollover maximum (used by 'rollover' or 'auto').

        Returns:
            Series of increments aligned to s.index, with the first element filled according to
            fill_first ('zero' or 'nan').

        Raises:
            ValueError: If series contains non-numeric values (excluding existing NaNs),
                or if strategy is 'rollover' but rollover_value is None,
                or if an unknown reset strategy is provided.
        """
        # Try to coerce to numeric; if this introduces new NaNs (beyond existing ones), error out
        v = pd.to_numeric(s, errors="coerce")
        if v.isna().sum() > s.isna().sum():
            raise ValueError(
                "CounterDiffTransformer: non-numeric values found in counter series. "
                "Ensure all counter values are numeric or NaN."
            )
        prev = v.shift(1)
        diff = v.diff()

        # Clamp small negative diffs to zero (treat as noise)
        if self.small_negative_tolerance > 0:
            small_neg = (diff < 0) & ((-diff) <= self.small_negative_tolerance)
            diff = diff.mask(small_neg, 0.0)

        neg_mask = diff < 0

        # Map 'auto' to a concrete strategy
        if strategy == "auto":
            strategy = "rollover" if rollover_value is not None else "zero"

        if strategy == "nan":
            inc = diff.mask(neg_mask, np.nan)
        elif strategy == "zero":
            # Treat reset-to-zero as increment equals current value
            inc = diff.where(~neg_mask, v)
        elif strategy == "rollover":
            if rollover_value is None:
                # Explicit 'rollover' requires a value.
                raise ValueError(
                    "CounterDiffTransformer: rollover strategy requires a rollover_value for the "
                    f"counter '{s.name}'. Use reset_strategy='auto' to fallback to 'zero' when not provided."
                )
            # Add the wrapped amount: current + (rollover - previous)
            inc = diff.where(~neg_mask, v + (rollover_value - prev))
        else:
            raise ValueError(f"CounterDiffTransformer: unknown reset_strategy '{strategy}'")

        return inc

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform counters into diffs or rates, with optional gap masking.

        For each configured counter:
          1) Compute per-sample increment with reset handling.
          2) If compute_rate=True, divide by dt seconds.
          3) If gap_policy='mask', set values to NaN where dt > gap_threshold.

        Args:
            x: Input DataFrame. Requires a DatetimeIndex if compute_rate=True or gap_policy='mask'.

        Returns:
            A DataFrame with transformed columns appended (if keep_original=True) or replacing the
            original counters (if keep_original=False). Column order matches fit()'s schema.

        Raises:
            ValueError: If DatetimeIndex is required but missing or non-monotonic.
        """
        check_is_fitted(self)
        x_ = x.copy()

        dt = self._time_deltas_seconds(x_)
        gap_thr = self._gap_threshold(dt) if dt is not None else None

        new_cols = {}
        for c in self.counters_:
            increment = self._compute_increment(
                x_[c], strategy=self.reset_strategy, rollover_value=self.rollover_values.get(c)
            )
            series = (increment / dt) if self.compute_rate and dt is not None else increment

            # Ensure first sample respects fill_first setting
            series.iloc[0] = 0.0 if self.fill_first == "zero" else np.nan

            # Mask large gaps for both diffs and rates to avoid misleading values
            if gap_thr is not None:
                series = series.mask(dt > gap_thr)

            new_cols[f"{c}{self.output_suffix_}"] = series

        # Attach new columns
        for name, col in new_cols.items():
            x_[name] = col

        # Optionally remove original counter columns
        if not self.keep_original:
            x_ = x_.drop(columns=self.counters_, errors='ignore')

        # Reorder to the schema established during fit
        x_ = x_[self.feature_names_out_]
        return x_

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """If original counter columns are present, drop the derived columns and restore original feature order.
        Otherwise, returns the input as is.

        Args:
            x: Input DataFrame.

        Returns:
            The input DataFrame unchanged.
        """
        check_is_fitted(self)
        x_ = x.copy()
        orig_counters_present = all(c in x_.columns for c in self.counters_)
        if orig_counters_present:
            if all(col in x_.columns for col in self.feature_names_in_):
                x_ = x_[self.feature_names_in_]
            return x_
        return x

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return the output feature names determined in fit().

        Args:
            input_features: Unused. Present for compatibility with sklearn API.

        Returns:
            List of output column names.
        """
        check_is_fitted(self)
        return self.feature_names_out_
