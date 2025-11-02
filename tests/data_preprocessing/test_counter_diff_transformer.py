import unittest
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from energy_fault_detector.data_preprocessing.counter_diff_transformer import CounterDiffTransformer


class TestCounterDiffTransformer(unittest.TestCase):
    """Unit tests for CounterDiffTransformer."""

    def setUp(self) -> None:
        """Create small helper datasets used across tests."""
        # Regular 1-second interval index
        self.t0 = datetime(2024, 1, 1, 0, 0, 0)
        self.idx_1s = pd.date_range(self.t0, periods=5, freq="1s", tz="UTC")

    def _df(
        self,
        values_a: List[float],
        values_b: List[float] | None = None,
        index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """Helper to build a DataFrame with optional second counter."""
        index = index if index is not None else self.idx_1s
        if len(values_a) < len(index):
            index = index[:len(values_a)]
        data = {"counter_a": values_a}
        if values_b is not None:
            data["counter_b"] = values_b
        return pd.DataFrame(data, index=index)

    def test_fit_requires_datetime_index_when_rate_or_mask(self) -> None:
        """fit should error on non-DatetimeIndex when rate/mask are requested."""
        df = pd.DataFrame({"counter_a": [0, 1, 2]}, index=[0, 1, 2])

        # compute_rate=True requires DatetimeIndex
        with self.assertRaises(ValueError):
            CounterDiffTransformer(counters=["counter_a"], compute_rate=True).fit(df)

        # gap_policy='mask' requires DatetimeIndex
        with self.assertRaises(ValueError):
            CounterDiffTransformer(counters=["counter_a"], gap_policy="mask").fit(df)

        # If neither rate nor mask, fit should succeed
        CounterDiffTransformer(counters=["counter_a"], gap_policy="ignore").fit(df)

    def test_fit_requires_monotonic_index(self) -> None:
        """fit should error on non-monotonic DatetimeIndex when rate/mask are requested."""
        idx = pd.DatetimeIndex(
            [self.t0, self.t0 + timedelta(seconds=2), self.t0 + timedelta(seconds=1)],
            tz="UTC",
        )
        df = self._df(values_a=[0, 1, 2], index=idx)
        with self.assertRaises(ValueError):
            CounterDiffTransformer(counters=["counter_a"], compute_rate=True).fit(df)

        # No problem if index is sorted
        CounterDiffTransformer(counters=["counter_a"], compute_rate=True).fit(df.sort_index())

    def test_diff_zero_strategy_default(self) -> None:
        """Default 'zero' strategy: negative diff -> increment equals current value."""
        # 0 -> 1 -> 3 -> 0 (reset) -> 2
        df = self._df(values_a=[0, 1, 4, 1, 3])

        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            reset_strategy="zero",
            fill_first="nan",
            keep_original=False,
            gap_policy="ignore",
        ).fit(df)

        out = tr.transform(df)
        self.assertListEqual(list(out.columns), ["counter_a_diff"])

        expected = pd.Series([np.nan, 1, 3, 1, 2], index=df.index, name="counter_a_diff")
        pd.testing.assert_series_equal(out["counter_a_diff"], expected, check_dtype=False)

    def test_diff_fill_first_zero(self) -> None:
        """First increment filled with zero when fill_first='zero'."""
        df = self._df(values_a=[5, 7, 8, 10, 12])

        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            reset_strategy="zero",
            fill_first="zero",
            keep_original=False,
            gap_policy="ignore",
        ).fit(df)

        out = tr.transform(df)
        expected = pd.Series([0, 2, 1, 2, 2], index=df.index, name="counter_a_diff")
        pd.testing.assert_series_equal(out["counter_a_diff"], expected, check_dtype=False)

    def test_rollover_strategy_with_value(self) -> None:
        """'rollover' strategy uses provided rollover value to compute increment."""
        # 95 -> 98 -> 2 (rollover at 100) => inc: NaN/0, 3, 2 + (100 - 98) = 4
        df = self._df(values_a=[95, 98, 2, 7, 20])

        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            reset_strategy="rollover",
            rollover_values={"counter_a": 100.0},
            fill_first="zero",
            keep_original=False,
            gap_policy="ignore",
        ).fit(df)

        out = tr.transform(df)
        expected = pd.Series([0, 3, 4, 5, 13], index=df.index, name="counter_a_diff")
        pd.testing.assert_series_equal(out["counter_a_diff"], expected, check_dtype=False)

    def test_rollover_strategy_without_value_errors(self) -> None:
        """'rollover' without a rollover_value should raise a ValueError."""
        df = self._df(values_a=[50, 10])  # negative diff
        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            reset_strategy="rollover",
            rollover_values={},  # missing
            fill_first="zero",
            keep_original=False,
            gap_policy="ignore",
        ).fit(df)
        with self.assertRaises(ValueError):
            tr.transform(df)

    def test_nan_strategy(self) -> None:
        """'nan' strategy sets negative diffs to NaN."""
        df = self._df(values_a=[10, 8, 9])
        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            reset_strategy="nan",
            fill_first="zero",
            keep_original=False,
            gap_policy="ignore",
        ).fit(df)
        out = tr.transform(df)
        expected = pd.Series([0, np.nan, 1], index=df.index, name="counter_a_diff")
        pd.testing.assert_series_equal(out["counter_a_diff"], expected, check_dtype=False)

    def test_auto_strategy_prefers_rollover_when_available(self) -> None:
        """'auto' uses rollover if a value is supplied; else behaves like 'zero'."""
        df = self._df(values_a=[95, 98, 2])

        # With rollover value -> like 'rollover'
        tr1 = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            reset_strategy="auto",
            rollover_values={"counter_a": 100.0},
            fill_first="zero",
            gap_policy="ignore",
        ).fit(df)
        out1 = tr1.transform(df)
        expected1 = pd.Series([0, 3, 4], index=df.index, name="counter_a_diff")
        pd.testing.assert_series_equal(out1["counter_a_diff"], expected1, check_dtype=False)

        # Without rollover value -> like 'zero'
        tr2 = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            reset_strategy="auto",
            rollover_values={},  # none
            fill_first="zero",
            gap_policy="ignore",
        ).fit(df)
        out2 = tr2.transform(df)
        expected2 = pd.Series([0, 3, 2], index=df.index, name="counter_a_diff")
        pd.testing.assert_series_equal(out2["counter_a_diff"], expected2, check_dtype=False)

    def test_small_negative_tolerance(self) -> None:
        """Small negative diff within tolerance is clamped to zero."""
        df = self._df(values_a=[10.0, 9.9995, 10.5])
        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            reset_strategy="zero",
            small_negative_tolerance=0.01,
            fill_first="zero",
            gap_policy="ignore",
        ).fit(df)
        out = tr.transform(df)
        # diff: 0, -0.0005 (-> 0), 0.5005
        expected = pd.Series([0.0, 0.0, 0.5005], index=df.index, name="counter_a_diff")
        pd.testing.assert_series_equal(out["counter_a_diff"], expected)

    def test_compute_rate(self) -> None:
        """Rate equals increment divided by dt seconds."""
        idx = pd.DatetimeIndex(
            [self.t0, self.t0 + timedelta(seconds=2), self.t0 + timedelta(seconds=5)],
            tz="UTC",
        )
        df = self._df(values_a=[0, 4, 7], index=idx)

        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=True,
            reset_strategy="zero",
            fill_first="zero",
            gap_policy="ignore",
        ).fit(df)
        out = tr.transform(df)
        # increments: [0, 4, 3]; dt: [NaN, 2, 3]; rate: [0, 2, 1]
        expected = pd.Series([0.0, 2.0, 1.0], index=df.index, name="counter_a_rate")
        pd.testing.assert_series_equal(out["counter_a_rate"], expected)

    def test_gap_masking_with_max_gap_seconds(self) -> None:
        """Values at positions where dt > threshold should be masked (NaN)."""
        idx = pd.DatetimeIndex(
            [
                self.t0,
                self.t0 + timedelta(seconds=1),
                self.t0 + timedelta(seconds=10),  # big gap from previous
                self.t0 + timedelta(seconds=11),
            ],
            tz="UTC",
        )
        df = self._df(values_a=[0, 1, 2, 3], index=idx)

        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            gap_policy="mask",
            max_gap_seconds=8.0,  # gap = 9 seconds
            fill_first="zero",
        ).fit(df)

        out = tr.transform(df)
        # increments: [0,1,1,1]; dt: [NaN,1,9,1]; mask where dt>5 -> index 2
        self.assertTrue(np.isnan(out["counter_a_diff"].iloc[2]))
        self.assertEqual(out["counter_a_diff"].iloc[1], 1.0)
        self.assertEqual(out["counter_a_diff"].iloc[3], 1.0)

    def test_gap_masking_with_factor_median(self) -> None:
        """Threshold computed as factor * median(dt)."""
        idx = pd.DatetimeIndex(
            [
                self.t0,
                self.t0 + timedelta(seconds=2),
                self.t0 + timedelta(seconds=4),
                self.t0 + timedelta(seconds=20),  # gap 16 > factor*median (median=2)
            ],
            tz="UTC",
        )
        df = self._df(values_a=[0, 2, 3, 5], index=idx)

        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            gap_policy="mask",
            max_gap_seconds=None,
            max_gap_factor=3.0,  # 3 * median = 6
            fill_first="zero",
        ).fit(df)

        out = tr.transform(df)
        self.assertTrue(np.isnan(out["counter_a_diff"].iloc[3]))  # masked at data gap
        self.assertEqual(out["counter_a_diff"].iloc[1], 2.0)
        self.assertEqual(out["counter_a_diff"].iloc[2], 1.0)

    def test_gap_policy_ignore(self) -> None:
        """No masking when gap_policy='ignore'."""
        idx = pd.DatetimeIndex(
            [self.t0, self.t0 + timedelta(seconds=1), self.t0 + timedelta(seconds=10)],
            tz="UTC",
        )
        df = self._df(values_a=[0, 1, 30], index=idx)

        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            gap_policy="ignore",
            fill_first="zero",
        ).fit(df)

        out = tr.transform(df)
        expected = pd.Series([0, 1, 29], index=df.index, name="counter_a_diff")
        pd.testing.assert_series_equal(out["counter_a_diff"], expected, check_dtype=False)

    def test_keep_original_false_drops_counters(self) -> None:
        """When keep_original=False, original counters are dropped from output."""
        df = self._df(values_a=[0, 1, 2], values_b=[0, 10, 20])

        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            keep_original=False,
            gap_policy="ignore",
            fill_first="zero",
        ).fit(df)

        out = tr.transform(df)
        # 'counter_b' should be kept, 'counter_a' replaced by 'counter_a_diff'
        self.assertListEqual(list(out.columns), ["counter_b", "counter_a_diff"])

    def test_keep_original_true_keeps_counters(self) -> None:
        """When keep_original=True, original counters remain alongside outputs."""
        df = self._df(values_a=[0, 1, 2], values_b=[0, 10, 20])

        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            keep_original=True,
            gap_policy="ignore",
            fill_first="zero",
        ).fit(df)

        out = tr.transform(df)
        self.assertListEqual(list(out.columns), ["counter_a", "counter_b", "counter_a_diff"])

    def test_feature_names_out(self) -> None:
        """get_feature_names_out returns correct output ordering."""
        df = self._df(values_a=[0, 1, 2], values_b=[0, 10, 20])
        tr = CounterDiffTransformer(
            counters=["counter_a", "missing_counter"],
            compute_rate=False,
            keep_original=False,
            gap_policy="ignore",
            fill_first="zero",
        ).fit(df)

        # Only present counters are transformed; others ignored
        self.assertEqual(tr.counters_, ["counter_a"])
        self.assertEqual(tr.get_feature_names_out(), ["counter_b", "counter_a_diff"])

        out = tr.transform(df)
        self.assertListEqual(tr.get_feature_names_out(), list(out.columns))

    def test_non_numeric_values_raise_error(self) -> None:
        """Non-numeric values should be coerced to NaN then diff computed."""
        df = self._df(values_a=[0, "1", "3", "bad", 7])  # 'bad' -> NaN
        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            gap_policy="ignore",
            fill_first="zero",
        ).fit(df)

        with self.assertRaises(ValueError):
            tr.transform(df)

    def test_inverse_transform(self) -> None:
        """inverse_transform returns input unchanged."""
        df = self._df(values_a=[0, 1, 2])
        tr = CounterDiffTransformer(
            counters=["counter_a"],
            compute_rate=False,
            gap_policy="ignore",
            fill_first="zero",
        ).fit(df)
        out = tr.transform(df)
        back = tr.inverse_transform(out.copy())
        pd.testing.assert_frame_equal(out, back)


if __name__ == "__main__":
    unittest.main()
