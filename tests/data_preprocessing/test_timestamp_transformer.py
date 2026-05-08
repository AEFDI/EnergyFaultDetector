import unittest
import numpy as np
import pandas as pd

from energy_fault_detector.data_preprocessing.timestamp_transformer import TimestampTransformer


class TestTimestampTransformer(unittest.TestCase):

    def setUp(self):
        idx = pd.date_range("2024-02-28 23:59:30", periods=4, freq="30s")
        df = pd.DataFrame(
            {
                "sensor1": [1.0, 2.0, 3.0, 4.0],
                "sensor2": [10.0, 20.0, 30.0, 40.0],
            },
            index=idx,
        )
        self.df = df
        self.df_with_ts_col = self.df.reset_index().rename(columns={"index": "timestamp"})

    def test_fit_with_index_and_valid_features(self):
        tt = TimestampTransformer(
            features=[
                "second_of_minute",
                "minute_of_hour",
                "hour_of_day",
                "day_of_week",
                "day_of_month",
                "day_of_year",
                "month_of_year",
                "is_weekend",
                "year",
            ]
        )

        tt.fit(self.df)

        # input attributes
        self.assertEqual(tt.n_features_in_, self.df.shape[1])
        self.assertEqual(tt.feature_names_in_, list(self.df.columns))

        # periodic features -> two columns; non-periodic -> one
        expected_added = []
        for fname in tt.features:
            if fname in {
                "second_of_minute",
                "minute_of_hour",
                "hour_of_day",
                "day_of_week",
                "day_of_month",
                "day_of_year",
                "month_of_year",
            }:
                expected_added.append(f"{fname}_sine")
                expected_added.append(f"{fname}_cosine")
            else:
                expected_added.append(fname)

        self.assertEqual(tt.feature_names_added_, expected_added)
        self.assertEqual(
            tt.feature_names_out_, list(self.df.columns) + expected_added
        )

    def test_fit_raises_on_unknown_feature(self):
        tt = TimestampTransformer(features=["hour_of_day", "unknown_feature"])

        with self.assertRaises(ValueError) as cm:
            tt.fit(self.df)
        msg = str(cm.exception)
        self.assertIn("unknown features", msg)
        self.assertIn("unknown_feature", msg)

    def test_fit_uses_timestamp_column(self):
        tt = TimestampTransformer(features=["hour_of_day"], timestamp_col="timestamp")
        tt.fit(self.df_with_ts_col)

        self.assertEqual(tt.feature_names_in_, list(self.df_with_ts_col.columns))
        self.assertEqual(
            tt.feature_names_added_,
            ["hour_of_day_sine", "hour_of_day_cosine"],
        )

    def test_fit_raises_if_timestamp_column_missing(self):
        tt = TimestampTransformer(features=["hour_of_day"], timestamp_col="time")

        with self.assertRaises(ValueError) as cm:
            tt.fit(self.df_with_ts_col)
        self.assertIn("column 'time' not found", str(cm.exception))

    def test_fit_raises_if_index_not_datetime_and_no_timestamp_col(self):
        df = self.df_with_ts_col
        tt = TimestampTransformer(features=["hour_of_day"])

        with self.assertRaises(ValueError) as cm:
            tt.fit(df)
        self.assertIn("DataFrame index must be a DatetimeIndex", str(cm.exception))

    def test_transform_adds_expected_columns_and_values_index(self):
        tt = TimestampTransformer(
            features=["second_of_minute", "day_of_week", "is_weekend", "year"]
        )
        tt.fit(self.df)
        out = tt.transform(self.df)

        expected_cols = (
            list(self.df.columns)
            + ["second_of_minute_sine", "second_of_minute_cosine"]
            + ["day_of_week_sine", "day_of_week_cosine"]
            + ["is_weekend", "year"]

        )
        self.assertEqual(list(out.columns), expected_cols)
        self.assertEqual(len(out), len(self.df))

        ts0 = self.df.index[0]

        # second_of_minute
        phase_s = ts0.second / 60.0
        rad_s = 2 * np.pi * phase_s
        self.assertAlmostEqual(
            out.iloc[0]["second_of_minute_sine"], np.sin(rad_s), places=7
        )
        self.assertAlmostEqual(
            out.iloc[0]["second_of_minute_cosine"], np.cos(rad_s), places=7
        )

        # day_of_week
        phase_dow = ts0.day_of_week / 7.0
        rad_dow = 2 * np.pi * phase_dow
        self.assertAlmostEqual(
            out.iloc[0]["day_of_week_sine"], np.sin(rad_dow), places=7
        )
        self.assertAlmostEqual(
            out.iloc[0]["day_of_week_cosine"], np.cos(rad_dow), places=7
        )

        # is_weekend (2024-02-28 -> Thursday)
        self.assertEqual(out.iloc[0]["is_weekend"], 0.0)
        # year
        self.assertEqual(out.iloc[0]["year"], 2024.0)

    def test_transform_uses_timestamp_column_values(self):
        df = self.df_with_ts_col
        tt = TimestampTransformer(features=["hour_of_day"], timestamp_col="timestamp")
        tt.fit(df)
        out = tt.transform(df)

        self.assertIn("hour_of_day_sine", out.columns)
        self.assertIn("hour_of_day_cosine", out.columns)
        self.assertEqual(len(out), len(df))

        ts0 = df["timestamp"].iloc[0]
        phase = ts0.hour / 24.0
        rad = 2 * np.pi * phase
        self.assertAlmostEqual(out.iloc[0]["hour_of_day_sine"], np.sin(rad), places=7)
        self.assertAlmostEqual(
            out.iloc[0]["hour_of_day_cosine"], np.cos(rad), places=7
        )

    def test_inverse_transform_drops_added_columns(self):
        tt = TimestampTransformer(features=["hour_of_day", "is_weekend"])
        tt.fit(self.df)
        out = tt.transform(self.df)

        restored = tt.inverse_transform(out)

        # Only original columns
        self.assertEqual(list(restored.columns), list(self.df.columns))
        pd.testing.assert_frame_equal(
            restored.reset_index(drop=True), self.df.reset_index(drop=True)
        )

    def test_multiindex_auto_detection(self):
        """MultiIndex with groupby_level='auto' extracts time features correctly."""
        # Create MultiIndex: (device, timestamp)
        devices = ['device_a', 'device_b']
        times = pd.date_range("2024-03-15 10:30:00", periods=3, freq="1h")
        idx = pd.MultiIndex.from_product([devices, times], names=['device_id', 'timestamp'])

        df = pd.DataFrame({
            'sensor1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'sensor2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        }, index=idx)

        tt = TimestampTransformer(
            features=['hour_of_day', 'day_of_week'],
            groupby_level='auto'  # Auto-detect
        )
        tt.fit(df)

        # Check that auto-detection resolved to first non-datetime level
        self.assertEqual(tt.groupby_level_, 0)

        out = tt.transform(df)

        # Verify columns
        expected_cols = ['sensor1', 'sensor2', 'hour_of_day_sine', 'hour_of_day_cosine',
                        'day_of_week_sine', 'day_of_week_cosine']
        self.assertEqual(list(out.columns), expected_cols)

        # Verify values for first timestamp (10:30 AM)
        # hour_of_day uses only the hour component: hour / 24 = 10 / 24
        ts_first = times[0]
        phase_hour = ts_first.hour / 24.0
        rad_hour = 2 * np.pi * phase_hour
        self.assertAlmostEqual(out.iloc[0]['hour_of_day_sine'], np.sin(rad_hour), places=7)
        self.assertAlmostEqual(out.iloc[0]['hour_of_day_cosine'], np.cos(rad_hour), places=7)

        # Verify same time features for both devices at same timestamp
        # Rows 0 and 3 are same timestamp (10:30) for different devices
        self.assertAlmostEqual(out.iloc[0]['hour_of_day_sine'], out.iloc[3]['hour_of_day_sine'])
        self.assertAlmostEqual(out.iloc[0]['hour_of_day_cosine'], out.iloc[3]['hour_of_day_cosine'])

    def test_multiindex_explicit_groupby(self):
        """MultiIndex with explicit groupby_level extracts time features correctly."""
        # Create MultiIndex: (location, timestamp)
        locations = ['site_1', 'site_2']
        times = pd.date_range("2024-01-01 00:00:00", periods=2, freq="12h")
        idx = pd.MultiIndex.from_product([locations, times], names=['location', 'timestamp'])

        df = pd.DataFrame({
            'value': [100, 200, 300, 400]
        }, index=idx)

        tt = TimestampTransformer(
            features=['hour_of_day', 'is_weekend'],
            groupby_level='location'  # Explicit
        )
        tt.fit(df)

        self.assertEqual(tt.groupby_level_, 'location')

        out = tt.transform(df)

        # First row: midnight (hour=0, phase=0)
        self.assertAlmostEqual(out.iloc[0]['hour_of_day_sine'], 0.0, places=7)
        self.assertAlmostEqual(out.iloc[0]['hour_of_day_cosine'], 1.0, places=7)

        # Second row: noon (hour=12, phase=0.5)
        phase_noon = 12.0 / 24.0
        rad_noon = 2 * np.pi * phase_noon
        self.assertAlmostEqual(out.iloc[1]['hour_of_day_sine'], np.sin(rad_noon), places=7)
        self.assertAlmostEqual(out.iloc[1]['hour_of_day_cosine'], np.cos(rad_noon), places=7)

        # is_weekend for 2024-01-01 (Monday)
        self.assertEqual(out.iloc[0]['is_weekend'], 0.0)

    def test_multiindex_with_all_features(self):
        """MultiIndex works with all time features."""
        devices = ['dev_1', 'dev_2']
        times = pd.date_range("2024-02-29 23:59:00", periods=2, freq="2min")  # Leap year
        idx = pd.MultiIndex.from_product([devices, times], names=['device', 'time'])

        df = pd.DataFrame({'data': [1, 2, 3, 4]}, index=idx)

        tt = TimestampTransformer(
            features=[
                'second_of_minute',
                'minute_of_hour',
                'hour_of_day',
                'day_of_week',
                'day_of_month',
                'day_of_year',
                'month_of_year',
                'is_weekend',
                'year'
            ],
            groupby_level='auto'
        )
        tt.fit(df)
        out = tt.transform(df)

        # Verify all features are added
        self.assertIn('second_of_minute_sine', out.columns)
        self.assertIn('year', out.columns)
        self.assertIn('is_weekend', out.columns)

        # Verify day_of_year accounts for leap year (60/366)
        # 2024-02-29 is day 60 of 366
        ts = times[0]
        phase_doy = ts.dayofyear / 366.0
        rad_doy = 2 * np.pi * phase_doy
        self.assertAlmostEqual(out.iloc[0]['day_of_year_sine'], np.sin(rad_doy), places=7)

    def test_simple_index_with_auto_still_works(self):
        """Simple DatetimeIndex with groupby_level='auto' works (resolves to None)."""
        tt = TimestampTransformer(
            features=['hour_of_day', 'is_weekend'],
            groupby_level='auto'  # Should resolve to None
        )
        tt.fit(self.df)

        # Auto-detection should resolve to None for simple index
        self.assertIsNone(tt.groupby_level_)

        out = tt.transform(self.df)

        # Should work just like before
        self.assertIn('hour_of_day_sine', out.columns)
        self.assertIn('is_weekend', out.columns)
        self.assertEqual(len(out), len(self.df))


if __name__ == "__main__":
    unittest.main()
