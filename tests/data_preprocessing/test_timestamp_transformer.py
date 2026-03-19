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


if __name__ == "__main__":
    unittest.main()
