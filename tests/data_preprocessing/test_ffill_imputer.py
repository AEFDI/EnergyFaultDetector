import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.exceptions import NotFittedError

from energy_fault_detector.data_preprocessing.ffill_imputer import ForwardFillImputer


class TestForwardFillImputer(unittest.TestCase):

    def setUp(self):
        """Set up common test fixtures."""
        # Create datetime index with 1-minute frequency
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        self.timestamps = pd.date_range(start=base_time, periods=10, freq='1Min')
        
        # Create numeric data with some NaNs
        self.data_numeric = pd.DataFrame({
            'temp': [20.0, np.nan, 22.0, np.nan, np.nan, 25.0, np.nan, np.nan, np.nan, 30.0],
            'humidity': [60.0, 61.0, np.nan, np.nan, 64.0, 65.0, np.nan, 67.0, 68.0, 69.0],
            'pressure': [1013.0, 1013.5, 1014.0, np.nan, np.nan, np.nan, 1015.5, 1016.0, np.nan, 1016.5]
        }, index=self.timestamps)
        
        # Create categorical data
        self.data_categorical = pd.DataFrame({
            'status': ['A', 'B', 'A', np.nan, 'A', 'B', 'A', 'A', 'B', 'A'],
            'region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South']
        }, index=self.timestamps)
        
        # Combined DataFrame
        self.data_full = pd.concat([self.data_numeric, self.data_categorical], axis=1)
        
        # Create non-declared categorical features in numeric columns (strings)
        self.data_mixed_categorical = self.data_numeric.copy()
        self.data_mixed_categorical['category_col'] = ['cat1', 'cat2', 'cat1', 'cat3', 'cat2', 'cat1', 'cat3', 'cat1', 'cat2', 'cat1']

        # Create full dataframe containing non-declared categorical features
        self.data_full_with_non_declared = pd.concat([self.data_mixed_categorical, self.data_categorical], axis=1)

    def tearDown(self):
        """Clean up after each test."""
        del self.timestamps, self.data_numeric, self.data_categorical, self.data_full, self.data_mixed_categorical, self.data_full_with_non_declared

    def test_init_defaults(self):
        """Test default initialization."""
        imputer = ForwardFillImputer()
        self.assertEqual(imputer.ffill_limit, "15min")
        self.assertEqual(imputer.categorical_features, [])

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        imputer = ForwardFillImputer(ffill_limit="10min", categorical_features=["status", "region"])
        self.assertEqual(imputer.ffill_limit, "10min")
        self.assertEqual(imputer.categorical_features, ["status", "region"])

    def test_fit(self):
        """Test fitting the imputer."""
        imputer = ForwardFillImputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_full_with_non_declared)
        
        self.assertEqual(imputer.n_features_in_, 6)  # 3 numeric + 1 non-declared categorical + 2 declared categorical
        self.assertListEqual(imputer.feature_names_in_, ['temp', 'humidity', 'pressure', 'category_col', 'status', 'region'])
        self.assertListEqual(imputer.numerical_columns, ['temp', 'humidity', 'pressure'])
        self.assertListEqual(imputer.categorical_columns, ['status', 'region'])
        self.assertEqual(len(imputer.non_declared_categorical_features), 1)

    def test_fit_numerical_only(self):
        """Test fitting on numerical-only data."""
        imputer = ForwardFillImputer()
        imputer.fit(self.data_numeric)
        
        self.assertEqual(imputer.n_features_in_, 3)
        self.assertListEqual(imputer.feature_names_in_, ['temp', 'humidity', 'pressure'])
        self.assertListEqual(imputer.numerical_columns, ['temp', 'humidity', 'pressure'])
        self.assertEqual(len(imputer.categorical_columns), 0)
        self.assertEqual(len(imputer.non_declared_categorical_features), 0)

    def test_fit_with_categorical(self):
        """Test fitting with categorical features declared."""
        imputer = ForwardFillImputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_full)
        
        self.assertEqual(imputer.n_features_in_, 5)
        self.assertListEqual(imputer.numerical_columns, ['temp', 'humidity', 'pressure'])
        self.assertListEqual(imputer.categorical_columns, ['status', 'region'])
        self.assertEqual(len(imputer.non_declared_categorical_features), 0)

    def test_fit_non_declared_categorical_in_numerical_columns(self):
        """Test detection of non-declared categorical features."""
        imputer = ForwardFillImputer()
        imputer.fit(self.data_mixed_categorical)
        
        # Should detect 'category_col' as non-declared categorical
        self.assertIn('category_col', imputer.non_declared_categorical_features)
        self.assertNotIn('category_col', imputer.numerical_columns)
        self.assertEqual(imputer.numerical_columns, ['temp', 'humidity', 'pressure'])

    def test_transform_basic(self):
        """Test basic forward fill transformation."""
        imputer = ForwardFillImputer(ffill_limit="10min")
        imputer.fit(self.data_numeric)
        result = imputer.transform(self.data_numeric)
        
        # Should not have any NaNs
        self.assertFalse(result.isna().any().any())
        # Should maintain same columns
        self.assertEqual(result.shape[1], 3)
        # Should have same column names
        self.assertListEqual(list(result.columns), ['temp', 'humidity', 'pressure'])

    def test_transform_with_categorical(self):
        """Test transformation with categorical features."""
        imputer = ForwardFillImputer(categorical_features=['status', 'region'], ffill_limit="10min")
        imputer.fit(self.data_full)
        result = imputer.transform(self.data_full)
        
        # Should not have any NaNs
        self.assertFalse(result.isna().any().any())
        # Should maintain same columns
        self.assertEqual(result.shape[1], 5)
        # Should preserve categorical data
        self.assertListEqual(list(result.columns), ['temp', 'humidity', 'pressure', 'status', 'region'])
        # Check that status column is preserved (with some imputed)
        self.assertEqual(result['status'].iloc[3], 'A')  # Should be forward-filled

    def test_transform_limit_exceeded(self):
        """Test that values beyond ffill_limit are not filled."""
        # Create data with long gaps (> ffill_limit)
        long_gaps_data = pd.DataFrame({
            'value1': [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.0],
            'value2': list(range(10, 27))
        }, index=pd.date_range('2024-01-01', periods=17, freq='1Min'))
        
        imputer = ForwardFillImputer(ffill_limit="5min")
        imputer.fit(long_gaps_data)
        result = imputer.transform(long_gaps_data)
        print(result)
        # The 7th row should be 2.0 because elapsed time (6min) exceeds the 5min limit
        self.assertTrue(result.iloc[6]['value1']==2.0)

    def test_transform_drops_duplicates(self):
        """Test that duplicate rows are removed."""
        # Create data with duplicate rows
        dup_data = pd.DataFrame({
            'value': [1.0, 2.0, 2.0, 3.0]
        }, index=pd.date_range('2024-01-01', periods=4, freq='1Min'))
        
        imputer = ForwardFillImputer()
        imputer.fit(dup_data)
        result = imputer.transform(dup_data)
        
        # Should have fewer rows after deduplication
        self.assertLess(result.shape[0], dup_data.shape[0])

    def test_transform_drops_rows_with_any_na(self):
        """Test that rows with any NaN are dropped after imputation."""
        # Create data where after ffill, some rows still have NaN
        some_na_data = pd.DataFrame({
            'value1': [1.0, np.nan, np.nan, 3.0, np.nan],
            'value2': [np.nan, 2.0, 3.0, 4.0, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        imputer = ForwardFillImputer(ffill_limit="1min")
        imputer.fit(some_na_data)
        result = imputer.transform(some_na_data)
        
        # Should have dropped rows with any remaining NaNs
        self.assertFalse(result.isna().any().any())
        self.assertEqual(result.shape[0], 3)  # Should keep three rows

    def test_transform_preserves_index_type(self):
        """Test that index type is preserved after transformation."""
        imputer = ForwardFillImputer()
        imputer.fit(self.data_numeric)
        result = imputer.transform(self.data_numeric)
        
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_transform_non_datetime_index_raises(self):
        """Test that non-datetime index raises TypeError."""
        imputer = ForwardFillImputer()
        imputer.fit(self.data_numeric)
        
        # Create DataFrame with non-datetime index
        bad_index_data = self.data_numeric.copy()
        bad_index_data.index = range(len(bad_index_data))
        
        with self.assertRaises(TypeError) as context:
            imputer.transform(bad_index_data)
        self.assertIn("DatetimeIndex", str(context.exception))

    def test_transform_missing_columns_raises(self):
        """Test that missing columns raise ValueError."""
        imputer = ForwardFillImputer()
        imputer.fit(self.data_numeric)
        
        # Drop one column
        partial_data = self.data_numeric.drop(columns=['temp'])
        
        with self.assertRaises(ValueError) as context:
            imputer.transform(partial_data)
        self.assertIn("missing columns", str(context.exception).lower())

    def test_transform_invalid_type_raises(self):
        """Test that non-DataFrame input raises TypeError."""
        imputer = ForwardFillImputer()
        imputer.fit(self.data_numeric)
        
        with self.assertRaises(TypeError) as context:
            imputer.transform([1, 2, 3])
        self.assertIn("pandas DataFrame", str(context.exception))

    def test_transform_unfitted_raises(self):
        """Test that transform on unfitted model raises error."""
        imputer = ForwardFillImputer()
        
        with self.assertRaises(NotFittedError) as context:
            imputer.transform(self.data_numeric)
        self.assertIn("not fitted", str(context.exception).lower())

    def test_inverse_transform(self):
        """Test inverse transform maintains column order."""
        imputer = ForwardFillImputer(categorical_features=['status', 'region'], ffill_limit="10min")
        imputer.fit(self.data_full)
        transformed = imputer.transform(self.data_full)
        inverse = imputer.inverse_transform(transformed)
        
        # Should match original column names
        self.assertListEqual(list(inverse.columns), self.data_full.columns.tolist())
        self.assertEqual(inverse.shape[1], 5)

    def test_get_feature_names_out(self):
        """Test feature names out method."""
        imputer = ForwardFillImputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_full)
        
        feature_names = imputer.get_feature_names_out()
        self.assertEqual(len(feature_names), 5)
        self.assertIn('temp', feature_names)
        self.assertIn('humidity', feature_names)
        self.assertIn('pressure', feature_names)
        self.assertIn('status', feature_names)
        self.assertIn('region', feature_names)

    def test_empty_dataframe_handling(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=self.data_numeric.columns, 
                               index=pd.DatetimeIndex([]))
        
        imputer = ForwardFillImputer()
        imputer.fit(self.data_numeric)
        
        result = imputer.transform(empty_df)
        
        self.assertEqual(result.shape[0], 0)

    def test_all_nan_column_handling(self):
        """Test that all-NaN columns are dropped during fit."""
        all_nan_data = pd.DataFrame({
            'value1': [1.0, 2.0, 3.0],
            'value2': [np.nan, np.nan, np.nan]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1Min'))
        
        imputer = ForwardFillImputer()
        imputer.fit(all_nan_data)

        # All-NaN column should be dropped from numerical_columns
        self.assertNotIn('value2', imputer.numerical_columns)
        self.assertIn('value1', imputer.numerical_columns)

        # Transform should keep rows from the valid column, not drop everything
        result = imputer.transform(all_nan_data)
        self.assertEqual(result.shape[1], 1)
        self.assertListEqual(list(result.columns), ['value1'])
        self.assertEqual(result.shape[0], 3)

    def test_numerical_conversion_error(self):
        """Test error handling for non-convertible numerical columns."""

        
        imputer = ForwardFillImputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_full)

        # Add non-numeric column to numeric section
        bad_numeric = self.data_full.copy()
        bad_numeric['temp'] = ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        
        # Try transform - should raise error because 'bad_numeric' cannot convert to float
        with self.assertRaises(ValueError) as context:
            imputer.transform(bad_numeric)
        self.assertIn("cannot be converted to float", str(context.exception))

    def test_feature_names_out_after_fit(self):
        """Test feature names out are available after fit."""
        imputer = ForwardFillImputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_full)
        
        feature_names = imputer.get_feature_names_out()
        self.assertEqual(len(feature_names), imputer.n_features_in_)


if __name__ == '__main__':
    unittest.main()