import unittest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from energy_fault_detector.data_preprocessing.imputer import Imputer


class TestImputer(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        # Create sample numerical data with missing values
        self.data_numerical = pd.DataFrame({
            'temperature': [20.5, np.nan, 19.8, 22.1, 23.0],
            'humidity': [60, 62, np.nan, 65, 67],
            'pressure': [1013.2, 1014.5, 1012.8, np.nan, 1013.7]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with categorical features (object dtype)
        self.data_with_categorical = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'humidity': [60, 62, 58, 65, 67],
            'status': ['A', 'B', 'A', 'C', 'B'],
            'region': ['North', 'South', 'North', 'East', 'West'],
            'category': ['X', 'Y', 'X', 'Z', 'Y']
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with mixed numerical and categorical columns
        self.data_mixed = pd.DataFrame({
            'temperature': [20.5, np.nan, 19.8, 22.1, 23.0],
            'humidity': [60, 62, np.nan, 65, 67],
            'status': ['A', 'B', 'A', 'C', 'B'],
            'region': ['North', 'South', 'North', 'East', 'West'],
            'voltage': [220.5, 221.3, np.nan, 219.8, 220.1]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with NaN values in categorical columns
        self.data_with_nan_categorical = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, np.nan, 23.0],
            'humidity': [60, 62, 58, 65, np.nan],
            'status': ['A', 'B', np.nan, 'C', 'B'],
            'region': ['North', 'South', 'North', 'East', 'West']
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with all NaN in a column
        self.data_all_nan = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'humidity': [60, 62, 58, 65, 67],
            'broken_sensor': [np.nan, np.nan, np.nan, np.nan, np.nan]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data for inverse transform testing
        self.data_for_inverse = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8],
            'humidity': [60, 62, 58],
            'status': ['A', 'B', 'A']
        }, index=pd.date_range('2024-01-01', periods=3, freq='1Min'))
        
        # Create transformed data (with NaN values filled)
        self.transformed_data = pd.DataFrame({
            'temperature': [20.5, 21.1, 19.8, 22.1, 23.0],
            'humidity': [60, 62, 63.5, 65, 67],
            'status': ['A', 'B', 'A', 'C', 'B'],
            'region': ['North', 'South', 'North', 'East', 'West']
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with missing columns (for error testing)
        self.data_missing_columns = pd.DataFrame({
            'temperature': [21.0, 22.0],
            'humidity': [61, 63]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1Min'))
        
        # Create data with non-declared categorical features (object dtype in numerical columns)
        self.data_with_object_in_num = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'humidity': [60, 62, 58, 65, 67],
            'sensor_id': ['S1', 'S2', 'S3', 'S4', 'S5']  # This should be detected as categorical
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with only categorical features
        self.data_only_categorical = pd.DataFrame({
            'status': ['A', 'B', 'A', 'C', 'B'],
            'region': ['North', 'South', 'North', 'East', 'West']
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with boolean columns
        self.data_with_boolean = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'is_active': [True, False, True, False, True],
            'status': ['A', 'B', 'A', 'C', 'B']
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))

    def tearDown(self):
        """Clean up after each test."""
        del (self.data_numerical, self.data_with_categorical, 
             self.data_mixed, self.data_with_nan_categorical,
             self.data_all_nan, self.data_for_inverse,
             self.transformed_data, self.data_missing_columns,
             self.data_with_object_in_num, self.data_only_categorical,
             self.data_with_boolean)

    def test_init_defaults(self):
        """Test default initialization."""
        imputer = Imputer()
        self.assertEqual(imputer.strategy, 'mean')
        self.assertEqual(imputer.categorical_features, [])
        self.assertIsNotNone(imputer.numerical_imputer)
        self.assertIsNotNone(imputer.categorical_imputer)

    def test_init_median_strategy(self):
        """Test initialization with median strategy."""
        imputer = Imputer(strategy='median')
        self.assertEqual(imputer.strategy, 'median')
        self.assertEqual(imputer.numerical_imputer.strategy, 'median')
        self.assertEqual(imputer.categorical_imputer.strategy, 'most_frequent')

    def test_init_invalid_strategy(self):
        """Test initialization with invalid strategy raises ValueError."""
        with self.assertRaises(ValueError):
            Imputer(strategy='invalid_strategy')

    def test_init_with_categorical_features(self):
        """Test initialization with categorical features."""
        imputer = Imputer(categorical_features=['status', 'region'])
        self.assertEqual(imputer.categorical_features, ['status', 'region'])

    def test_fit_numerical_data(self):
        """Test fitting with numerical data containing NaN."""
        imputer = Imputer()
        imputer.fit(self.data_numerical)
        
        self.assertEqual(imputer.n_features_in_, 3)
        self.assertEqual(len(imputer.feature_names_in_), 3)
        self.assertEqual(len(imputer.numerical_columns), 3)
        self.assertEqual(len(imputer.categorical_columns), 0)
        self.assertIsNotNone(imputer.numerical_imputer.statistics_)

    def test_fit_with_categorical_features(self):
        """Test fitting with categorical features."""
        imputer = Imputer(categorical_features=['status'])
        imputer.fit(self.data_with_categorical)
        
        self.assertEqual(imputer.n_features_in_, 5)
        self.assertIn('status', imputer.categorical_columns)
        self.assertIn('temperature', imputer.numerical_columns)
        self.assertEqual(len(imputer.numerical_columns), 2)  # temperature and humidity
        self.assertEqual(len(imputer.categorical_columns), 1)  # status

    def test_fit_mixed_data(self):
        """Test fitting with mixed numerical and categorical data."""
        imputer = Imputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_mixed)
        
        self.assertEqual(imputer.n_features_in_, 5)
        self.assertEqual(len(imputer.numerical_columns), 3)  # temperature, humidity, voltage
        self.assertEqual(len(imputer.categorical_columns), 2)  # status, region

    def test_fit_detects_non_declared_categorical(self):
        """Test that non-declared categorical features are detected."""
        imputer = Imputer()
        imputer.fit(self.data_with_object_in_num)
        
        # sensor_id should be detected as non-declared categorical
        self.assertIn('sensor_id', imputer.non_declared_categorical_features)
        # sensor_id should be excluded from numerical columns
        self.assertNotIn('sensor_id', imputer.numerical_columns)
        # sensor_id should be dropped entirely (not in categorical either)
        self.assertEqual(len(imputer.numerical_columns), 2)  # temperature, humidity only

    def test_fit_with_all_nan_column(self):
        """Test fitting with a column that has all NaN values."""
        imputer = Imputer()
        imputer.fit(self.data_all_nan)

        # All-NaN column should be dropped from numerical_columns
        self.assertEqual(imputer.n_features_in_, 3)
        self.assertNotIn('broken_sensor', imputer.numerical_columns)
        self.assertListEqual(imputer.numerical_columns, ['temperature', 'humidity'])

        # Transform should work and return only the valid columns
        result = imputer.transform(self.data_all_nan)
        self.assertEqual(result.shape[1], 2)
        self.assertListEqual(list(result.columns), ['temperature', 'humidity'])
        self.assertFalse(result.isna().any().any())

    def test_fit_empty_categorical(self):
        """Test fitting when categorical columns DataFrame is empty."""
        imputer = Imputer(categorical_features=['nonexistent'])
        imputer.fit(self.data_numerical)
        
        # Should handle empty categorical data gracefully
        self.assertEqual(len(imputer.categorical_columns), 0)
        self.assertIsNotNone(imputer.categorical_imputer)

    def test_transform_with_numerical_data(self):
        """Test transformation with numerical data."""
        imputer = Imputer()
        imputer.fit(self.data_numerical)
        result = imputer.transform(self.data_numerical)
        
        # Should return DataFrame with same shape and columns
        self.assertEqual(result.shape, self.data_numerical.shape)
        self.assertListEqual(list(result.columns), self.data_numerical.columns.tolist())
        # No NaN values should remain
        self.assertFalse(result.isna().any().any())
        # Index should be preserved
        self.assertTrue(result.index.equals(self.data_numerical.index))

    def test_transform_with_categorical_data(self):
        """Test transformation with categorical data."""
        imputer = Imputer(categorical_features=['status'])
        imputer.fit(self.data_with_categorical)
        result = imputer.transform(self.data_with_categorical)
        
        # Should return DataFrame without non-declared categorical features
        self.assertEqual(result.shape[1], 3)  # temperature, humidity, status
        # Categorical values should be imputed (most frequent)
        self.assertFalse(result['status'].isna().any())
        # Index should be preserved
        self.assertTrue(result.index.equals(self.data_with_categorical.index))

    def test_transform_mixed_data(self):
        """Test transformation with mixed data."""
        imputer = Imputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_mixed)
        result = imputer.transform(self.data_mixed)
        
        # Should handle mixed data correctly
        self.assertEqual(result.shape, self.data_mixed.shape)
        # No NaN values should remain
        self.assertFalse(result.isna().any().any())

    def test_transform_unfitted_raises(self):
        """Test that transform on unfitted model raises error."""
        imputer = Imputer()
        with self.assertRaises(NotFittedError):
            imputer.transform(self.data_numerical)

    def test_transform_with_nan_categorical(self):
        """Test transformation with NaN values in categorical columns."""
        imputer = Imputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_with_nan_categorical)
        result = imputer.transform(self.data_with_nan_categorical)
        
        # Categorical NaN values should be imputed
        self.assertFalse(result[['status', 'region']].isna().any().any())

    def test_inverse_transform(self):
        """Test inverse transform method."""
        imputer = Imputer(categorical_features=['status'])
        imputer.fit(self.data_for_inverse)
        transformed = imputer.transform(self.data_for_inverse)
        inverse = imputer.inverse_transform(transformed)
        
        # Should return DataFrame with same shape
        self.assertEqual(inverse.shape, self.data_for_inverse.shape)
        # Index should be restored to original
        self.assertTrue(inverse.index.equals(self.data_for_inverse.index))
        # Column names should match original
        self.assertListEqual(list(inverse.columns), self.data_for_inverse.columns.tolist())

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method."""
        imputer = Imputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_mixed)
        
        feature_names = imputer.get_feature_names_out()
        
        # Should return numerical columns first, then categorical
        expected_order = ['temperature', 'humidity', 'voltage', 'status', 'region']
        self.assertListEqual(feature_names, expected_order)

    def test_transform_with_missing_columns(self):
        """Test that transform raises error for missing columns."""
        imputer = Imputer()
        imputer.fit(self.data_numerical)
        
        with self.assertRaises(ValueError):
            imputer.transform(self.data_missing_columns)

    def test_transform_with_boolean_columns(self):
        """Test transformation with boolean columns."""
        imputer = Imputer(categorical_features=['status'])
        imputer.fit(self.data_with_boolean)
        result = imputer.transform(self.data_with_boolean)
        
        # Boolean column should be handled as numerical
        self.assertFalse(result.isna().any().any())

    def test_transform_preserves_index_type(self):
        """Test that index type is preserved after transformation."""
        imputer = Imputer()
        imputer.fit(self.data_numerical)
        result = imputer.transform(self.data_numerical)
        
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_transform_with_median_strategy(self):
        """Test transformation with median strategy."""
        imputer = Imputer(strategy='median')
        imputer.fit(self.data_numerical)
        result = imputer.transform(self.data_numerical)
        
        # Should handle NaN values correctly
        self.assertFalse(result.isna().any().any())
        # Should return DataFrame with same shape
        self.assertEqual(result.shape, self.data_numerical.shape)

    def test_transform_with_empty_categorical_data(self):
        """Test transformation when no categorical columns match."""
        imputer = Imputer(categorical_features=['nonexistent'])
        imputer.fit(self.data_numerical)
        result = imputer.transform(self.data_numerical)
        
        # Should handle empty categorical data gracefully
        self.assertEqual(result.shape, self.data_numerical.shape)

    def test_transform_with_only_categorical(self):
        """Test transformation with only categorical features."""
        imputer = Imputer(categorical_features=['status', 'region'])
        imputer.fit(self.data_only_categorical)
        result = imputer.transform(self.data_only_categorical)

        # Should handle categorical-only data
        self.assertEqual(result.shape, self.data_only_categorical.shape)
        self.assertFalse(result.isna().any().any())

if __name__ == '__main__':
    unittest.main()