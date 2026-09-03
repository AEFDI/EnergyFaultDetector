import unittest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from energy_fault_detector.data_preprocessing.categorical_encoder import CategoricalEncoder


class TestCategoricalEncoder(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        # Create sample data with numerical, categorical, and mixed features
        self.data_clean = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'humidity': [60, 62, 58, 65, 67],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'region': ['North', 'South', 'North', 'East', 'West']
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))

        self.data_clean_encoded_features = ['temperature', 'humidity', 'category_A', 'category_B', 'category_C', 'region_North', 'region_South', 'region_East', 'region_West']
        
        # Create data with non-declared categorical features in numerical columns
        self.data_with_non_declared = self.data_clean.copy()
        self.data_with_non_declared['status'] = ['high', 'medium', 'low', 'high', 'medium']
        
        # Create data with only numerical columns
        self.data_numerical_only = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'humidity': [60, 62, 58, 65, 67]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        self.data_numerical_only_encoded_features = ['temperature', 'humidity']
        
        # Create data with missing values in categorical columns
        self.data_with_nan = self.data_clean.copy()
        self.data_with_nan.iloc[2, 2] = np.nan
        self.data_with_nan.iloc[3, 3] = np.nan
        
        # Create data with duplicate rows for testing deduplication
        self.data_with_duplicates = pd.DataFrame({
            'temperature': [20.5, 20.5, 21.3, 22.1],
            'category': ['A', 'A', 'B', 'C']
        }, index=pd.date_range('2024-01-01', periods=4, freq='1Min'))
        
        # Create data for testing inverse transform
        self.data_for_inverse = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8],
            'category': ['A', 'B', 'A'],
            'region': ['North', 'South', 'North']
        }, index=pd.date_range('2024-01-01', periods=3, freq='1Min'))

    def tearDown(self):
        """Clean up after each test."""
        del (self.data_clean, self.data_with_non_declared, self.data_numerical_only,
             self.data_with_nan, self.data_with_duplicates, self.data_for_inverse)

    def test_init_defaults(self):
        """Test default initialization."""
        encoder = CategoricalEncoder()
        self.assertEqual(encoder.categorical_features, [])
        self.assertIsNone(encoder.categorical_columns)
        self.assertIsNone(encoder.numerical_columns)

    def test_init_with_categorical_features(self):
        """Test initialization with categorical features."""
        encoder = CategoricalEncoder(categorical_features=['category', 'region'])
        self.assertListEqual(encoder.categorical_features, ['category', 'region'])

    def test_fit_with_categorical_features(self):
        """Test fitting with categorical features."""
        encoder = CategoricalEncoder(categorical_features=['category', 'region'])
        encoder.fit(self.data_clean)
        
        self.assertEqual(encoder.n_features_in_, 4)
        self.assertListEqual(encoder.feature_names_in_ or [], ['temperature', 'humidity', 'category', 'region'])
        self.assertListEqual(encoder.numerical_columns or [], ['temperature', 'humidity'])
        self.assertListEqual(encoder.categorical_columns or [], ['category', 'region'])

    def test_fit_without_declaring_categorical_features(self):
        """Test fitting without specifying categorical features."""
        encoder = CategoricalEncoder()
        encoder.fit(self.data_clean)
        
        # Should ignore categorical columns
        self.assertEqual(encoder.n_features_in_, 4)
        self.assertListEqual(encoder.numerical_columns or [], ['temperature', 'humidity'])
        self.assertListEqual(encoder.categorical_columns or [], [])

    def test_fit_with_non_declared_categorical_features(self):
        """Test detection of non-declared categorical features in numerical columns."""
        encoder = CategoricalEncoder(categorical_features=['category', 'region'])
        encoder.fit(self.data_with_non_declared)
        
        # Should detect 'status' as non-declared categorical
        self.assertIn('status', encoder.non_declared_categorical_features or [])
        self.assertNotIn('status', encoder.numerical_columns or [])
        self.assertNotIn('status', encoder.categorical_columns or [])

    def test_fit_with_only_numerical_data(self):
        """Test fitting on numerical-only data."""
        encoder = CategoricalEncoder()
        encoder.fit(self.data_numerical_only)
        
        self.assertEqual(encoder.n_features_in_, 2)
        self.assertListEqual(encoder.numerical_columns or [], ['temperature', 'humidity'])
        self.assertEqual(len(encoder.categorical_columns or []), 0)

    def test_transform_with_categorical_features(self):
        """Test transformation with categorical features."""
        encoder = CategoricalEncoder(categorical_features=['category', 'region'])
        encoder.fit(self.data_clean)
        result = encoder.transform(self.data_clean)
        
        # Should have transformed columns: temperature, humidity, category_A, category_B, category_C, region_*
        for col in self.data_clean_encoded_features:
            self.assertIn(col, result.columns)
        # Should maintain same index
        self.assertTrue(result.index.equals(self.data_clean.index))
        # Should not contain NaN
        self.assertFalse(result.isna().any().any())

    def test_transform_with_only_numerical_data(self):
        """Test transformation when no categorical features."""
        encoder = CategoricalEncoder()
        encoder.fit(self.data_numerical_only)
        result = encoder.transform(self.data_numerical_only)
        
        # Should return same columns and shape
        self.assertEqual(result.shape, self.data_numerical_only.shape)
        self.assertListEqual(list(result.columns), self.data_numerical_only_encoded_features)

    def test_transform_missing_columns_raises(self):
        """Test that missing columns raise KeyError."""
        encoder = CategoricalEncoder(categorical_features=['category'])
        encoder.fit(self.data_clean)
        
        # Create data missing one feature
        partial_data = self.data_clean.drop(columns=['temperature'])
        
        with self.assertRaises(KeyError):
            encoder.transform(partial_data)

    def test_transform_invalid_type_raises(self):
        """Test that non-DataFrame input raises TypeError."""
        encoder = CategoricalEncoder()
        encoder.fit(self.data_clean)
        
        with self.assertRaises(TypeError):
            encoder.transform([1, 2, 3])

    def test_transform_unfitted_raises(self):
        """Test that transform on unfitted model raises error."""
        encoder = CategoricalEncoder()
        
        with self.assertRaises(NotFittedError):
            encoder.transform(self.data_clean)

    def test_inverse_transform(self):
        """Test inverse transform returns original categorical values."""
        encoder = CategoricalEncoder(categorical_features=['category', 'region'])
        encoder.fit(self.data_for_inverse)
        
        transformed = encoder.transform(self.data_for_inverse)
        inverse = encoder.inverse_transform(transformed)
        
        # Should have same columns and shape as original
        self.assertListEqual(list(inverse.columns), list(self.data_for_inverse.columns))
        self.assertEqual(inverse.shape, self.data_for_inverse.shape)
        # Should match original data (except for any NaN handling)
        pd.testing.assert_frame_equal(inverse, self.data_for_inverse)

    def test_get_feature_names_out(self):
        """Test feature names out method."""
        encoder = CategoricalEncoder(categorical_features=['category', 'region'])
        encoder.fit(self.data_clean)
        
        feature_names = encoder.get_feature_names_out()
        
        # Should include original numerical features
        self.assertIn('temperature', feature_names)
        self.assertIn('humidity', feature_names)
        # Should include encoded categorical features (exact names depend on OneHotEncoder output)
        categorical_feature_names = [name for name in feature_names if 'category' in name or 'region' in name]
        self.assertGreater(len(categorical_feature_names), 0)
        self.assertEqual(len(feature_names), encoder.n_features_in_ + len(categorical_feature_names) - 2)

    def test_empty_dataframe_handling(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=self.data_clean.columns, 
                                index=pd.DatetimeIndex([]))
        
        encoder = CategoricalEncoder(categorical_features=['category', 'region'])
        encoder.fit(self.data_clean)
        
        result = encoder.transform(empty_df)
        
        self.assertEqual(result.shape[0], 0)

    def test_transform_preserves_index_type(self):
        """Test that index type is preserved after transformation."""
        encoder = CategoricalEncoder(categorical_features=['category'])
        encoder.fit(self.data_clean)
        result = encoder.transform(self.data_clean)
        
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_transform_with_nan_in_categorical(self):
        """Test transformation with NaN values in categorical columns."""
        encoder = CategoricalEncoder(categorical_features=['category', 'region'])
        encoder.fit(self.data_with_nan)
        
        # Should ignore NaN values in categorical columns during transformation
        result = encoder.transform(self.data_with_nan)
        self.assertFalse(result.isna().any().any())

    def test_transform_with_new_categories(self):
        """Test transformation with unseen category values — should not crash, encoded as all-zeros."""
        encoder = CategoricalEncoder(categorical_features=['category'])
        encoder.fit(self.data_clean)

        # Create data with unseen category value
        new_data = pd.DataFrame({
            'temperature': [25.0],
            'humidity': [70],
            'category': ['D'],
            'region': ['North']
        }, index=pd.date_range('2024-01-01', periods=1, freq='1Min'))

        # With handle_unknown='ignore', unseen categories are encoded as all-zeros (no error)
        transformed = encoder.transform(new_data)
        # No new column for 'D' — only fitted categories A, B, C have columns
        category_cols = [c for c in transformed.columns if c.startswith('category_')]
        self.assertEqual(sorted(category_cols), ['category_A', 'category_B', 'category_C'])
        self.assertTrue((transformed[category_cols] == 0).all().all(),
                        "Unseen category should be encoded as all-zeros.")

if __name__ == '__main__':
    unittest.main()