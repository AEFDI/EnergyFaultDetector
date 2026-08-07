import unittest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from energy_fault_detector.data_preprocessing.scaler import Scaler


class TestScaler(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        # Create sample numerical data
        self.data_numerical = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'humidity': [60, 62, 58, 65, 67],
            'pressure': [1013.2, 1014.5, 1012.8, 1015.0, 1013.7]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with categorical features (already encoded for this test)
        self.data_with_categorical = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'humidity': [60, 62, 58, 65, 67],
            'category_A': [1, 0, 1, 0, 0],
            'category_B': [0, 1, 0, 0, 1],
            'category_C': [0, 0, 0, 1, 0],
            'region_East': [0, 0, 1, 0, 0],
            'region_North': [1, 0, 1, 0, 0],
            'region_South': [0, 1, 0, 0, 0],
            'region_West': [0, 0, 0, 1, 1]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with negative values
        self.data_with_negatives = pd.DataFrame({
            'temperature': [-5.2, -3.1, 0.0, 2.5, 4.8],
            'humidity': [-10, -5, 0, 15, 20]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with outliers
        self.data_with_outliers = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 150.0],
            'humidity': [60, 62, 58, 65, 500]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with missing values
        self.data_with_nan = pd.DataFrame({
            'temperature': [20.5, np.nan, 19.8, 22.1, 23.0],
            'humidity': [60, 62, np.nan, 65, 67]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data for inverse transform testing
        self.data_for_inverse = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8],
            'humidity': [60, 62, 58]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1Min'))
        
        # Create transformed data for testing
        self.transformed_data = pd.DataFrame({
            'temperature': [0.612, 0.918, -0.408, 1.428, 1.734],
            'humidity': [-0.408, -0.102, -0.714, 0.816, 1.122]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        # Create data with different columns than during fit
        self.data_with_different_columns = pd.DataFrame({
            'temperature': [21.0, 22.0],
            'humidity': [61, 63]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1Min'))

    def tearDown(self):
        """Clean up after each test."""
        del (self.data_numerical, self.data_with_categorical, 
             self.data_with_negatives, self.data_with_outliers, 
             self.data_with_nan, self.data_for_inverse, 
             self.transformed_data, self.data_with_different_columns)

    def test_init_defaults(self):
        """Test default initialization."""
        scaler = Scaler()
        self.assertEqual(scaler.scaler_type, 'standard')
        self.assertTrue(scaler.scale_categorical_features)
        self.assertEqual(scaler.categorical_features, [])
        self.assertIsNotNone(scaler.scaler)

    def test_init_invalid_scaler_type(self):
        """Test initialization with invalid scaler type raises ValueError."""
        with self.assertRaises(ValueError):
            Scaler(scaler_type='invalid_type')

    def test_init_with_minmax_scaler(self):
        """Test initialization with MinMax scaler."""
        scaler = Scaler(scaler_type='minmax')
        self.assertEqual(scaler.scaler_type, 'minmax')
        self.assertIsNotNone(scaler.scaler)
        self.assertTrue(hasattr(scaler.scaler, 'feature_range'))

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        scaler = Scaler(scaler_type='minmax', feature_range=(0, 1))
        self.assertEqual(scaler.scaler.feature_range, (0, 1))

    def test_init_with_categorical_features(self):
        """Test initialization with categorical features."""
        scaler = Scaler(scale_categorical_features=False, categorical_features=['category_A', 'category_B'])
        self.assertEqual(scaler.categorical_features, ['category_A', 'category_B'])
        self.assertFalse(scaler.scale_categorical_features)

    def test_fit_standard_scaler(self):
        """Test fitting with standard scaler."""
        scaler = Scaler()
        scaler.fit(self.data_numerical)
        
        self.assertEqual(scaler.n_features_in_, 3)
        self.assertListEqual(scaler.feature_names_in_, ['temperature', 'humidity', 'pressure'])
        self.assertIsNotNone(scaler.scaler.mean_)
        self.assertIsNotNone(scaler.scaler.var_)

    def test_fit_minmax_scaler(self):
        """Test fitting with minmax scaler."""
        scaler = Scaler(scaler_type='minmax')
        scaler.fit(self.data_numerical)
        
        self.assertEqual(scaler.n_features_in_, 3)
        self.assertIsNotNone(scaler.scaler.data_min_)
        self.assertIsNotNone(scaler.scaler.data_max_)

    def test_fit_with_categorical_features_scaled(self):
        """Test fitting with categorical features and scale_categorical_features=True."""
        scaler = Scaler(scale_categorical_features=True)
        scaler.fit(self.data_with_categorical)
        
        self.assertEqual(scaler.n_features_in_, 9)
        self.assertListEqual(scaler.feature_names_out_, self.data_with_categorical.columns.tolist())

    def test_fit_with_categorical_features_not_scaled(self):
        """Test fitting with categorical features and scale_categorical_features=False."""
        scaler = Scaler(scale_categorical_features=False, categorical_features=['category', 'region'])
        scaler.fit(self.data_with_categorical)
        
        # Should only fit on non-categorical features
        self.assertEqual(scaler.n_features_in_, 9)
        # Check that scaling was applied only to numerical features
        numerical_cols = ['temperature', 'humidity']
        # Verify the scaler was fitted on the numerical columns only
        subset_to_fit = self.data_with_categorical[numerical_cols]
        self.assertEqual(scaler.scaler.mean_.shape[0], len(numerical_cols))

    def test_transform_standard_scaler(self):
        """Test transformation with standard scaler."""
        scaler = Scaler()
        scaler.fit(self.data_numerical)
        result = scaler.transform(self.data_numerical)
        
        # Should return DataFrame with same shape and columns
        self.assertEqual(result.shape, self.data_numerical.shape)
        self.assertListEqual(list(result.columns), self.data_numerical.columns.tolist())
        
        # For standard scaler, the transformed data should have mean ~0 and std ~1
        np.testing.assert_array_almost_equal(result.mean(), [0.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(result.std(ddof=0), [1.0, 1.0, 1.0], decimal=5)

    def test_transform_minmax_scaler(self):
        """Test transformation with minmax scaler."""
        scaler = Scaler(scaler_type='minmax')
        scaler.fit(self.data_numerical)
        result = scaler.transform(self.data_numerical)
        
        # For minmax scaler, values should be scaled to [0,1]
        self.assertGreaterEqual(result.min().min(), 0.0)
        self.assertLessEqual(result.max().max(), 1.0)
        
        # Should preserve index
        self.assertTrue(result.index.equals(self.data_numerical.index))

    def test_transform_with_categorical_features_scaled(self):
        """Test transformation with categorical features and scale_categorical_features=True."""
        scaler = Scaler(scale_categorical_features=True)
        scaler.fit(self.data_with_categorical)
        result = scaler.transform(self.data_with_categorical)
        
        # Should have same shape and columns
        self.assertEqual(result.shape, self.data_with_categorical.shape)
        self.assertListEqual(list(result.columns), self.data_with_categorical.columns.tolist())

    def test_transform_with_categorical_features_not_scaled(self):
        """Test transformation with categorical features and scale_categorical_features=False."""
        scaler = Scaler(scale_categorical_features=False, categorical_features=['category_A', 'category_B'])
        scaler.fit(self.data_with_categorical)
        result = scaler.transform(self.data_with_categorical)
        
        # Only numerical columns should be scaled
        numerical_cols = ['temperature', 'humidity']
        self.assertFalse(result[numerical_cols].equals(self.data_with_categorical[numerical_cols]))
        # Categorical columns should remain unchanged
        pd.testing.assert_frame_equal(result[['category_A', 'category_B']], self.data_with_categorical[['category_A', 'category_B']].astype(float))  # Ensure numeric types for comparison

    def test_transform_unfitted_raises(self):
        """Test that transform on unfitted model raises error."""
        scaler = Scaler()
        with self.assertRaises(NotFittedError):
            scaler.transform(self.data_numerical)

    def test_inverse_transform_standard_scaler(self):
        """Test inverse transform with standard scaler."""
        scaler = Scaler()
        scaler.fit(self.data_for_inverse)
        transformed = scaler.transform(self.data_for_inverse)
        inverse = scaler.inverse_transform(transformed)
        
        # Should return to original values
        pd.testing.assert_frame_equal(inverse, self.data_for_inverse.apply(lambda x: x.astype(float)))

    def test_inverse_transform_minmax_scaler(self):
        """Test inverse transform with minmax scaler."""
        scaler = Scaler(scaler_type='minmax')
        scaler.fit(self.data_for_inverse)
        transformed = scaler.transform(self.data_for_inverse)
        inverse = scaler.inverse_transform(transformed)
        
        # Should return to original values
        pd.testing.assert_frame_equal(inverse, self.data_for_inverse.apply(lambda x: x.astype(float)))  # Ensure numeric types for comparison

    def test_inverse_transform_with_categorical_features_not_scaled(self):
        """Test inverse transform with categorical features and scale_categorical_features=False."""
        scaler = Scaler(scale_categorical_features=False, categorical_features=['category_A'])
        scaler.fit(self.data_with_categorical)
        transformed = scaler.transform(self.data_with_categorical)
        inverse = scaler.inverse_transform(transformed)
        
        # Categorical columns should be unchanged
        pd.testing.assert_frame_equal(inverse[['category_A']], self.data_with_categorical[['category_A']].astype(float))
        # Numerical columns should return to original values
        pd.testing.assert_frame_equal(inverse[['temperature', 'humidity']], self.data_with_categorical[['temperature', 'humidity']].astype(float))

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method."""
        scaler = Scaler()
        scaler.fit(self.data_numerical)
        
        feature_names = scaler.get_feature_names_out()
        
        self.assertListEqual(feature_names, self.data_numerical.columns.tolist())

    def test_transform_preserves_index_type(self):
        """Test that index type is preserved after transformation."""
        scaler = Scaler()
        scaler.fit(self.data_numerical)
        result = scaler.transform(self.data_numerical)
        
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_transform_with_different_columns(self):
        """Test that transform works with different number of columns than fitted."""
        scaler = Scaler()
        scaler.fit(self.data_numerical)
        
        # This should raise an error as per sklearn behavior
        with self.assertRaises(ValueError):
            scaler.transform(self.data_with_different_columns)

    def test_transform_with_negative_values(self):
        """Test transformation with negative values."""
        scaler = Scaler()
        scaler.fit(self.data_with_negatives)
        result = scaler.transform(self.data_with_negatives)
        
        # Should handle negative values correctly
        self.assertFalse(result.isna().any().any())

    def test_transform_with_outliers(self):
        """Test transformation with outliers."""
        scaler = Scaler()
        scaler.fit(self.data_with_outliers)
        result = scaler.transform(self.data_with_outliers)
        
        # Should handle outliers, though they may affect the distribution
        self.assertFalse(result.isna().any().any())
        # Standard scaled values should be reasonable (not NaN or infinite)
        self.assertTrue(np.all(np.isfinite(result.values)))

    def test_transform_with_nan_values(self):
        """Test transformation with NaN values."""
        scaler = Scaler()
        scaler.fit(self.data_with_nan)
        
        result = scaler.transform(self.data_with_nan)
        # Should handle NaN values, resulting in NaN in the same positions
        self.assertTrue(result.isna().any().any())

    def test_transform_with_all_zeros(self):
        """Test transformation with all zero variance columns."""
        data_all_zeros = pd.DataFrame({
            'temperature': [0.0, 0.0, 0.0, 0.0, 0.0],
            'humidity': [60, 62, 58, 65, 67]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        scaler = Scaler()
        scaler.fit(data_all_zeros)
        result = scaler.transform(data_all_zeros)
        
        # Zero variance column should be scaled to zeros
        self.assertTrue((result['temperature'] == 0.0).all())
        # Should not raise any errors

    def test_transform_with_constant_column(self):
        """Test transformation with constant value columns."""
        data_constant = pd.DataFrame({
            'temperature': [5.0, 5.0, 5.0, 5.0, 5.0],
            'humidity': [60, 62, 58, 65, 67]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        scaler = Scaler()
        scaler.fit(data_constant)
        result = scaler.transform(data_constant)
        
        # Constant column should have zero variance after scaling
        self.assertTrue((result['temperature'] == 0.0).all())
        # Should not raise any errors

    def test_transform_with_mixed_types(self):
        """Test transformation with mixed data types after preprocessing."""
        # Simulate data after categorical encoding
        data_mixed = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'category_A': [1, 0, 1, 0, 0],
            'category_B': [0, 1, 0, 0, 1],
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))
        
        scaler = Scaler()
        scaler.fit(data_mixed)
        result = scaler.transform(data_mixed)
        
        # Should scale all columns when scale_categorical_features=True
        self.assertEqual(result.shape, data_mixed.shape)
        self.assertFalse(result.isna().any().any())

if __name__ == '__main__':
    unittest.main()