import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from energy_fault_detector.data_preprocessing.categorical_encoder import CategoricalEncoder

class TestCategoricalEncoder:
    @pytest.fixture
    def sample_data(self):
        """Create a sample DataFrame with mixed categorical and numerical features."""
        return pd.DataFrame({
            'Category1': ['A', 'B', 'A', 'C', 'B'],
            'Category2': ['X', 'Y', 'Z', 'X', 'Y'],
            'Numerical1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Numerical2': [10, 20, 30, 40, 50],
        })

    @pytest.fixture
    def encoder_with_config(self):
        """Create an encoder with declared categorical features."""
        return CategoricalEncoder(categorical_features=['Category1', 'Category2'])

    @pytest.fixture
    def empty_encoder(self):
        """Create an encoder without declared categorical features."""
        return CategoricalEncoder(categorical_features=[])

    def test_initialization(self, encoder_with_config, empty_encoder):
        """Test the initialization of CategoricalEncoder."""
        assert encoder_with_config.categorical_features == ['Category1', 'Category2']
        assert empty_encoder.categorical_features == []
        assert isinstance(encoder_with_config.one_hot_encoder, type(MagicMock().__class__))

    def test_fit_with_valid_data(self, encoder_with_config, sample_data):
        """Test that fitting works correctly on valid data."""
        encoder_with_config.fit(sample_data)
        
        # Check fitted attributes
        assert encoder_with_config.feature_names_in_ == sample_data.columns.tolist()
        assert encoder_with_config.categorical_columns == ['Category1', 'Category2']
        assert encoder_with_config.numerical_columns == ['Numerical1', 'Numerical2']
        assert encoder_with_config.n_features_in_ == 4
        assert len(encoder_with_config.categorical_columns) == 2
        assert len(encoder_with_config.numerical_columns) == 2

    def test_fit_with_non_declared_categorical_in_numerical_columns(self, encoder_with_config):
        """Test handling of non-declared categorical features in numerical columns."""
        data_with_mixed = pd.DataFrame({
            'Category1': ['A', 'B', 'A'],
            'Numerical': ['1', '2', '3'],  # String instead of numeric
            'RealNumerical': [1.0, 2.0, 3.0]
        })
        
        encoder_with_config.categorical_features = ['Category1']
        with patch('energy_fault_detector.data_preprocessing.categorical_encoder.logger') as mock_logger:
            encoder_with_config.fit(data_with_mixed)
            
            # Verify non-declared categorical features were identified and dropped
            assert 'Numerical' in encoder_with_config.non_declared_categorical_features
            assert 'Numerical' not in encoder_with_config.numerical_columns
            assert 'Numerical' not in encoder_with_config.categorical_columns
            mock_logger.info.assert_called_once()

    def test_fit_with_no_categorical_features(self, empty_encoder, sample_data):
        """Test fit behavior when no categorical features are declared."""
        empty_encoder.fit(sample_data)
        
        assert empty_encoder.categorical_columns == []
        assert empty_encoder.numerical_columns == sample_data.columns.tolist()
        assert empty_encoder.n_features_in_ == 4

    def test_transform_with_valid_data(self, encoder_with_config, sample_data):
        """Test transformation with valid data."""
        encoder_with_config.fit(sample_data)
        transformed = encoder_with_config.transform(sample_data)
        
        # Verify output shape and column names
        expected_columns = [
            'Numerical1', 'Numerical2', 
            'Category1_A', 'Category1_B', 'Category1_C', 
            'Category2_X', 'Category2_Y', 'Category2_Z'
        ]
        assert list(transformed.columns) == expected_columns
        assert transformed.shape == (5, len(expected_columns))
        
        # Check values are binary for one-hot columns
        onehot_columns = [col for col in transformed.columns if col.startswith('Category')]
        assert all(transformed[col].isin([0, 1]).all() for col in onehot_columns)

    def test_transform_missing_features_raises_key_error(self, encoder_with_config, sample_data):
        """Test that transform raises KeyError when input is missing required features."""
        encoder_with_config.fit(sample_data)
        
        incomplete_data = sample_data.drop(columns=['Category1'])
        with pytest.raises(KeyError, match="Category1"):
            encoder_with_config.transform(incomplete_data)

    def test_transform_with_no_categorical_features(self, empty_encoder, sample_data):
        """Test transform behavior when no categorical features are declared."""
        empty_encoder.fit(sample_data)
        transformed = empty_encoder.transform(sample_data)
        
        # Should return only numerical data without transformation
        pd.testing.assert_frame_equal(transformed, sample_data)

    def test_inverse_transform_roundtrip(self, encoder_with_config, sample_data):
        """Test inverse_transform correctly reverses the transformation."""
        encoder_with_config.fit(sample_data)
        transformed = encoder_with_config.transform(sample_data)
        reconstructed = encoder_with_config.inverse_transform(transformed)
        
        # Sort columns before comparison to handle potential column order differences
        pd.testing.assert_frame_equal(
            sample_data.sort_index(axis=1), 
            reconstructed.sort_index(axis=1)
        )

    def test_inverse_transform_with_missing_categorical_columns(self, encoder_with_config, sample_data):
        """Test inverse_transform handles missing categorical columns gracefully."""
        encoder_with_config.fit(sample_data)
        transformed = encoder_with_config.transform(sample_data)
        
        # Create a DataFrame missing some one-hot encoded columns
        incomplete_transformed = transformed.drop(columns=['Category1_C', 'Category2_Z'])
        reconstructed = encoder_with_config.inverse_transform(incomplete_transformed)
        
        # Original data should still be reconstructed correctly
        pd.testing.assert_frame_equal(
            sample_data.sort_index(axis=1), 
            reconstructed.sort_index(axis=1)
        )

    def test_inverse_transform_with_no_categorical_features(self, empty_encoder, sample_data):
        """Test inverse_transform behavior when no categorical features are declared."""
        empty_encoder.fit(sample_data)
        transformed = empty_encoder.transform(sample_data)
        reconstructed = empty_encoder.inverse_transform(transformed)
        
        # Should return only numerical data without transformation
        pd.testing.assert_frame_equal(reconstructed, sample_data)

    def test_get_feature_names_out(self, encoder_with_config, sample_data):
        """Test that get_feature_names_out returns correct feature names."""
        encoder_with_config.fit(sample_data)
        feature_names = encoder_with_config.get_feature_names_out()
        
        expected_names = [
            'Numerical1', 'Numerical2',
            'Category1_A', 'Category1_B', 'Category1_C',
            'Category2_X', 'Category2_Y', 'Category2_Z'
        ]
        assert feature_names == expected_names

    def test_get_feature_names_out_with_empty_categorical_features(self, empty_encoder, sample_data):
        """Test get_feature_names_out when no categorical features are declared."""
        empty_encoder.fit(sample_data)
        feature_names = empty_encoder.get_feature_names_out()
        
        assert feature_names == ['Category1', 'Category2', 'Numerical1', 'Numerical2']

    def test_unfitted_transform_raises_error(self, sample_data):
        """Test that transform raises error when called before fitting."""
        encoder = CategoricalEncoder()
        with pytest.raises(NotImplementedError):
            encoder.transform(sample_data)

    def test_unfitted_inverse_transform_raises_error(self, sample_data):
        """Test that inverse_transform raises error when called before fitting."""
        encoder = CategoricalEncoder()
        with pytest.raises(NotImplementedError):
            encoder.inverse_transform(sample_data)

    def test_unfitted_get_feature_names_out_raises_error(self, sample_data):
        """Test that get_feature_names_out raises error when called before fitting."""
        encoder = CategoricalEncoder()
        with pytest.raises(NotImplementedError):
            encoder.get_feature_names_out()

    def test_handle_empty_categorical_data(self):
        """Test behavior with empty categorical dataframes."""
        data = pd.DataFrame({
            'Numerical1': [1.0, 2.0, 3.0],
            'Numerical2': [10, 20, 30]
        })
        encoder = CategoricalEncoder(categorical_features=[])
        
        encoder.fit(data)
        transformed = encoder.transform(data)
        
        pd.testing.assert_frame_equal(transformed, data)

    def test_handle_unknown_categories_in_transform(self, encoder_with_config):
        """Test how transform handles categories not seen during fit."""
        train_data = pd.DataFrame({
            'Category1': ['A', 'B', 'A'],
            'Category2': ['X', 'Y', 'Z'],
            'Numerical1': [1.0, 2.0, 3.0]
        })
        test_data = pd.DataFrame({
            'Category1': ['A', 'C', 'D'],  # 'D' is unknown category
            'Category2': ['X', 'Y', 'Z'],
            'Numerical1': [4.0, 5.0, 6.0]
        })
        
        encoder_with_config.fit(train_data)
        
        # Note: With handle_unknown='ignore', unknown categories are encoded as all zeros
        transformed = encoder_with_config.transform(test_data)
        
        # Verify unknown categories are encoded as zeros
        assert transformed.loc[2, 'Category1_D'] == 0  # Unknown category encoded as 0

    def test_pandas_dtype_preservation(self, encoder_with_config):
        """Test that numerical dtypes are preserved in transformed data."""
        data = pd.DataFrame({
            'Category1': ['A', 'B', 'C'],
            'Numerical1': [1, 2, 3],
            'Numerical2': [1.1, 2.2, 3.3]
        })
        
        encoder_with_config.fit(data)
        transformed = encoder_with_config.transform(data)
        
        # Check dtypes for numerical columns in transformed data
        assert transformed['Numerical1'].dtype == np.float64
        assert transformed['Numerical2'].dtype == np.float64

    def test_index_preservation_in_transform(self, encoder_with_config):
        """Test that index is preserved during transform."""
        data = pd.DataFrame({
            'Category1': ['A', 'B', 'C'],
            'Numerical1': [1.0, 2.0, 3.0]
        }, index=['row1', 'row2', 'row3'])
        
        encoder_with_config.fit(data)
        transformed = encoder_with_config.transform(data)
        
        assert list(transformed.index) == ['row1', 'row2', 'row3']

    def test_index_preservation_in_inverse_transform(self, encoder_with_config):
        """Test that index is preserved during inverse_transform."""
        data = pd.DataFrame({
            'Category1': ['A', 'B', 'C'],
            'Numerical1': [1.0, 2.0, 3.0]
        }, index=['row1', 'row2', 'row3'])
        
        encoder_with_config.fit(data)
        transformed = encoder_with_config.transform(data)
        reconstructed = encoder_with_config.inverse_transform(transformed)
        
        assert list(reconstructed.index) == ['row1', 'row2', 'row3']

    def test_repeated_fit_and_transform(self, encoder_with_config, sample_data):
        """Test that repeated fitting and transforming works correctly."""
        encoder_with_config.fit(sample_data)
        transformed1 = encoder_with_config.transform(sample_data)
        
        # Fit again with different subset
        subset_data = sample_data.iloc[:3]
        encoder_with_config.fit(subset_data)
        transformed2 = encoder_with_config.transform(subset_data)
        
        # Check that results differ due to different fit
        assert not transformed1.equals(transformed2)
        assert transformed2.shape == (3, transformed1.shape[1])

    def test_get_feature_names_out_with_custom_input_features(self, encoder_with_config, sample_data):
        """Test that input_features argument is ignored in get_feature_names_out."""
        encoder_with_config.fit(sample_data)
        
        # Provide arbitrary input_features - should be ignored
        custom_features = ['custom1', 'custom2']
        feature_names = encoder_with_config.get_feature_names_out(custom_features)
        
        # Result should not be affected by input_features
        assert 'Category1' not in feature_names
        assert feature_names == encoder_with_config.get_feature_names_out()