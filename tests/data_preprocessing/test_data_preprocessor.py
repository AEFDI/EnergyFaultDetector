import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from sklearn.utils.validation import check_is_fitted, NotFittedError

from energy_fault_detector.config.config import Config
from energy_fault_detector.data_preprocessing.data_preprocessor import DataPreprocessor
from energy_fault_detector.fault_detector import FaultDetector


class TestDataPreprocessorPipeline(TestCase):
    def setUp(self) -> None:
        self.standard_preprocessor = DataPreprocessor(
            steps=[
                {'name': 'column_selector',
                 'params': {'max_nan_frac_per_col': 0.2}},
                {'name': 'angle_transform',
                 'params': {'angles': ['Sensor_6']}},
                {'name': 'duplicate_values_to_nan'},
                {'name': 'low_unique_value_filter',}
            ]
        )
        self.another_preprocessor = DataPreprocessor(
            steps=[
                {'name': 'column_selector',
                 'params': {'max_nan_frac_per_col': 0.2}},
                {'name': 'angle_transform',
                 'params': {'angles': ['Sensor_6']}},
                {'name': 'duplicate_values_to_nan',
                 'params': {'n_max_duplicates': 4,
                            'value_to_replace': 0}},
                {'name': 'low_unique_value_filter',
                 'params': {'min_unique_value_count': 1}},
            ]
        )
        # Feature consistent, does not drop columns
        self.fc_preprocessor = DataPreprocessor(
            steps=[
                {'name': 'column_selector', 'enabled': False},
                {'name': 'angle_transform',
                 'params': {'angles': ['Sensor_6']}},
            ]
        )

        # Includes categorical encoder and ffill imputer
        self.preprocessor_with_encoder = DataPreprocessor(
            steps=[
                {'name': 'column_selector', 'params': {'max_nan_frac_per_col': 0.8}},
                {'name': 'ffill_imputer',
                 'params': {'freq': '1Min', 'ffill_limit': 1, 'categorical_features': ['category', 'region']}},
                {'name': 'categorical_encoder',
                 'params': {'categorical_features': ['category', 'region']}},
            ]
        )
        # generate data for standard and feature consistent preprocessor tests
        length = 10  # choose an even number for simplicity
        time_index = pd.date_range(start='1/1/2021', end='10/1/2021', periods=length)
        data = {'Sensor_1': list(range(length)),
                'Sensor_2': [None] + list(range(1, length)),
                'Sensor_3': list(range(int(length / 2))) + [None] * int(length / 2),
                'Sensor_4': [0] + [None] * (length - 1),
                'Sensor_5': [0] * length,
                'Sensor_6': list(range(length))}
        self.test_data1 = pd.DataFrame(index=time_index, data=data)

        # generate data for ts preprocessor tests
        data = {'Sensor_1': list(range(length)),
                'Sensor_2': [None] + list(range(1, length)),
                'Sensor_3': list(range(int(length / 2))) + [None] * int(length / 2),
                'Sensor_4': [0] + [None] * (length - 1),
                'Sensor_5': [0] * length,
                'Sensor_6': list(range(length)),
                'Sensor_7': [0] * 4 + list(range(6))
                }
        self.test_data2 = pd.DataFrame(index=time_index, data=data)
        self.exp_result2 = np.array([[-1.5666989, 0., -1.5666989],
                                     [-1.21854359, -1.63299316, -1.21854359],
                                     [-0.87038828, -1.22474487, -0.87038828],
                                     [-0.52223297, -0.81649658, -0.52223297],
                                     [-0.17407766, -0.40824829, -0.17407766],
                                     [0.17407766, 0., 0.17407766],
                                     [0.52223297, 0.40824829, 0.52223297],
                                     [0.87038828, 0.81649658, 0.87038828],
                                     [1.21854359, 1.22474487, 1.21854359],
                                     [1.5666989, 1.63299316, 1.5666989]])

        # generate data for the extended preprocessor tests
        data = {'Sensor_1': list(range(length)),
                'Sensor_2': [None] + list(range(1, length)),
                'Sensor_3': list(range(int(length / 2))) + [None] * int(length / 2),
                'Sensor_4': [0] + [None] * (length - 1),
                'Sensor_5': [0] * length,
                'Sensor_6': list(range(length)),
                'Sensor_7': [0] * (length - 5) + [1] * (length - 5),
                }
        self.test_data3 = pd.DataFrame(index=time_index, data=data)

        # generate data for tests with categorical encoder and ffill imputer
        self.test_data4 = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 22.1, 23.0],
            'humidity': [60, 62, np.nan, np.nan, 67],
            'category': ['A', 'B', 'A', np.nan, 'B'],
            'region': ['North', 'South', 'North', 'East', 'West'],
            'flowrate': [np.nan, np.nan, np.nan, np.nan, np.nan]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1Min'))

        self.exp_result4 = pd.DataFrame({
            'temperature': [-0.5,  0.1, -1.1,  1.6],
            'humidity': [-1.1, -0.3, -0.3,  1.6],
            'category_A': [ 1. , -1. ,  1. , -1. ],
            'category_B': [-1. ,  1. , -1. ,  1. ],
            'region_North': [ 1. , -1. ,  1. , -1. ],
            'region_South': [-0.6,  1.7, -0.6, -0.6],
            'region_West': [-0.6, -0.6, -0.6,  1.7],
        }, index=pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:01:00', '2024-01-01 00:02:00', '2024-01-01 00:04:00']))

        self.exp_inv4 = pd.DataFrame({
            'temperature': [20.5, 21.3, 19.8, 23.0],
            'humidity': [60.0, 62.0, 62.0, 67.0],
            'category': ['A', 'B', 'A', 'B'],
            'region': ['North', 'South', 'North', 'West']
        }, index=pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:01:00', '2024-01-01 00:02:00', '2024-01-01 00:04:00']))

    def test_transform(self):
        # expected output
        exp_result = np.array([[-1.5666989, 0.],
                               [-1.21854359, -1.63299316],
                               [-0.87038828, -1.22474487],
                               [-0.52223297, -0.81649658],
                               [-0.17407766, -0.40824829],
                               [0.17407766, 0.],
                               [0.52223297, 0.40824829],
                               [0.87038828, 0.81649658],
                               [1.21854359, 1.22474487],
                               [1.5666989, 1.63299316]])
        sincos = np.stack([np.sin(self.test_data1['Sensor_6'] * np.pi / 180.),
                           np.cos(self.test_data1['Sensor_6'] * np.pi / 180.)]).T
        sincos = (sincos - sincos.mean(axis=0)) / sincos.std(axis=0)
        exp_result = np.hstack([exp_result, sincos])

        self.standard_preprocessor.fit(self.test_data1)
        data = self.standard_preprocessor.transform(self.test_data1)

        assert_array_almost_equal(data, exp_result)

    def test_transform_extended(self):
        exp_result = np.array([[-1.5666989, 0., -1.178511],
                               [-1.21854359, -1.63299316, -1.178511],
                               [-0.87038828, -1.22474487, -1.178511],
                               [-0.52223297, -0.81649658, -1.178511],
                               [-0.17407766, -0.40824829, 0.],
                               [0.17407766, 0., 0.942809],
                               [0.52223297, 0.40824829, 0.942809],
                               [0.87038828, 0.81649658, 0.942809],
                               [1.21854359, 1.22474487, 0.942809],
                               [1.5666989, 1.63299316, 0.942809]])
        sincos = np.stack([np.sin(self.test_data3['Sensor_6'] * np.pi / 180.),
                           np.cos(self.test_data3['Sensor_6'] * np.pi / 180.)]).T
        sincos = (sincos - sincos.mean(axis=0)) / sincos.std(axis=0)
        exp_result = np.hstack([exp_result, sincos])

        self.another_preprocessor.fit(self.test_data3)
        data = self.another_preprocessor.transform(self.test_data3)

        assert_array_almost_equal(data, exp_result)

    def test_transform_fc(self):
        exp_result = np.array([[-1.5666989, 0., -2., 0., 0., -1.56912063, 1.06193254],
                               [-1.21854359, -1.63299316, -1., 0., 0., -1.21964717, 1.02462177],
                               [-0.87038828, -1.22474487, 0., 0., 0., -0.87028016, 0.91270083],
                               [-0.52223297, -0.81649658, 1., 0., 0., -0.52112602, 0.72620382],
                               [-0.17407766, -0.40824829, 2., 0., 0., -0.17229111, 0.46518754],
                               [0.17407766, 0., 0., 0., 0., 0.17611831, 0.12973149],
                               [0.52223297, 0.40824829, 0., 0., 0., 0.52399611, -0.28006212],
                               [0.87038828, 0.81649658, 0., 0., 0., 0.87123633, -0.76406849],
                               [1.21854359, 1.22474487, 0., 0., 0., 1.21773319, -1.32214018],
                               [1.5666989, 1.63299316, 0., 0., 0., 1.56338116, -1.95410719]])

        self.fc_preprocessor.fit(self.test_data1)
        data = self.fc_preprocessor.transform(self.test_data1)

        assert_array_almost_equal(data, exp_result)

    def test_transform_with_encoder_ffill(self):
        """Test that the preprocessor with categorical encoder and ffill imputer works as expected."""
        self.preprocessor_with_encoder.fit(self.test_data4)
        transformed = self.preprocessor_with_encoder.transform(self.test_data4)
        self.assertTrue(transformed.round(1).equals(self.exp_result4))

    def test_not_fitted(self):
        with self.assertRaises(NotFittedError):
            self.standard_preprocessor.transform(self.test_data1)

        with self.assertRaises(NotFittedError):
            self.another_preprocessor.transform(self.test_data1)

    def test_inverse_transform(self):
        preprocessor = self.standard_preprocessor
        preprocessor.fit(self.test_data1)

        output = preprocessor.inverse_transform(preprocessor.transform(self.test_data1)).astype(float)
        expected = self.test_data1[['Sensor_1', 'Sensor_2', 'Sensor_6']].astype(float)
        expected.loc[pd.isnull(expected['Sensor_2']), 'Sensor_2'] = 5.

        assert_frame_equal(output.reset_index(drop=True), expected.reset_index(drop=True))

    def test_inverse_transform_extended(self):
        preprocessor = self.another_preprocessor
        preprocessor.fit(self.test_data3)

        output = preprocessor.inverse_transform(
            preprocessor.transform(self.test_data3)
        ).astype(float)
        expected = self.test_data3[['Sensor_1', 'Sensor_2', 'Sensor_6', 'Sensor_7']].astype(float)
        expected.loc[pd.isnull(expected['Sensor_2']), 'Sensor_2'] = 5.
        expected.loc['2021-05-02 08:00:00', 'Sensor_7'] = 0.555556

        assert_frame_equal(
            output.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_inverse_transform_fc(self):
        preprocessor = self.fc_preprocessor
        preprocessor.fit(self.test_data1)

        output = preprocessor.inverse_transform(
            preprocessor.transform(self.test_data1)
        ).astype(float)
        expected = self.test_data1.astype(float)
        expected.loc[pd.isnull(expected['Sensor_2']), 'Sensor_2'] = 5.
        expected.loc[pd.isnull(expected['Sensor_3']), 'Sensor_3'] = 2.
        expected.loc[pd.isnull(expected['Sensor_4']), 'Sensor_4'] = 0.

        assert_frame_equal(
            output.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_steps_mode_no_duplicate_imputer(self) -> None:
        """Providing 'simple_imputer' explicitly should not add a second default imputer."""
        dp = DataPreprocessor(
            steps=[
                {"name": "column_selector", "params": {"max_nan_frac_per_col": 0.2}},
                {"name": "simple_imputer", "params": {"strategy": "median"}},
                {"name": "scaler"},
            ]
        )
        # Count imputers by estimator type
        n_imputers = sum(
            est.__class__.__name__ == "Imputer" for _, est in dp.steps
        )
        self.assertEqual(n_imputers, 1, "There should be exactly one SimpleImputer.")

        # Ensure imputer precedes scaler
        imputer_idx = next(
            i for i, (_, est) in enumerate(dp.steps) if est.__class__.__name__ == "Imputer"
        )
        scaler_idx = next(
            i for i, (_, est) in enumerate(dp.steps)
            if est.__class__.__name__ in {"Scaler"}
        )
        self.assertLess(imputer_idx, scaler_idx, "Imputer must precede scaler.")

    def test_steps_mode_default_imputer_inserted(self) -> None:
        """Omitting 'simple_imputer' should auto-insert a default imputer before the scaler."""
        dp = DataPreprocessor(
            steps=[
                {"name": "column_selector", "params": {"max_nan_frac_per_col": 0.2}},
                {"name": "scaler"},
            ]
        )
        # Exactly one imputer should be present
        n_imputers = sum(
            est.__class__.__name__ == "Imputer" for _, est in dp.steps
        )
        self.assertEqual(n_imputers, 1, "A single default SimpleImputer should be added.")

        # Imputer must be before scaler
        imputer_idx = next(
            i for i, (_, est) in enumerate(dp.steps) if est.__class__.__name__ == "Imputer"
        )
        scaler_idx = next(
            i for i, (_, est) in enumerate(dp.steps)
            if est.__class__.__name__ in {"Scaler"}
        )
        self.assertLess(imputer_idx, scaler_idx, "Default imputer must be inserted before scaler.")

    def test_steps_mode_alias_imputer_is_normalized(self) -> None:
        """Using 'imputer' alias should be normalized to 'simple_imputer' internally."""
        dp = DataPreprocessor(
            steps=[
                {"name": "imputer", "params": {"strategy": "mean"}},  # alias
                {"name": "scaler"},
            ]
        )
        # Named steps should include the canonical 'simple_imputer'
        self.assertIn("simple_imputer", dp.named_steps)

    def test_singleton_violation_raises(self) -> None:
        """Two enabled simple_imputer steps should raise a ValueError."""
        with self.assertRaises(ValueError):
            _ = DataPreprocessor(
                steps=[
                    {"name": "simple_imputer", "params": {"strategy": "mean"}},
                    {"name": "simple_imputer", "params": {"strategy": "median"}},
                    {"name": "standard_scaler"},
                ]
            )

    def test_only_one_scaler_allowed(self) -> None:
        """Defining more than one scaler should raise a ValueError."""
        with self.assertRaises(ValueError):
            _ = DataPreprocessor(
                steps=[
                    {"name": "column_selector", "params": {"max_nan_frac_per_col": 0.2}},
                    {"name": "standard_scaler"},
                    {"name": "minmax_scaler"},
                ]
            )

    def test_inverse_transform_with_encoder_ffill(self):
        """Test that the inverse_transform works correctly with the preprocessor that includes categorical encoder and ffill imputer."""
        self.preprocessor_with_encoder.fit(self.test_data4)
        transformed = self.preprocessor_with_encoder.transform(self.test_data4)
        inversed = self.preprocessor_with_encoder.inverse_transform(transformed)
        print("Inversed DataFrame:\n", inversed)
        self.assertEqual(inversed.shape, self.exp_inv4.shape, "Inverse transform shape mismatch.")
        self.assertTrue(inversed.round(1).equals(self.exp_inv4), "Inverse transform did not return the expected DataFrame.")

class TestDataPreprocessorPipelineWithTimestamp(TestCase):
    def setUp(self) -> None:
        # Pipeline including TimestampTransformer
        self.preprocessor_with_ts = DataPreprocessor(
            steps=[
                {'name': 'column_selector',
                 'params': {'max_nan_frac_per_col': 0.2}},
                {'name': 'timestamp_transformer',
                 'params': {'features': ['second_of_minute', 'minute_of_hour', 'is_weekend']}},
                {'name': 'angle_transform',
                 'params': {'angles': ['Sensor_6']}},
                {'name': 'duplicate_values_to_nan'},
                {'name': 'low_unique_value_filter',}
            ]
        )

        length = 6
        time_index = pd.date_range(start='2021-01-01', periods=length, freq='D')
        data = {
            'Sensor_1': list(range(length)),
            'Sensor_2': [None] * length,
            'Sensor_6': list(range(length)),
        }
        self.test_data_with_ts = pd.DataFrame(index=time_index, data=data)

    def test_fit_transform_with_timestamp(self):
        # Fit and transform with TimestampTransformer
        self.preprocessor_with_ts.fit(self.test_data_with_ts)
        transformed = self.preprocessor_with_ts.transform(self.test_data_with_ts)

        # Ensure transform returns a DataFrame and shape matches expectations
        self.assertIsNotNone(transformed)
        self.assertTrue(len(transformed) > 0)
        self.assertEqual(transformed.shape[0], self.test_data_with_ts.shape[0])

        # Check that timestamp-derived features exist
        self.assertIn('second_of_minute_sine', transformed.columns)
        self.assertIn('second_of_minute_cosine', transformed.columns)
        self.assertIn('minute_of_hour_sine', transformed.columns)
        self.assertIn('minute_of_hour_cosine', transformed.columns)
        self.assertIn('is_weekend', transformed.columns)


class TestDataPreprocessorProtectedFeatures(TestCase):
    """Test that protected features (e.g., conditional features for autoencoders) are never dropped."""

    def setUp(self) -> None:
        length = 10
        time_index = pd.date_range(start='1/1/2021', end='10/1/2021', periods=length)
        # Create test data where 'conditional_feature' would normally be dropped
        data = {
            'normal_feature': list(range(length)),
            'conditional_feature': [1] * length,  # constant - would be dropped by LowUniqueValueFilter
            'high_nan_feature': [None] * 8 + [1, 2],  # 80% NaN - would be dropped by ColumnSelector
            'another_feature': list(range(length)),
        }
        self.test_data = pd.DataFrame(index=time_index, data=data)

    def test_protected_feature_not_dropped_by_low_unique_value_filter(self):
        """Test that a constant conditional feature is protected from LowUniqueValueFilter."""
        preprocessor = DataPreprocessor(
            steps=[
                {'name': 'low_unique_value_filter',
                 'params': {'min_unique_value_count': 2}},
            ]
        )

        # Fit without protection - conditional_feature should be dropped
        preprocessor.fit(self.test_data)
        transformed = preprocessor.transform(self.test_data)
        self.assertNotIn('conditional_feature', transformed.columns)

        # Fit WITH protection - conditional_feature should be kept
        preprocessor_protected = DataPreprocessor(
            steps=[
                {'name': 'low_unique_value_filter',
                 'params': {'min_unique_value_count': 2}},
            ]
        )
        fit_params = {'low_unique_value_filter__protected_features': ['conditional_feature']}
        preprocessor_protected.fit(self.test_data, **fit_params)
        transformed_protected = preprocessor_protected.transform(self.test_data)
        self.assertIn('conditional_feature', transformed_protected.columns,
                      "Protected feature should be kept despite being constant")

    def test_protected_feature_not_dropped_by_column_selector(self):
        """Test that a high-NaN conditional feature is protected from ColumnSelector."""
        preprocessor = DataPreprocessor(
            steps=[
                {'name': 'column_selector',
                 'params': {'max_nan_frac_per_col': 0.5}},  # 50% threshold
            ]
        )

        # Fit without protection - high_nan_feature should be dropped (80% NaN > 50%)
        preprocessor.fit(self.test_data)
        transformed = preprocessor.transform(self.test_data)
        self.assertNotIn('high_nan_feature', transformed.columns)

        # Fit WITH protection - high_nan_feature should be kept
        preprocessor_protected = DataPreprocessor(
            steps=[
                {'name': 'column_selector',
                 'params': {'max_nan_frac_per_col': 0.5}},
            ]
        )
        fit_params = {'column_selector__protected_features': ['high_nan_feature']}
        preprocessor_protected.fit(self.test_data, **fit_params)
        transformed_protected = preprocessor_protected.transform(self.test_data)
        self.assertIn('high_nan_feature', transformed_protected.columns,
                      "Protected feature should be kept despite high NaN percentage")

    def test_protected_features_with_both_filters(self):
        """Test that protected features work with both ColumnSelector and LowUniqueValueFilter."""
        preprocessor = DataPreprocessor(
            steps=[
                {'name': 'column_selector',
                 'params': {'max_nan_frac_per_col': 0.5}},
                {'name': 'low_unique_value_filter',
                 'params': {'min_unique_value_count': 2}},
            ]
        )

        # Protect both problematic features
        fit_params = {
            'column_selector__protected_features': ['high_nan_feature', 'conditional_feature'],
            'low_unique_value_filter__protected_features': ['high_nan_feature', 'conditional_feature']
        }
        preprocessor.fit(self.test_data, **fit_params)
        transformed = preprocessor.transform(self.test_data)

        # Both protected features should be present
        self.assertIn('conditional_feature', transformed.columns,
                      "Constant protected feature should be kept")
        self.assertIn('high_nan_feature', transformed.columns,
                      "High-NaN protected feature should be kept")


class TestComprehensivePipelineFlow(TestCase):
    """Test comprehensive preprocessing pipeline"""

    def setUp(self):
        """Set up test fixtures for comprehensive pipeline testing."""
        self.n_samples = 500
        self.time_index = pd.date_range('2025-01-01', periods=self.n_samples, freq='5min')
        
        # Create realistic test data matching the updated advanced_config.yaml scenario
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            # Numerical features
            'temp_sensor_1': np.random.normal(100, 5, self.n_samples),
            'temp_sensor_2': np.random.normal(85, 3, self.n_samples),
            'pressure_sensor': np.random.normal(50, 2, self.n_samples),
            'flow_rate': np.random.exponential(10, self.n_samples),
            # Categorical features
            'category_col': np.random.choice(['A', 'B', 'C', 'D'], self.n_samples),
            'equipment_type': np.random.choice(['Type1', 'Type2', 'Type3'], self.n_samples),
            # Conditional feature (numerical)
            'operating_condition': np.random.normal(0, 1, self.n_samples),
        }, index=self.time_index)
        
        # Add some NaN values to test imputation
        self.test_data.iloc[50:100, self.test_data.columns.get_loc('temp_sensor_1')] = np.nan
        self.test_data.iloc[200:250, self.test_data.columns.get_loc('pressure_sensor')] = np.nan
        self.test_data.iloc[150:170, self.test_data.columns.get_loc('category_col')] = np.nan

    def _create_config_dict(self, protect_conditional_features: bool = True):
        """Create configuration matching updated advanced_config.yaml structure."""
        return {
            'train': {
                'data_clipping': {
                    'lower_percentile': 0.001,
                    'upper_percentile': 0.999,
                },
                'data_preprocessor': {
                    'steps': [
                        {'name': 'column_selector', 'params': {'max_nan_frac_per_col': 0.9}},
                        {'name': 'low_unique_value_filter', 'params': {'min_unique_value_count': 2}},
                        {'name': 'ffill_imputer', 'params': {'freq': '1Min', 'ffill_limit': 60, 'categorical_features': ['category_col', 'equipment_type']}},
                        {'name': 'categorical_encoder', 'params': {'categorical_features': ['category_col', 'equipment_type']}},
                        {'name': 'scaler', 'params': {'scaler_type': 'standard', 'with_mean': True, 'with_std': True, 'scale_categorical_features': False, 'categorical_features': ['category_col', 'equipment_type']}},
                        {'name': 'timestamp_transformer', 'params': {'features': ['minute_of_hour', 'hour_of_day', 'day_of_week']}}
                    ]
                },
                'autoencoder': {
                    'name': 'ConditionalAutoencoder',
                    'params': {
                        'act': 'prelu',
                        'batch_size': 256,
                        'code_size': 9,
                        'epochs': 10,
                        'last_act': 'linear',
                        'layers': [64, 32],
                        'learning_rate': 0.0001,
                        'loss_name': 'mean_squared_error',
                        'noise': 0.0,
                        'conditional_features': ['operating_condition', 'category_col'],
                        'verbose': 0
                    }
                },
                'protect_conditional_features': protect_conditional_features,
                'anomaly_score': {'name': 'rmse'},
                'threshold_selector': {'name': 'quantile'},
            }
        }

    def test_pipeline_dtype_conversion(self):
        """Test that all dtypes are numerical after preprocessing."""
        config_dict = self._create_config_dict()
        preprocessor = DataPreprocessor(steps=config_dict['train']['data_preprocessor']['steps'])
        
        preprocessor.fit(self.test_data)
        transformed_data = preprocessor.transform(self.test_data)
        
        # Verify all dtypes are numerical after preprocessing
        for dtype in transformed_data.dtypes:
            self.assertTrue(pd.api.types.is_numeric_dtype(dtype),
                          f"Column {dtype.name} should be numeric after preprocessing")

    def test_pipeline_expected_columns(self):
        """Test that expected columns exist after preprocessing."""
        config_dict = self._create_config_dict()
        preprocessor = DataPreprocessor(steps=config_dict['train']['data_preprocessor']['steps'])
        
        preprocessor.fit(self.test_data)
        transformed_data = preprocessor.transform(self.test_data)
        
        # Check expected numerical columns
        expected_numerical = ['temp_sensor_1', 'temp_sensor_2', 'pressure_sensor', 'flow_rate', 'operating_condition']
        for col in expected_numerical:
            self.assertIn(col, transformed_data.columns, f"Expected numerical column {col}")
        
        # Check categorical features were one-hot encoded
        expected_categorical = ['category_col_A', 'category_col_B', 'category_col_C', 'category_col_D',
                               'equipment_type_Type1', 'equipment_type_Type2', 'equipment_type_Type3']
        for col in expected_categorical:
            self.assertIn(col, transformed_data.columns, f"Expected encoded categorical column {col}")
        
        # Check timestamp features
        timestamp_features = ['minute_of_hour_sine', 'minute_of_hour_cosine',
                             'hour_of_day_sine', 'hour_of_day_cosine',
                             'day_of_week_sine', 'day_of_week_cosine']
        for col in timestamp_features:
            self.assertIn(col, transformed_data.columns, f"Expected timestamp feature {col}")

    def test_pipeline_nan_handling(self):
        """Test that NaN values are properly handled."""
        config_dict = self._create_config_dict()
        preprocessor = DataPreprocessor(steps=config_dict['train']['data_preprocessor']['steps'])
        
        preprocessor.fit(self.test_data)
        transformed_data = preprocessor.transform(self.test_data)
        
        # Verify no NaN values remain after preprocessing
        self.assertFalse(transformed_data.isna().any().any(),
                        "Transformed data should not contain NaN values after preprocessing")

    def test_pipeline_shape_transformation(self):
        """Test that data shape is appropriate after preprocessing."""
        config_dict = self._create_config_dict()
        preprocessor = DataPreprocessor(steps=config_dict['train']['data_preprocessor']['steps'])
        
        preprocessor.fit(self.test_data)
        transformed_data = preprocessor.transform(self.test_data)
        
        # Verify data shape (should have more columns than original due to encoding)
        self.assertGreater(transformed_data.shape[1], self.test_data.shape[1],
                          "Transformed data should have more columns due to categorical encoding and timestamp features")

    def test_pipeline_inverse_transform(self):
        """Test that inverse transform works correctly."""
        config_dict = self._create_config_dict()
        preprocessor = DataPreprocessor(steps=config_dict['train']['data_preprocessor']['steps'])
        
        preprocessor.fit(self.test_data)
        transformed_data = preprocessor.transform(self.test_data)
        inverse_data = preprocessor.inverse_transform(transformed_data)
        
        # Verify inverse transform preserves row count of the transformed data
        # (the ffill_imputer may drop rows that still contain NaNs after filling)
        self.assertEqual(inverse_data.shape[0], transformed_data.shape[0],
                        "Inverse transform should preserve the transformed data's row count")

    def test_pipeline_fault_detector_integration(self):
        """Test integration with FaultDetector."""
        config_dict = self._create_config_dict()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dict=config_dict)
            fault_detector = FaultDetector(config=config, model_directory=tmpdir)
            
            # Create normal index (all normal for training)
            normal_index = pd.Series([True] * self.n_samples, index=self.time_index)
            
            # Fit the model
            result = fault_detector.fit(
                sensor_data=self.test_data,
                normal_index=normal_index,
                save_models=False
            )
            
            # Verify model was trained successfully
            self.assertIsNotNone(fault_detector.autoencoder,
                               "Autoencoder should be created and trained")
            self.assertEqual(fault_detector.autoencoder.__class__.__name__, 'ConditionalAE',
                           "Autoencoder should be ConditionalAE as specified in config")
            self.assertIsNotNone(result.train_recon_error,
                               "Training reconstruction error should be available")

    def test_resolution_of_conditions(self):
        """Test pipeline handling of missing conditional features."""
        config_dict = self._create_config_dict()
        
        # Create data without the conditional feature
        test_data_no_cond = self.test_data.drop(columns=['category_col'])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dict=config_dict)
            fault_detector = FaultDetector(config=config, model_directory=tmpdir)
        
            # Create normal index (all normal for training)
            normal_index = pd.Series([True] * self.n_samples, index=self.time_index)
        
            # Fit the model
            result = fault_detector.fit(
                sensor_data=test_data_no_cond,
                normal_index=normal_index,
                save_models=False
            )

        # Verify that autoencoder conditions are updated with available feature names
        self.assertIn('operating_condition', fault_detector.autoencoder.conditional_features,
                      "Autoencoder should have 'operating_condition' as a conditional feature")
        self.assertNotIn('category_col', fault_detector.autoencoder.conditional_features,
                         "Autoencoder should not have 'category_col' as a conditional feature since it's missing")

    def test_declared_category_not_as_condition(self):
        """Test pipeline handling of declared categorical features not being declared as conditional features."""
        config_dict = self._create_config_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dict=config_dict)
            fault_detector = FaultDetector(config=config, model_directory=tmpdir)

            # Create normal index (all normal for training)
            normal_index = pd.Series([True] * self.n_samples, index=self.time_index)

            # Fit the model
            result = fault_detector.fit(
                sensor_data=self.test_data,
                normal_index=normal_index,
                save_models=False
            )

        self.assertIn('equipment_type_Type1', fault_detector.data_preprocessor.get_feature_names_out(),
                      "Data preprocessor should have 'equipment_type' as a feature")
        self.assertNotIn('equipment_type', fault_detector.autoencoder.conditional_features,
                            "Autoencoder should not have 'equipment_type' as a conditional feature since it's not declared as such.")

    def test_declared_condition_not_as_category(self):
        """Test pipeline handling of declared conditional features not being declared as categorical features."""
        config_dict = self._create_config_dict(False)

        # Remove 'category_col' from categorical features in the config to simulate it being declared as conditional but not categorical
        config_dict['train']['data_preprocessor']['steps'][2]['params']['categorical_features'] = ['equipment_type']  # Only 'equipment_type' is categorical
        config_dict['train']['data_preprocessor']['steps'][3]['params']['categorical_features'] = ['equipment_type']  # Only 'equipment_type' is categorical
        config_dict['train']['data_preprocessor']['steps'][4]['params']['categorical_features'] = ['equipment_type']  # Only 'equipment_type' is categorical

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dict=config_dict)
            fault_detector = FaultDetector(config=config, model_directory=tmpdir)

            # Create normal index (all normal for training)
            normal_index = pd.Series([True] * self.n_samples, index=self.time_index)

            # Fit the model
            result = fault_detector.fit(
                sensor_data=self.test_data,
                normal_index=normal_index,
                save_models=False
            )

        # Verify that autoencoder conditions are updated properly.
        self.assertFalse(any('category_col' in col for col in fault_detector.data_preprocessor.get_feature_names_out()),
                        "Data preprocessor should not have any column containing substring 'category_col' since it's not declared as categorical.")

    def test_protected_features_are_not_dropped_in_pipeline(self):
        """Test that protected features are not dropped in the comprehensive pipeline."""
        config_dict = self._create_config_dict()
        config = Config(config_dict=config_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            fault_detector = FaultDetector(config=config, model_directory=tmpdir)
        
            # Create normal index (all normal for training)
            normal_index = pd.Series([True] * self.n_samples, index=self.time_index)
        
            # Fit the model
            result = fault_detector.fit(
                sensor_data=self.test_data,
                normal_index=normal_index,
                save_models=False
            )

        protected_features = config_dict['train']['autoencoder']['params']['conditional_features']
    
        output_features = fault_detector.data_preprocessor.get_feature_names_out()
        available_protected = [f for f in protected_features if f in self.test_data.columns]
        missing_features = [f for f in available_protected
                            if not any(f in col for col in output_features)]
        self.assertEqual(len(missing_features), 0, f"Protected features {available_protected} should not be dropped in the pipeline.")
