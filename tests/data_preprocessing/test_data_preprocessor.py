from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from sklearn.utils.validation import check_is_fitted, NotFittedError

from energy_fault_detector.data_preprocessing.data_preprocessor import DataPreprocessor


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
        # legacy set up
        self.standard_preprocessor_old = DataPreprocessor(
            max_nan_frac_per_col=0.2,
            imputer_strategy='mean',
            angles=['Sensor_6'],
            include_column_selector=True,
            include_duplicate_value_to_nan=False,
            include_low_unique_value_filter=True,
            min_unique_value_count=2,
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
        # legacy set up
        self.another_preprocessor_old = DataPreprocessor(
            max_nan_frac_per_col=0.2,
            imputer_strategy='mean',
            min_unique_value_count=1,
            angles=['Sensor_6'],
            n_max_duplicates=4,
            value_to_replace=0,
            include_column_selector=True,
            include_duplicate_value_to_nan=True,
            include_low_unique_value_filter=True
        )
        # Feature consistent, does not drop columns
        self.fc_preprocessor = DataPreprocessor(
            steps=[
                {'name': 'column_selector', 'enabled': False},
                {'name': 'angle_transform',
                 'params': {'angles': ['Sensor_6']}},
            ]
        )
        # legacy set up
        self.fc_preprocessor_old = DataPreprocessor(
            imputer_strategy='mean',
            angles=['Sensor_6'],
            include_low_unique_value_filter=False,
            include_duplicate_value_to_nan=False,
            include_column_selector=False
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

    def test_fit_standard_preprocessor(self):
        self.standard_preprocessor_old.fit(self.test_data1)
        check_is_fitted(self.standard_preprocessor_old.named_steps['scaler'])
        self.standard_preprocessor.fit(self.test_data1)
        check_is_fitted(self.standard_preprocessor.named_steps['scaler'])

    def test_fit_extended(self):
        self.another_preprocessor_old.fit(self.test_data3)
        check_is_fitted(self.another_preprocessor_old.named_steps['scaler'])

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

        self.another_preprocessor_old.fit(self.test_data3)
        data = self.another_preprocessor_old.transform(self.test_data3)

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

        self.fc_preprocessor_old.fit(self.test_data1)
        data = self.fc_preprocessor_old.transform(self.test_data1)

        assert_array_almost_equal(data, exp_result)

    def test_not_fitted(self):
        with self.assertRaises(NotFittedError):
            self.standard_preprocessor.transform(self.test_data1)

        with self.assertRaises(NotFittedError):
            self.another_preprocessor.transform(self.test_data1)

    def test_inverse_transform(self):
        for preprocessor in [self.standard_preprocessor, self.standard_preprocessor_old]:
            preprocessor.fit(self.test_data1)

            output = preprocessor.inverse_transform(
                preprocessor.transform(self.test_data1)
            ).astype(float)
            expected = self.test_data1[['Sensor_1', 'Sensor_2', 'Sensor_6']].astype(float)
            expected.loc[pd.isnull(expected['Sensor_2']), 'Sensor_2'] = 5.

            assert_frame_equal(
                output.reset_index(drop=True),
                expected.reset_index(drop=True),
            )

    def test_inverse_transform_extended(self):
        for preprocessor in [self.another_preprocessor, self.another_preprocessor_old]:
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
        for preprocessor in [self.fc_preprocessor, self.fc_preprocessor_old]:
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
