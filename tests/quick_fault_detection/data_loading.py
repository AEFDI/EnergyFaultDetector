import os
import unittest
import numpy as np
import pandas as pd
from energy_fault_detector.quick_fault_detection.data_loading import detect_encoding, detect_separator, \
    read_csv_file, get_sensor_data, get_boolean_feature, load_data, load_train_test_data

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..')


class TestQuickFaultDetectionDataLoading(unittest.TestCase):

    def setUp(self) -> None:
        self.num_data = 100
        time_index = pd.date_range(start='01-01-2025',
                                   freq='10min',
                                   periods=self.num_data,
                                   name='time_stamp')
        self.csv_path_1 = os.path.join(PROJECT_ROOT, 'tests/test_data/dummy_csv_1.csv')
        self.csv_path_2 = os.path.join(PROJECT_ROOT, 'tests/test_data/dummy_csv_2.csv')
        df = pd.DataFrame(data=np.zeros(shape=(self.num_data, 100)),
                          index=time_index)
        df['string_feature'] = len(df) * ["€"]  # use this column to test encoding detection
        df_2 = pd.DataFrame(data=np.zeros(shape=(self.num_data, 100)),
                            index=time_index)

        self.csv_path_3 = os.path.join(PROJECT_ROOT, 'tests/test_data/dummy_csv_3.csv')
        df_3 = pd.DataFrame(data=np.zeros(shape=(self.num_data, 100)),
                            index=time_index)
        self.normal_index = self.num_data * [True]
        df_3['status'] = self.normal_index
        self.train_test = int(self.num_data / 2) * [True] + (self.num_data - int(self.num_data / 2)) * [False]
        df_3['train_test'] = self.train_test

        df.to_csv(self.csv_path_1, encoding='utf-8', sep=';')
        df_2.to_csv(self.csv_path_2, encoding='ascii', sep=',')
        df_3.to_csv(self.csv_path_3, sep=';')

    def test_detect_encoding(self):
        encoding_1 = detect_encoding(self.csv_path_1)
        encoding_2 = detect_encoding(self.csv_path_2)
        self.assertEqual(encoding_1, 'utf-8')
        self.assertEqual(encoding_2, 'ascii')

    def test_detect_separator(self):
        sep = detect_separator(self.csv_path_1, encoding='utf-8')
        self.assertEqual(sep, ';')
        sep = detect_separator(self.csv_path_2, encoding='ascii')
        self.assertEqual(sep, ',')

    def test_read_csv_file(self):
        df = read_csv_file(self.csv_path_2, time_column_name='time_stamp')
        self.assertEqual(df.shape, (100, 100))
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))

    def test_get_sensor_data(self):
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['3', '4'], 'col3': ['strings', 'strings']})
        sensor_data = get_sensor_data(df)
        self.assertEqual(sensor_data.shape[1], 2)

    def test_get_boolean_feature(self):
        df = pd.DataFrame({'bool_col': [1, 0, 1, 0]})
        boolean_feature = get_boolean_feature(df, bool_data_column_name='bool_col')
        self.assertTrue(boolean_feature.equals(pd.Series([True, False, True, False])))

        df = pd.DataFrame({'bool_col': [9, 8, 9, 8]})
        boolean_feature = get_boolean_feature(df, bool_data_column_name='bool_col', boolean_mapping={9: True, 8: False})
        self.assertTrue(boolean_feature.equals(pd.Series([True, False, True, False])))

    def test_load_data(self):
        train_data, normal_index, test_data = load_data(csv_data_path=self.csv_path_3,
                                                        train_test_column_name='train_test',
                                                        status_data_column_name='status')
        self.assertEqual(np.sum(self.train_test), train_data.shape[0])
        self.assertEqual(len(self.train_test) - np.sum(self.train_test), test_data.shape[0])
        self.assertEqual(normal_index.shape[0], train_data.shape[0])

    def test_load_train_test_data(self):
        train_data, normal_index, test_data = load_train_test_data(csv_data_path=self.csv_path_1,
                                                                   csv_test_data_path=self.csv_path_2,
                                                                   time_column_name='time_stamp')
        self.assertEqual(train_data.shape[0], self.num_data)
        self.assertEqual(test_data.shape[0], self.num_data)
        self.assertTrue(isinstance(train_data.index, pd.DatetimeIndex))
        self.assertTrue(isinstance(test_data.index, pd.DatetimeIndex))
        self.assertEqual(normal_index, None)

