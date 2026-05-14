"""
TODO:
1	get_sensor_data mutates input DataFrame
2	validate_mapping mutates input dict
3	Call order in load_data makes boolean extraction fragile
4	Empty file handling in detect_separator
"""

import os
import logging
from typing import Tuple, Union, Optional, Any
import chardet
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger('energy_fault_detector')


def detect_encoding(file_path: str) -> str:
    """ Uses chardet to detect the encoding of the file """
    file_size = os.path.getsize(file_path)
    bytes_to_read = min(file_size, 10000)
    with open(file_path, 'rb') as f:
        raw_data = f.read(bytes_to_read)  # Read first 10,000 bytes
        result = chardet.detect(raw_data)
        return result['encoding']


def detect_separator(file_path: str, encoding: str) -> str:
    """ Estimates the seperator based on the number of occurrences. """
    with open(file_path, 'r', encoding=encoding) as f:
        first_line = f.readline()
        possible_separators = [',', ';', '\t', '|']
        separator_count = {sep: first_line.count(sep) for sep in possible_separators}
        return max(separator_count, key=separator_count.get)


def read_csv_file(csv_data_path: str, time_column_name: Union[str, None]) -> pd.DataFrame:
    """ Checks if the csv file exists and extracts a dataframe after determining the file encoding and the seperator.
    if time_column_name is not None this column is set as index. If csv_data_path does not point to a file an
    exception is raised.

    Args:
        csv_data_path (str): Path to a csv-file containing data.
        time_column_name (Union[str, None]): Name of the time stamp column in the data.

    Returns:
        pd.DataFrame: Contents of csv_data_path
    """
    if os.path.isfile(csv_data_path):
        encoding = detect_encoding(file_path=csv_data_path)
        sep = detect_separator(file_path=csv_data_path, encoding=encoding)
        df = pd.read_csv(csv_data_path, sep=sep, encoding=encoding)
        # Cast column names to string
        df.columns = df.columns.astype(str)
        if time_column_name is not None:
            df[time_column_name] = pd.to_datetime(df[time_column_name])
            df = df.set_index(time_column_name)
        return df
    else:
        raise ValueError(f'The specified data path {csv_data_path} is not valid.')


def get_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Selects all features from df which are either numeric features or features which can be casted to numeric
    datatypes """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    other_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    castable_cols = []
    for column in other_cols:
        try:
            # errors='coerce' and checking for NaNs could also be used under the condition that there are no NaNs
            # in the data
            df[column] = pd.to_numeric(df[column])
            castable_cols.append(column)
        except ValueError:
            pass
    return df[numeric_cols + castable_cols]


def get_boolean_feature(df: pd.DataFrame, bool_data_column_name: Optional[str] = None,
                        boolean_mapping: Optional[dict] = None) -> Optional[pd.Series]:
    """ Extracts a boolean feature from the dataframe df. If the specified column does not contain boolean values and
    boolean mapping is provided, casting of the column is attempted. If the casting fails None is returned.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        bool_data_column_name (Optional[str]): Name of the column where the boolean feature is located. Default is None.
        boolean_mapping (Optional[dict]): Dictionary defining a mapping of non-boolean values in bool_data_column to
            booleans

    Returns:
        Pandas Series containing booleans or None if casting steps fail or if the column does not exist.
    """
    if bool_data_column_name is not None:
        if bool_data_column_name in df.columns:
            feature = apply_boolean_mapping(data=df[bool_data_column_name], mapping=boolean_mapping)
            try:
                boolean_feature = feature.astype(bool)
            except ValueError:
                logger.warning(f'Boolean feature {bool_data_column_name} could not be evaluated, since data can not be '
                               f'cast to boolean.')
                boolean_feature = None
        else:
            logger.warning(f'{bool_data_column_name} is not a feature of the provided dataset.')
            boolean_feature = None
    else:
        boolean_feature = None
    return boolean_feature


def apply_boolean_mapping(data: pd.Series, mapping: Optional[dict]) -> pd.Series:
    """ Applies the provided boolean mapping to the data if the mapping is valid. """
    if mapping is None:
        return data
    else:
        is_valid = validate_mapping(mapping=mapping, data_type=data.dtype)
        if is_valid:
            return data.replace(mapping)
        else:
            raise ValueError('Invalid Mapping for boolean data was provided.')


def validate_mapping(mapping: dict, data_type: Any) -> bool:
    """ Validates the data types of keys and values in the mapping dictionary. """
    for key, value in mapping.items():
        try:
            # test if casting the key to data_type is possible
            pd.Series([key], dtype=data_type)
        except ValueError:
            return False
        if not isinstance(value, bool):
            try:
                # test if casting the value to bool is possible and replace the value with a bool
                new_value = bool(value)
                mapping[key] = new_value
            except ValueError:
                return False
    return True


def load_data(csv_data_path: str, train_test_column_name: Optional[str] = None,
              train_test_mapping: Optional[dict] = None, time_column_name: Optional[str] = None,
              status_data_column_name: Optional[str] = None, status_mapping: Optional[dict] = None
              ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """ Load data from csv_data_path and split it into numerical data and normal_index. Optionally performs a train test
    split is performed if train_test_column_name is not None.

    Args:
        csv_data_path (str): Path to a csv-file containing tabular data which must contain training data for the
            autoencoder. This data can also contain test data for evaluation, but in this case train_test_column and
            optionally train_test_mapping must be provided.
        train_test_column_name (Optional str): Name of the column which specifies which part of the data in
            csv_data_path is training data and which is test data. If this column does not contain boolean values or
            values which can be cast into boolean values, then train_test_mapping must be provided. Default is None.
        train_test_mapping (Optional dict): Dictionary which defines a mapping of all non-boolean values in the
            train_test_column to booleans. Keys of the dictionary must be values from train_test_column, and they must
            have a datatype which can be cast to the datatype of train_test_column. Values of this dictionary must be
            booleans or at least castable to booleans. Default is None.
        time_column_name (Optional str): Name of the column containing time stamp information.
        status_data_column_name (Optional str): Name of the column which specifies the status of each row in
            csv_data_path. The status is used to define which rows represent normal behavior (i.e. which rows can be
            used for the autoencoder training) and which rows contain anomalous behavior. If this column does not
            contain boolean values, status_mapping must be provided. If status_data_column_name is not provided, all
            rows in csv_data_path are assumed to be normal and a warning will be logged. Default is None.
        status_mapping: Dictionary which defines a mapping of all non-boolean values in the
            status_data_column to booleans. Keys of the dictionary must be values from status_data_column, and they must
            have a datatype which can be cast to the datatype of train_test_column. Values of this dictionary must be
            booleans or at least castable to booleans. Default is None.

    Returns: tuple
        train_data: numerical data of the training section of the data.
        normal_index: boolean series specifying which samples of the training data are normal.
        test_data (Union[pd.DataFrame, None]): numerical data of the test section of the data. Is only not None if
        train_test_column_name is given.
    """
    df = read_csv_file(csv_data_path=csv_data_path, time_column_name=time_column_name)
    # train test data split assumes that True stands for training data and False for prediction data
    train_test_split = get_boolean_feature(df=df, bool_data_column_name=train_test_column_name,
                                           boolean_mapping=train_test_mapping)
    sensor_data = get_sensor_data(df=df)
    normal_index = get_boolean_feature(df=df, bool_data_column_name=status_data_column_name,
                                       boolean_mapping=status_mapping)
    if normal_index is None:
        logger.warning('No status information was provided for the training data. The normal behavior model will be '
                       'trained on the entire provided training data which may lead to distorted normal behavior '
                       'models if the training data does not contain exclusively normal behavior.')
    if train_test_split is not None:
        train_data = sensor_data[train_test_split]
        if normal_index is not None:
            normal_index = normal_index[train_test_split]
        test_data = sensor_data[~train_test_split]
    else:
        train_data = sensor_data
        test_data = None
    return train_data, normal_index, test_data


def load_train_test_data(csv_data_path: str, csv_test_data_path: Optional[str] = None,
                         train_test_column_name: Optional[str] = None, train_test_mapping: Optional[dict] = None,
                         time_column_name: Optional[str] = None, status_data_column_name: Optional[str] = None,
                         status_mapping: Optional[dict] = None
                         ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """ This function extracts numerical training and test data from csv-files and provides a normal index for the
    training data. If multiple sources of test data are given, both sources will be fused into one test data set.
    If no test data is provided an exception is raised.

    Args:
        csv_data_path (str): Path to a csv-file containing tabular data which must contain training data for the
            autoencoder. This data can also contain test data for evaluation, but in this case train_test_column and
            optionally train_test_mapping must be provided.
        csv_test_data_path (Optional str): Path to a csv file containing test data for evaluation. If test data is
            provided in both ways (i.e. via csv_test_data_path and in csv_data_path + train_test_column) then both test
            data sets will be fused into one. Default is None.
        train_test_column_name (Optional str): Name of the column which specifies which part of the data in
            csv_data_path is training data and which is test data. If this column does not contain boolean values or
            values which can be cast into boolean values, then train_test_mapping must be provided. Default is None.
        train_test_mapping (Optional dict): Dictionary which defines a mapping of all non-boolean values in the
            train_test_column to booleans. Keys of the dictionary must be values from train_test_column, and they must
            have a datatype which can be cast to the datatype of train_test_column. Values of this dictionary must be
            booleans or at least castable to booleans. Default is None.
        time_column_name (Optional str): Name of the column containing time stamp information.
        status_data_column_name (Optional str): Name of the column which specifies the status of each row in
            csv_data_path. The status is used to define which rows represent normal behavior (i.e. which rows can be
            used for the autoencoder training) and which rows contain anomalous behavior. If this column does not
            contain boolean values, status_mapping must be provided. If status_data_column_name is not provided, all
            rows in csv_data_path are assumed to be normal and a warning will be logged. Default is None.
        status_mapping: Dictionary which defines a mapping of all non-boolean values in the
            status_data_column to booleans. Keys of the dictionary must be values from status_data_column, and they must
            have a datatype which can be cast to the datatype of train_test_column. Values of this dictionary must be
            booleans or at least castable to booleans. Default is None.

    Returns: tuple
        train_data (pd.DataFrame): Contains training data for the AnomalyDetector (only numeric values).
        train_normal_index (pd.Series): Contains boolean information about which rows of train_data are normal and which
        contain anomalous behavior.
        test_data (pd.DataFrame): Contains test data for the AnomalyDetector (only numeric values).
    """
    train_data, train_normal_index, test_data = load_data(csv_data_path=csv_data_path,
                                                          time_column_name=time_column_name,
                                                          status_data_column_name=status_data_column_name,
                                                          status_mapping=status_mapping,
                                                          train_test_column_name=train_test_column_name,
                                                          train_test_mapping=train_test_mapping)
    # normal index for prediction data can be ignored
    if csv_test_data_path is not None:
        separated_test_data, _, _ = load_data(csv_data_path=csv_test_data_path, time_column_name=time_column_name,
                                              status_data_column_name=None)
    else:
        separated_test_data = None

    if test_data is None and separated_test_data is None:
        raise ValueError('Neither separate test data nor a train-test-split was provided.')

    if test_data is not None and separated_test_data is not None:
        logger.warning('csv prediction data path and train test column name are both specified. Since there are two'
                       'sources of prediction data, the data will be concatenated into one prediction data set.')
        test_data = pd.concat([test_data, separated_test_data])
    elif separated_test_data is not None:
        test_data = separated_test_data
    return train_data, train_normal_index, test_data
