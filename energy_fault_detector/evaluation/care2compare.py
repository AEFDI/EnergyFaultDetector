"""Care to compare (https://doi.org/10.5281/zenodo.10958774) dataset loader."""

import logging
from pathlib import Path
from typing import Iterator, Tuple, List, Union, Dict

import pandas as pd

from energy_fault_detector.config import Config
from energy_fault_detector.utils.data_downloads import download_zenodo_data

logger = logging.getLogger('energy_fault_detector')


class Care2CompareDataset:
    """Loads the Care to Compare Dataset (accompanying paper https://doi.org/10.3390/data9120138).

    The data can be downloaded either manually from https://doi.org/10.5281/zenodo.14958989 (in this case specify
    `path`) or it can be downloaded automatically by setting download_dataset to True.

    All data is loaded into memory, which might be problematic for large datasets (consider using DataLoader classes of
    TensorFlow and PyTorch in that case).

    By default, only the averages are read. See statistics argument of the data loading methods.

    Methods:
        get_event_info: Returns event info for a given envent ID
        iter_datasets: Reads datasets and yields the resulting training and test DataFrames while iterating over
            event IDs.
        format_event_dataset: Extracts normal_index from a loaded dataset and returns normal_index and sensor_data.
        iter_formatted_datasets: Reads datasets, extracts normal_index and yields the resulting train and test
            DataFrames as well as the normal_indexes while iterating over event IDs.
        load_event_dataset: Reads dataset specified by event_id and returns training and test data.
        load_and_format_event_dataset: Reads dataset specified by event_id and returns training and test data as well as
            the corresponding normal indexes.
        iter_train_datasets_per_asset: Reads datasets and yields the resulting training DataFrames while
            iterating over asset IDs and aggregating event IDs for the same assets.
        update_c2c_config: Updates a specified FaultDetector config based on provided feature descriptions.

    Args:
        path (Path): The directory path where the dataset is located.
        download_dataset (bool): If True the Care to Compare dataset is automatically downloaded and unzipped.
    """

    def __init__(self, path: Union[Path, str] = "./CARE_To_Compare", download_dataset: bool = False) -> None:
        """Initialize the Care2CompareDataset class."""
        if download_dataset:
            logger.info("Downloading CARE to Compare dataset (~5 GB) from zenodo. Depending on your internet "
                        "connection this can take some time. Additionally the downloaded zip-file will be unzipped "
                        "(~20 GB) which can also take longer.")
            path = download_zenodo_data(identifier="10.5281/zenodo.15846963", dest=path, overwrite=True)
        self.path: Path = Path(path)

        self.wind_farms: Dict[str, Path] = {
            'A': self.path / 'Wind Farm A',
            'B': self.path / 'Wind Farm B',
            'C': self.path / 'Wind Farm C',
        }

        self.feature_descriptions: Dict[str, pd.DataFrame] = self._load_feature_descriptions()
        self.event_info_all: pd.DataFrame = self._load_event_info()

    def get_event_info(self, event_id: int) -> pd.Series:
        """Get event info of provided event ID."""
        return self.event_info_all[self.event_info_all['event_id'] == event_id].iloc[0]

    def iter_datasets(self, wind_farm: str = None, test_only: bool = False, statistics: List[str] = None,
                      index_column: str = 'id') -> Iterator[Tuple]:
        """Iterate over all datasets, optionally for a specific wind farm.

        Args:
            wind_farm (str, optional): Wind farm name. If not provided, all datasets will be loaded.
            test_only (bool, optional): If true, only test dataset will be returned.
            statistics (list[str], optional): describes which statistic features will be selected. Possible statistics
                are 'avg', 'min', 'max' and 'std'. If None are provided it defaults to ['avg'].
            index_column (str): The name of the index column, either 'time_stamp' or 'id'. Defaults to 'id'.

        Yields:
            Iterator[Tuple]: If test_only=False, yields a tuple of train and test data and event id.
                             If test_only=True, yields a tuple of test data and event id.
        """
        # Determine which wind farms to process
        if wind_farm is None:
            wind_farms = self.wind_farms.keys()
        else:
            wind_farms = [wind_farm]

        for wf in wind_farms:
            # Get all event IDs for the specified wind farm
            event_ids = self.event_info_all.loc[self.event_info_all['wind_farm'] == wf, 'event_id'].values
            for event_id in event_ids:
                # Load the dataset for the current event ID
                yield self.load_event_dataset(event_id=event_id,
                                              statistics=statistics,
                                              test_only=test_only,
                                              index_column=index_column), event_id

    @staticmethod
    def format_event_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """ Splits a given dataset into normal_index and numerical sensor data"""
        normal_index = data['status_type_id'] == 0
        sensor_data = data.drop(['asset_id', 'id', 'time_stamp', 'status_type_id'], axis=1, errors='ignore')
        return sensor_data, normal_index

    def iter_formatted_datasets(self, wind_farm: str = None, test_only: bool = False, statistics: List[str] = None,
                                index_column: str = 'id') -> Iterator[Tuple]:
        """Iterate over all datasets, optionally for a specific wind farm and format the dataset by splitting it into
        boolean normal_index and numerical sensor data.

        Args:
            wind_farm (str, optional): Wind farm name. If not provided, all datasets will be loaded.
            test_only (bool, optional): If true, only test dataset will be returned.
            statistics (list[str], optional): describes which statistic features will be selected. Possible statistics
                are 'avg', 'min', 'max' and 'std'. If None are provided it defaults to ['avg'].
            index_column (str): The name of the index column, either time_stamp or id. Defaults to 'id'.

        Yields:
            Iterator[Tuple]: If test_only=False, yields a tuple of train_sensor_data, train_normal_index,
                             test_sensor_data, test_normal_index and event id.
                             If test_only=True, yields a tuple of test_sensor_data, test_normal_index and event id.
        """

        for tup in self.iter_datasets(wind_farm=wind_farm, test_only=test_only,
                                      statistics=statistics, index_column=index_column):
            if not test_only:
                train_sensor_data, train_normal_index = self.format_event_dataset(tup[0])
                test_sensor_data, test_normal_index = self.format_event_dataset(tup[1])
                yield train_sensor_data, train_normal_index, test_sensor_data, test_normal_index, tup[2]
            else:
                test_sensor_data, test_normal_index = self.format_event_dataset(tup[0])
                yield test_sensor_data, test_normal_index, tup[1]

    def load_event_dataset(self, event_id: int, test_only: bool = False, statistics: List[str] = None,
                           index_column: str = 'id') -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """Load train and test datasets for a specific event ID.

        Args:
            event_id (int): The event ID for which to retrieve datasets.
            test_only (bool, optional): If true, only the test dataset will be returned.
            statistics (list[str], optional): describes which statistic features will be selected. Possible statistics
                are 'avg', 'min', 'max' and 'std'. If None are provided it defaults to ['avg'].
            index_column (str): The name of the index column, either time_stamp or id. Defaults to 'id'.

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
                If test_only is False, returns a tuple of training and testing datasets.
                If test_only is True, returns only the test dataset.
        """
        # Load the dataset for the specific event ID
        dataset = self._load_event_dataset(event_id, statistics, index_column=index_column)
        # Separate test data
        x_test = dataset[dataset['train_test'] == 'prediction'].drop('train_test', axis=1)
        if test_only:
            return x_test
        # Separate train data
        x_train = dataset[dataset['train_test'] == 'train'].drop('train_test', axis=1)
        return x_train, x_test

    def load_and_format_event_dataset(self, event_id: int, statistics: List[str] = None, test_only: bool = False,
                                      index_column: str = 'id'
                                      ) -> Union[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Load train and test datasets for a specific event ID and split them up into boolean normal index and
        numerical sensordata

        Args:
            event_id (int): The event ID for which to retrieve datasets.
            test_only (bool, optional): If true, only the test dataset will be returned.
            statistics (list[str], optional): describes which statistic features will be selected. Possible statistics
                are 'avg', 'min', 'max' and 'std'. If None are provided it defaults to ['avg'].
            index_column (str): The name of the index column, either time_stamp or id. Defaults to 'id'.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
                If test_only=False, yields a tuple of train_sensor_data, train_status,
                    test_sensor_data and test_status.
                If test_only=True, yields a tuple of test_sensor_data and test_status.
        """
        tup = self.load_event_dataset(event_id=event_id, test_only=test_only, statistics=statistics,
                                      index_column=index_column)
        if not test_only:
            train_sensor_data, train_status = self.format_event_dataset(data=tup[0])
            test_sensor_data, test_status = self.format_event_dataset(data=tup[1])
            return train_sensor_data, train_status, test_sensor_data, test_status
        else:
            test_sensor_data, test_status = self.format_event_dataset(data=tup)
            return test_sensor_data, test_status

    def iter_train_datasets_per_asset(self, wind_farm: str = None, statistics: List[str] = None,
                                      index_column: str = 'id') -> Iterator[Tuple[pd.DataFrame, int, List[int]]]:
        """Iterate over all asset IDs to generate a training dataset, optionally for a specific wind farm.

        Args:
            wind_farm (str, optional): Wind farm name. If not provided, all assets will be considered.
            statistics (list[str], optional): describes which statistic features will be selected. Possible statistics
                are 'avg', 'min', 'max' and 'std'. If None are provided it defaults to ['avg'].
            index_column (str): The name of the index column, either time_stamp or id. Defaults to 'id'.

        Yields:
            Iterator[Tuple[pd.DataFrame, int, List[int]]]: Yields a tuple containing the training dataset, asset ID,
                and list of event IDs for this asset.
        """
        # Determine which wind farms to process
        if wind_farm is None:
            wind_farms = self.wind_farms.keys()
        else:
            wind_farms = [wind_farm]

        for wf in wind_farms:
            # Filter events for the current wind farm
            events = self.event_info_all[self.event_info_all['wind_farm'] == wf]
            asset_ids = events['asset_id'].unique()

            for asset_id in asset_ids:
                # Filter events for the current asset ID
                events_asset_id = events[events['asset_id'] == asset_id]
                event_ids = events_asset_id['event_id'].values

                data = []
                for event_id in event_ids:
                    # Load the dataset for the current event ID
                    dataset = self._load_event_dataset(event_id, statistics, index_column=index_column).reset_index()
                    # Separate training data
                    x_train = dataset[dataset['train_test'] == 'train'].drop('train_test', axis=1)
                    data.append(x_train)

                # Yield the concatenated training data along with asset ID and event IDs
                yield pd.concat(data), asset_id, event_ids

    def update_c2c_config(self, config: Config, wind_farm: str) -> None:
        """Update config based on provided feature descriptions.
        Updates the feature to exclude and angle lists of the data preprocessor steps.

        Args:
            config (Config): Configuration object.
            wind_farm (str): name of wind farm (A, B or C)
        """

        def get_columns(feature_description_selection: pd.DataFrame) -> List[str]:
            columns = []
            for _, row in feature_description_selection.iterrows():
                if row.statistics_type == 'average':
                    # in this case the column can be either sensor_i or sensor_i_avg, so we add both
                    columns.append(row.sensor_name)
                for stat in self._map_statistic_names(stat_names=row['statistics_type'].split(',')):
                    columns.append(f'{row.sensor_name}_{stat}')
            return columns

        feature_descriptions = self.feature_descriptions[wind_farm]
        angles = feature_descriptions.loc[feature_descriptions['is_angle']]
        to_exclude = feature_descriptions.loc[feature_descriptions['is_counter']]

        angle_columns = get_columns(angles)
        to_exclude_columns = get_columns(to_exclude)

        config['train']['data_preprocessor']['params']['angles'] = (
                config['train']['data_preprocessor']['params'].get('angles', []) + angle_columns
        )
        config['train']['data_preprocessor']['params']['features_to_exclude'] = (
                config['train']['data_preprocessor']['params'].get('features_to_exclude', []) + to_exclude_columns
        )

        config.update_config(config.config_dict)

    def _load_event_info(self) -> pd.DataFrame:
        """Load event information from CSV files and add asset ID and wind farm name as columns.

        Returns:
            pd.DataFrame: A DataFrame containing all event information with additional columns.
        """
        event_info_all = []

        # Iterate over each wind farm to load their event info
        for wind_farm in self.wind_farms:
            # Load event information from CSV
            event_info = pd.read_csv(self.wind_farms[wind_farm] / 'event_info.csv', sep=';')
            event_info["event_start"] = pd.to_datetime(event_info["event_start"])
            event_info["event_end"] = pd.to_datetime(event_info["event_end"])
            # Add wind farm name to the DataFrame
            event_info['wind_farm'] = wind_farm

            # rename asset column, so it has the same column name as in the datasets
            event_info.rename(columns={'asset': 'asset_id'}, inplace=True)

            # Collect event information for all wind farms
            event_info_all.append(event_info)
        # Concatenate all event info DataFrames into a single DataFrame
        return pd.concat(event_info_all)

    def _load_feature_descriptions(self) -> Dict[str, pd.DataFrame]:
        """Load feature descriptions from CSV files."""

        feature_descriptions = {}
        for wind_farm in self.wind_farms:
            feature_description = pd.read_csv(self.wind_farms[wind_farm] / 'feature_description.csv', sep=';')
            feature_descriptions[wind_farm] = feature_description
        return feature_descriptions

    def _column_selection(self, wind_farm: str, dataset_path: Path, selected_statistics: List[str] = None) -> List[str]:
        """Create list of columns to read.

        Args:
            wind_farm (str): Name of the wind farm for which columns are selected.
            dataset_path (Path): Path object describing the path to the dataset.
            selected_statistics (List[str], optional): describes which statistic features will be selected.
                Possible statistics are 'avg', 'min', 'max' and 'std'. If None are provided it defaults to ['avg'].

        Returns:
            List of columns names
        """

        if selected_statistics is None:
            selected_statistics = ['average']

        selected_statistics = self._map_statistic_names(stat_names=selected_statistics)

        # read 1 row to get existing columns
        dataset_columns = pd.read_csv(dataset_path, sep=';', nrows=1).columns

        # Base columns to always include
        base_columns = ['id', 'train_test', 'time_stamp', 'asset_id', 'status_type_id']

        # Generate selected columns based on feature descriptions and statistics selection
        selected_columns = base_columns.copy()
        for _, row in self.feature_descriptions[wind_farm].iterrows():
            sensor_name = row['sensor_name']
            for stat in selected_statistics:
                if stat in self._map_statistic_names(stat_names=row['statistics_type'].split(',')):
                    # Include the sensor column if only average is selected
                    if stat == 'avg' and sensor_name in dataset_columns:
                        selected_columns.append(sensor_name)

                    # Include the statistic-specific column if it exists
                    stat_col_name = f"{sensor_name}_{stat}"
                    if stat_col_name in dataset_columns:
                        selected_columns.append(stat_col_name)

        return selected_columns

    @staticmethod
    def _map_statistic_names(stat_names: Union[List[str], str]) -> List[str]:
        """ Maps given statistic_names to a unified stat name, for internal comparisons.

        Args:
            stat_names (Union[List[str], str]): List of statistic names or a single statistic name.

        Returns:
            List[str]: List of unified stat_names
        """
        valid_statistic_mapping = {
            'average': 'avg', 'avg': 'avg',
            'minimum': 'min', 'min': 'min',
            'maximum': 'max', 'max': 'max',
            'std_dev': 'std', 'std': 'std', 'standard_deviation': 'std'
        }
        if isinstance(stat_names, str):
            stat_names = [stat_names]
        try:
            unified_stat_names = [valid_statistic_mapping[stat_name.lower()] for stat_name in stat_names]
        except KeyError:
            raise ValueError('Selected statistics not valid, selected are %s, must be one of %s.',
                             stat_names, list(valid_statistic_mapping.keys()))
        return unified_stat_names

    def _load_event_dataset(self, event_id: int, statistics: List[str] = None, index_column: str = 'id'
                            ) -> pd.DataFrame:
        """Returns the dataset for the provided event_id

        Args:
            event_id (int): ID of the event dataset
            statistics (List[str], optional): describes which statistic features will be selected. Possible statistics
                are 'avg', 'min', 'max' and 'std'. If None are provided it defaults to ['avg'].
            index_column (str, optional): Name of the index column. Default is 'id'
        """
        if index_column not in ['id', 'time_stamp']:
            raise ValueError('Index column must be one of [`id`, `time_stamp`].')

        wf = self.event_info_all.loc[self.event_info_all['event_id'] == event_id, 'wind_farm'].iloc[0]
        path = self.wind_farms[wf] / 'datasets' / f'{event_id}.csv'

        selected_columns = self._column_selection(wind_farm=wf, selected_statistics=statistics, dataset_path=path)
        dataset = pd.read_csv(path, sep=';', parse_dates=['time_stamp'], index_col=index_column,
                              usecols=selected_columns, date_format="%Y-%m-%d %H:%M:%S")
        numeric_columns = [col for col in dataset.columns
                           if col not in ['time_stamp', 'asset_id', 'id', 'status_type_id', 'train_test']]
        dataset[numeric_columns] = dataset[numeric_columns].astype(float)
        return dataset
