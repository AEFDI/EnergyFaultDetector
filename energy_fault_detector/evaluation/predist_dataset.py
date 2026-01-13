import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
import logging

from ..utils.data_downloads import download_zenodo_data

logger = logging.getLogger('energy_fault_detector')


class PreDistDataset:
    """Loader and preprocessor for the PreDist dataset.

    The data can be downloaded either manually from https://doi.org/10.5281/zenodo.17522254 (in this case specify
    `path`) or it can be downloaded automatically by setting download_dataset to True.

    Args:
        path (Union[str, Path]): Path to the dataset root.
        download_dataset (bool): If True, downloads the PreDist dataset from Zenodo.

    Attributes:
        events (Dict[int, pd.DataFrame): preloaded events dataframe for each manufacturer.
    """

    FAULT_HOURS_AFTER = 24
    FAULT_HOURS_BEFORE = 48

    def __init__(self, path: Union[str, Path], download_dataset: bool = False):
        if download_dataset:
            logger.info("Downloading PreDist dataset from Zenodo (10.5281/zenodo.17522254)...")
            path = download_zenodo_data(identifier="10.5281/zenodo.17522254", dest=path, overwrite=False)

        self.root_path = Path(path)

        # preload events
        self.events: Dict[int, pd.DataFrame] = {
            1: self._load_events(manufacturer=1),
            2: self._load_events(manufacturer=2)
        }

    def _load_events(self, manufacturer: int, filter_efd: bool = True) -> pd.DataFrame:
        """Loads and combines all events from faults.csv and normal_events.csv.

        Args:
            manufacturer (int): Dataset 1 or 2.
            filter_efd (bool): Whether to filter events with efd possible or not. Default: True.

        Returns:
            Events as dataframe, with start and end based on the possible anomaly start and report date for faults and
            based on event start and end for normal events.
        """

        m_path = self.root_path / f"Manufacturer {manufacturer}"

        faults = pd.read_csv(m_path / 'faults.csv', sep=';', parse_dates=[
            'Possible anomaly start', 'Report date', 'Possible anomaly end',
            'Training start', 'Training end'
        ], index_col='Event ID')

        normals = pd.read_csv(m_path / 'normal_events.csv', sep=';', parse_dates=[
            'Event start', 'Event end', 'Training start', 'Training end'
        ], index_col='Event ID')

        if filter_efd:
            # Only filter faults where early fault detection is possible (from a data perspective)
            faults = faults[faults['efd_possible']]

        faults['Event type'] = 'anomaly'
        faults['Event end'] = faults['Report date']  # for easy data selection later on
        normals['Event type'] = 'normal'

        return pd.concat([faults, normals])

    def load_substation_data(self, manufacturer: int, substation_id: int) -> pd.DataFrame:
        """Loads raw CSV, maps string values, and cleans indices."""
        file_path = self.root_path / f"Manufacturer {manufacturer}" / 'operational_data' / f"substation_{substation_id}.csv"
        df = pd.read_csv(file_path, sep=';', index_col='timestamp', parse_dates=['timestamp'], low_memory=False)
        df.index = df.index.tz_localize(None)
        df = df.sort_index()

        # Mapping string values (EIN/AUS) to (1/0)
        val_map = {'EIN': 1, 'AUS': 0}
        status_cols = [c for c in df.columns
                       if any(x in c for x in ['s_hc1_heating_pump_status_setpoint',
                                               's_hc1.2_heating_pump_status',
                                               's_hc1.3_heating_pump_status',
                                               's_hc2_dhw_3-way_valve_status',
                                               's_dhw_3-way_valve_status',
                                               's_hc1.1_heating_pump_status'])]
        for col in status_cols:
            if col in df.columns:
                df[col] = df[col].map(val_map).astype('Int32')

        # Map control unit mode to integer
        mode_map = {'Nacht': -1, 'Standby': 0, 'Tag': 1}
        for col in [c for c in df.columns if 'control_unit_mode' in c]:
            df[col] = df[col].map(mode_map).astype('Int32')

        # Handle noisy outside temperature value for specific substations
        # In these cases, the outside temperature is not known - the sensor value is just noise
        if manufacturer == 2 and substation_id in [18, 61]:
            df = df.drop(columns=['outdoor_temperature'], errors='ignore')

        return df[~df.index.duplicated(keep='first')]

    def create_normal_flag(self, data: pd.DataFrame, manufacturer: int, substation_id: int) -> pd.Series:
        """Create a normal flag based on disturbances, so we can select normal behaviour for training models.

        Args:
            data (pd.DataFrame): Dataframe containing sensor data for a specific substation.
            manufacturer (int): Dataset 1 or 2.
            substation_id (int): ID of the substation to load data from.

        Returns:
            pd.Series: Normal flag (boolean) based on disturbances with the same timestamp index as data.
        """

        dist_path = self.root_path / f"Manufacturer {manufacturer}" / 'disturbances.csv'
        disturbances = pd.read_csv(dist_path, sep=';', parse_dates=['Event start'])
        disturbances = disturbances[disturbances['substation ID'] == substation_id]

        normal_flag = pd.Series(True, index=data.index)

        # 1. Mark known anomalies from events_df
        events_df = self.events[manufacturer]
        anoms = events_df[(events_df['substation ID'] == substation_id) & (events_df['Event type'] == 'anomaly')]
        for _, row in anoms.iterrows():
            # If we do not know when an anomaly started, we mark FAULT_HOURS_BEFORE before report
            start = (row['Possible anomaly start']
                     if pd.notna(row['Possible anomaly start'])
                     else (row['Report date'] - pd.Timedelta(hours=self.FAULT_HOURS_BEFORE)))
            # If anomaly end was not provided, add some time after the fault for maintenance
            # (This does not happen, anomaly end is always provided in this dataset)
            end = (row['Possible anomaly end']
                   if pd.notna(row['Possible anomaly end'])
                   else (row['Report date'] + pd.Timedelta(hours=self.FAULT_HOURS_AFTER)))
            normal_flag.loc[start:end] = False

        # remove faults from disturbances already marked by the events dataframe
        faults_in_disturbances = disturbances[disturbances['type'] == 'fault']
        faults_in_event_data = faults_in_disturbances[faults_in_disturbances['Event start'].isin(anoms['Report date'])]
        disturbances = disturbances[~disturbances.index.isin(faults_in_event_data.index)]

        # 2. Mark disturbances (tasks, activities and remaining faults)
        for _, dist in disturbances.iterrows():
            # round to nearest 10 minutes to match timestamp index of the data
            d_start = dist['Event start'].floor('10min')
            if dist['type'] == 'fault':
                normal_flag.loc[d_start - pd.Timedelta(hours=self.FAULT_HOURS_BEFORE):
                                d_start + pd.Timedelta(hours=self.FAULT_HOURS_AFTER)] = False
            else:  # task/activity: mark the full day as possibly anomalous behaviour
                normal_flag.loc[d_start: d_start.normalize() + pd.Timedelta(days=1)] = False

        return normal_flag

    def get_event_data(self, manufacturer: int, event_id: int, max_training_days: int = 2*365) -> Dict[str, Any]:
        """Extracts training and test slices for a specific event row (fault or normal).
        """

        # get info from event
        event_row = self.events[manufacturer].loc[event_id]
        substation_id = event_row['substation ID']
        train_start = event_row['Training start']
        train_end = event_row['Training end']
        event_end = event_row['Event end']
        event_type = event_row['Event type'].lower()
        anomaly_end = event_row.loc['Possible anomaly end']

        # Max 2 years of training data
        train_start = max(train_start, train_end - pd.Timedelta(days=max_training_days))

        data = self.load_substation_data(manufacturer, event_row['substation ID'])

        # Training data
        train_data = data.loc[train_start:train_end]

        # Test data
        if event_type == 'normal':
            test_data = data.loc[train_end:event_end]
        else:  # anomaly
            # By default, 7 days before report, add 2 days after report for visualisations
            test_data = data.loc[event_end - pd.Timedelta(days=7):anomaly_end + pd.Timedelta(days=2)]
            # Exception: event 67 of manufacturer 1 (3 months)
            if event_id == 67 and manufacturer == 1:
                test_data = data.loc[
                    event_row['Possible anomaly start']:anomaly_end + pd.Timedelta(days=2)
                ]

        # Drop columns that are missing in the evaluation period
        eval_data = test_data.loc[:event_end]
        eval_data = eval_data.dropna(how='all', axis=1)
        train_data = train_data[eval_data.columns]
        test_data = test_data[eval_data.columns]

        # Create normal behaviour indicator
        train_normal_flag = self.create_normal_flag(data=train_data,
                                                    manufacturer=manufacturer,
                                                    substation_id=substation_id)
        test_normal_flag = self.create_normal_flag(data=test_data,
                                                   manufacturer=manufacturer,
                                                   substation_id=substation_id)
        return {
            'train_data': train_data,
            'test_data': test_data,
            'train_normal_flag': train_normal_flag,
            'test_normal_flag': test_normal_flag,
            'event_data': event_row,
        }
