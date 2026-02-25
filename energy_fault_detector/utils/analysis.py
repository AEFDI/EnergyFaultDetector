"""Analysis utility functions"""

from typing import Tuple, List

import numpy as np
import pandas as pd


def create_events(sensor_data: pd.DataFrame, boolean_information: pd.Series,
                  min_event_length: int = 10) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Create an event DataFrame based on boolean information such as predicted anomalies or a normal index
    and return a list of event DataFrames intended for further evaluation.

    Args:
        sensor_data (pd.DataFrame): A DataFrame with a timestamp as index and numerical sensor data.
        boolean_information (pd.Series): A Series with a timestamp as index and boolean values indicating events.
        min_event_length (int, optional): The smallest number of consecutive True timestamps needed to define an event.
            Defaults to 10.

    Returns:
        Tuple[pd.DataFrame, List[pd.DataFrame]]: A tuple containing:
            - event_meta_data (pd.DataFrame): A DataFrame with columns 'start', 'end', and 'duration' for each event.
            - event_data (List[pd.DataFrame]): A list of DataFrames corresponding to the sensor data during the defined events.
    """
    # Create a boolean mask for consecutive True values
    mask = (boolean_information != boolean_information.shift()).cumsum()

    # Group by the mask and filter groups where bool_series is True and has more
    # than consecutive_true_value_threshold consecutive True
    bool_mask = boolean_information.groupby(mask).transform(lambda data: data.all()).fillna(False)
    grouped_sensor_data = sensor_data[bool_mask].groupby(mask)
    event_data = [group[1] for group in grouped_sensor_data if len(group[1]) >= min_event_length]

    event_meta_data = pd.DataFrame()
    event_meta_data['start'] = [event.index[0] for event in event_data]
    event_meta_data['end'] = [event.index[-1] for event in event_data]
    try:
        event_meta_data['start'] = event_meta_data['start'].dt.round('min')
        event_meta_data['end'] = event_meta_data['end'].dt.round('min')
    except AttributeError:
        # if index is not datetimelike an attribute error is thrown. In this case do nothing
        pass
    event_meta_data['duration'] = event_meta_data['end'] - event_meta_data['start']
    return event_meta_data, event_data


def calculate_criticality(anomalies: pd.Series, normal_idx: pd.Series = None, init_criticality: int = 0,
                          max_criticality: int = 1000) -> pd.Series:
    """Calculate criticality based on anomaly detection results. Increases if an anomaly is detected during normal
    operation, eases if no anomalies are detected during normal operation. If normal_idx is not provided, it is assumed
    that all detected anomalies occur during normal operation.

    Args:
        anomalies (pd.Series): A pandas Series with boolean values indicating whether an anomaly was detected,
            indexed by timestamp.
        normal_idx (pd.Series, optional): A pandas Series with boolean values indicating normal operation, indexed by
            timestamp.
        init_criticality (int, optional): The initial criticality value. Defaults to 0.
        max_criticality (int, optional): The maximum criticality value. Defaults to 1000.

    Returns:
        pd.Series: A pandas Series representing the criticality over time, indexed by timestamp.

    Raises:
        ValueError: If the lengths of the given pandas Series for anomalies and normal_idx do not match.
    """

    # Ensure aligned and sorted indices
    anomalies = anomalies.sort_index()

    if normal_idx is None:
        # Assume everything is normal if not provided
        normal_idx = pd.Series(np.full(len(anomalies), True), index=anomalies.index)
    else:
        # Align normal_idx to anomalies index
        normal_idx = normal_idx.sort_index()
        normal_idx = normal_idx.reindex(anomalies.index)
        # Treat missing values as normal operation (True)
        normal_idx = normal_idx.fillna(True).astype(bool)

    if len(anomalies) != len(normal_idx):
        raise ValueError('length of given pandas series anomalies and normal idx do not match!')

    # Compute step deltas efficiently with numpy
    normal_arr = normal_idx.to_numpy(dtype=bool, copy=False)
    anomaly_arr = anomalies.to_numpy(dtype=bool, copy=False)

    # +1 when normal & anomaly, -1 when normal & not anomaly, else 0
    deltas = np.where(normal_arr & anomaly_arr, 1, np.where(normal_arr & ~anomaly_arr, -1, 0)).astype(int)

    # Apply bounded cumulative sum (reflecting at 0 and max_criticality)
    crit = np.empty_like(deltas, dtype=np.int64)
    c = int(init_criticality)
    max_c = int(max_criticality)
    for i, d in enumerate(deltas):
        c = c + int(d)
        if c < 0:
            c = 0
        elif c > max_c:
            c = max_c
        crit[i] = c

    return pd.Series(crit, index=anomalies.index)
