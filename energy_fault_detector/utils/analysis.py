"""Analysis utility functions"""

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd


def create_events(sensor_data: pd.DataFrame, boolean_information: pd.Series,
                  min_event_length: int = 10) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Create an event DataFrame based on boolean information such as predicted anomalies or a normal index
    and return a list of event DataFrames intended for further evaluation.

    Works with both a simple DatetimeIndex and a MultiIndex as index of ``sensor_data``.
    For a MultiIndex, we assume there is exactly one datetime-like level and at most one
    non-datetime grouping level (e.g. (asset_id, timestamp)) and create events per group.

    Args:
        sensor_data (pd.DataFrame): A DataFrame with a timestamp (or MultiIndex including a timestamp) as index
            and numerical sensor data.
        boolean_information (pd.Series): A Series with the same index (or broadcastable) and boolean values
            indicating events.
        min_event_length (int, optional): The smallest number of consecutive True timestamps needed to define an event.
            Defaults to 10.

    Returns:
        Tuple[pd.DataFrame, List[pd.DataFrame]]: A tuple containing:
            - event_meta_data (pd.DataFrame): Columns 'start', 'end', 'duration' and, for MultiIndex,
              an additional 'group' column with the grouping key.
            - event_data (List[pd.DataFrame]): A list of DataFrames corresponding to the sensor data during the defined events.
    """
    # Align boolean_information to sensor_data index
    boolean_information = boolean_information.astype(bool)
    boolean_information = boolean_information.reindex(sensor_data.index).fillna(False)

    idx = sensor_data.index

    # Helper to create events on a single-index view (DatetimeIndex)
    def _create_events_single(df_single: pd.DataFrame,
                              bool_single: pd.Series,
                              group_label=None) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        # contiguous segments of True
        mask = (bool_single != bool_single.shift()).cumsum()
        bool_mask = bool_single.groupby(mask).transform(lambda s: s.all()).fillna(False)

        grouped_sensor = df_single[bool_mask].groupby(mask)
        event_data_local = [g for _, g in grouped_sensor if len(g) >= min_event_length]

        if not event_data_local:
            return pd.DataFrame(columns=["start", "end", "duration"]), []

        meta = pd.DataFrame()
        meta["start"] = [ev.index[0] for ev in event_data_local]
        meta["end"] = [ev.index[-1] for ev in event_data_local]
        try:
            meta["start"] = meta["start"].dt.round("min")
            meta["end"] = meta["end"].dt.round("min")
        except AttributeError:
            # non-datetime index, ignore rounding
            pass
        meta["duration"] = meta["end"] - meta["start"]
        if group_label is not None:
            meta["group"] = group_label
        return meta, event_data_local

    # Case 1: simple DatetimeIndex (or any single index)
    if not isinstance(idx, pd.MultiIndex):
        return _create_events_single(sensor_data, boolean_information)

    # Case 2: MultiIndex â€“ detect datetime level and grouping level
    levels = idx.levels
    datetime_level_idx = None
    non_datetime_levels = []
    for i, lvl in enumerate(levels):
        if isinstance(lvl, pd.DatetimeIndex):
            datetime_level_idx = i
        else:
            non_datetime_levels.append(i)

    if datetime_level_idx is None:
        # No datetime level â€“ fallback to treating the full MultiIndex as a single index
        return _create_events_single(sensor_data, boolean_information)

    # Choose grouping level: first non-datetime level if present, otherwise no grouping
    group_level = non_datetime_levels[0] if non_datetime_levels else None

    event_meta_all: List[pd.DataFrame] = []
    event_data_all: List[pd.DataFrame] = []

    if group_level is None:
        # Only datetime levels, treat as one series
        meta, ev_data = _create_events_single(sensor_data, boolean_information)
        return meta, ev_data

    # Group by the selected non-datetime level (e.g. asset_id)
    for group_key, df_group in sensor_data.groupby(level=group_level):
        bool_group = boolean_information.loc[df_group.index]
        # Work on a view with only the datetime level as index
        dt_index = df_group.index.get_level_values(datetime_level_idx)
        df_group_single = df_group.copy()
        df_group_single.index = dt_index

        meta_g, ev_data_g = _create_events_single(df_group_single, bool_group.reset_index(drop=True).set_axis(dt_index),
                                                  group_label=group_key)
        if not meta_g.empty:
            event_meta_all.append(meta_g)
            # Restore the MultiIndex on event DataFrames
            for ev in ev_data_g:
                new_idx = pd.MultiIndex.from_arrays(
                    [np.full(len(ev), group_key), ev.index],
                    names=df_group.index.names,
                )
                ev_mi = ev.copy()
                ev_mi.index = new_idx
                event_data_all.append(ev_mi)

    if not event_meta_all:
        return pd.DataFrame(columns=["start", "end", "duration", "group"]), []

    event_meta_data = pd.concat(event_meta_all, ignore_index=True)
    return event_meta_data, event_data_all


def calculate_criticality(anomalies: pd.Series, normal_idx: pd.Series = None, init_criticality: int = 0,
                          max_criticality: int = 1000) -> pd.Series:
    """Calculate criticality based on anomaly detection results. Increases if an anomaly is detected during normal
    operation, eases if no anomalies are detected during normal operation. If normal_idx is not provided, it is assumed
    that all detected anomalies occur during normal operation.

    Args:
        anomalies (pd.Series): A pandas Series with boolean values indicating whether an anomaly was detected,
            indexed by timestamp, or MultiIndex (device_id, timestamp).
        normal_idx (pd.Series, optional): A pandas Series with boolean values indicating normal operation, same index
            as the anomalies pd.Series.
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
        raise ValueError('length mismatch between anomalies and normal_idx.')

    # Detect grouping level for MultiIndex
    group_level = _detect_group_level(anomalies.index)

    if group_level is not None:
        # Calculate per group, then concatenate
        results = []
        for _, group_anomalies in anomalies.groupby(level=group_level):
            group_normal = normal_idx.loc[group_anomalies.index]
            results.append(
                _compute_criticality_array(group_anomalies, group_normal,
                                           init_criticality, max_criticality)
            )
        return pd.concat(results)

    return _compute_criticality_array(anomalies, normal_idx, init_criticality, max_criticality)


def _detect_group_level(index: pd.Index) -> Optional[int]:
    """Return the non-datetime grouping level index, or None."""
    if not isinstance(index, pd.MultiIndex):
        return None
    for i, level in enumerate(index.levels):
        if not isinstance(level, pd.DatetimeIndex):
            return i
    return None


def _compute_criticality_array(anomalies: pd.Series, normal_idx: pd.Series,
                                init_criticality: int, max_criticality: int) -> pd.Series:
    """Bounded cumulative criticality for a single contiguous series."""
    normal_arr = normal_idx.to_numpy(dtype=bool)
    anomaly_arr = anomalies.to_numpy(dtype=bool)
    deltas = np.where(normal_arr & anomaly_arr, 1,
                      np.where(normal_arr & ~anomaly_arr, -1, 0))

    crit = np.empty(len(deltas), dtype=np.int64)
    c = init_criticality
    for i, d in enumerate(deltas):
        c = max(0, min(max_criticality, c + int(d)))
        crit[i] = c
    return pd.Series(crit, index=anomalies.index)
