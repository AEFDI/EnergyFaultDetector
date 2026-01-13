from typing import List, Tuple
from pathlib import Path
from copy import deepcopy
import gc

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score

from energy_fault_detector import FaultDetector, Config
from energy_fault_detector.core import FaultDetectionResult
from energy_fault_detector.evaluation import PreDistDataset
from energy_fault_detector.root_cause_analysis.arcana_utils import calculate_mean_arcana_importances


def train_or_get_model(event_id: int, dataset: PreDistDataset, manufacturer: int, model_name: str, conf: Config,
                       bottleneck_ratio: float, load_from_file: bool, time_features: List[str] | None
                       ) -> Tuple[int, FaultDetectionResult]:
    """Processes a single event: loads data, trains/loads model, and predicts.

    Args:
        event_id (int): ID of the event to process.
        dataset (PreDistDataset): Dataset containing the event data.
        manufacturer (int): Manufacturer ID for the event.
        model_name (str): Name of the model to use for training.
        conf (Config): Base configuration for training.
        bottleneck_ratio (float): Ratio to determine the bottleneck size for the autoencoder.
        load_from_file (bool): Whether to load the model from file if available. Otherwise, train and save a new model.
        time_features (List[str] | None): List of time features to use for conditional autoencoders.

    Returns:
        Tuple[int, FaultDetectionResult]: A tuple containing the event ID and the fault detection result.
    """

    # Local copy of time features and configuration to avoid mutation issues in parallel
    ts_features = time_features.copy() if time_features else None
    local_conf = deepcopy(conf)

    # Get specific event data
    data = dataset.get_event_data(manufacturer, event_id)
    train_data = data['train_data']
    test_data = data['test_data']

    # Create a new model or load from file
    model_path = Path(f'./models/m{manufacturer}/event_{event_id}/{model_name}')

    if model_path.exists() and load_from_file:
        model = FaultDetector()
        model.load_models(model_path)
        if (model_path / 'ts_features.txt').exists():
            with open(model_path / 'ts_features.txt', 'r') as f:
                ts_features = f.read().splitlines()
    else:
        # Add the code size to the AE configuration, based on the bottleneck ratio
        # code_size is part of the model configuration, so we overwrite the parameter of the underlying dictionary.
        bottleneck = calculate_bottleneck(train_data, local_conf, bottleneck_ratio)
        local_conf['train']['autoencoder']['params']['code_size'] = bottleneck

        # For the conditional autoencoders, add time features
        if ts_features:
            train_data = add_cyclic_time_features(train_data, ts_features)

        model = FaultDetector(local_conf, model_directory=model_path)
        model_data = model.fit(train_data, data['train_normal_flag'], save_models=True, overwrite_models=True)
        if ts_features:
            # For the conditional autoencoders, save the time features as well
            with open(Path(model_data.model_path) / 'ts_features.txt', 'w') as f:
                f.write('\n'.join(ts_features))

    # Predict
    if ts_features:
        # For the conditional autoencoders, add time features
        test_data = add_cyclic_time_features(test_data, ts_features)
    predictions = model.predict(test_data)

    # memory cleanup
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return event_id, predictions


def add_cyclic_time_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Calculates cyclical time features from the timestamp index."""
    df = df.copy()
    if 'hour_of_day' in features:
        phase = df.index.hour / 24
        df['hour_of_day_sine'] = np.sin(2 * np.pi * phase)
        df['hour_of_day_cosine'] = np.cos(2 * np.pi * phase)
    if 'day_of_week' in features:
        phase = df.index.day_of_week / 7
        df['day_of_week_sine'] = np.sin(2 * np.pi * phase)
        df['day_of_week_cosine'] = np.cos(2 * np.pi * phase)
    if 'day_of_year' in features:
        phase = df.index.dayofyear / (365 + df.index.is_leap_year)
        df['day_of_year_sine'] = np.sin(2 * np.pi * phase)
        df['day_of_year_cosine'] = np.cos(2 * np.pi * phase)
    return df


def calculate_bottleneck(df: pd.DataFrame, config: Config, ratio: float) -> int:
    """Calculates code_size (the bottleneck of the autoencoder) relative to input dimensions, accounting for excluded
    features and conditional features.

    Args:
        df (pd.DataFrame): Input dataframe to determine the number of input features of the AE.
        config (Config): Configuration for the AE.
        ratio (float, optional): Ratio between input and bottleneck dimensions.

    Returns:
        int: The calculated bottleneck size for the autoencoder.
    """

    # Get the conditional features from the config
    ae_params = config['train']['autoencoder']['params']
    cond_features = ae_params.get('conditional_features', [])

    # Exclude conditions (not compressed)
    input_dim = len(df.columns) - len([c for c in cond_features if c in df.columns])

    # Check for feature exclusions in config
    excluded = []
    dp_config = config['train'].get('data_preprocessor', {})
    if dp_config.get('params'):
        # params-based data prep config
        excluded = dp_config.get('params').get('features_to_exclude', [])
    else:
        # steps-based data prep config
        steps = config['train']['data_preprocessor'].get('steps', [])
        for step in steps:
            if step['name'] == 'column_selector':
                excluded = step['params'].get('features_to_exclude', [])
                break

    # Remove the excluded features from the input dimension
    input_dim -= len([e for e in excluded if e in df.columns])

    return max(1, round(input_dim * ratio))


def find_optimal_threshold(true_anomalies: pd.Series,
                           max_criticalities: pd.Series,
                           thresholds: np.ndarray = np.arange(1, 100),
                           k: int = 5) -> Tuple[int, float]:
    """Finds the threshold maximizing reliability (Event-wise F0.5) using CV.

    Args:
        true_anomalies (pd.Series): Series indicating whether each event is an anomaly. 1 = anomaly, 0 = normal.
        max_criticalities (pd.Series): Series containing the maximum criticality of each event.
        thresholds (np.ndarray, optional): Array of thresholds to evaluate. Defaults to np.arange(1, 100).
        k (int, optional): Number of folds for CV. Defaults to 5.

    Returns:
        Optimal criticality threshold and avg validation reliability score.
    """
    y_true = true_anomalies.values
    max_criticalities = max_criticalities.values
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    chosen_thresholds = []
    val_scores = []

    for train_idx, val_idx in skf.split(y_true, y_true):
        y_train, max_crit_train = y_true[train_idx], max_criticalities[train_idx]
        y_val, max_crit_val = y_true[val_idx], max_criticalities[val_idx]

        # Best threshold on training data
        best_t = None
        best_train_f05 = -1.0
        for t in thresholds:
            y_pred_train = (max_crit_train >= t).astype(int)
            f05_train = fbeta_score(y_train, y_pred_train, beta=0.5, zero_division=0)
            if f05_train > best_train_f05:
                best_train_f05 = f05_train
                best_t = t

        chosen_thresholds.append(best_t)

        # Evaluate on validation data
        y_pred_val = (max_crit_val >= best_t).astype(int)
        f05_val = fbeta_score(y_val, y_pred_val, beta=0.5, zero_division=0)
        val_scores.append(f05_val)

    robust_t = int(np.median(chosen_thresholds))
    mean_val_f05 = float(np.mean(val_scores))

    return robust_t, mean_val_f05


def get_arcana_importances(manufacturer: int, event_id: int, config_name: str, data: pd.DataFrame) -> pd.Series:
    """Get ARCANA importances for a given event."""

    model_path = Path(f'models/m{manufacturer}/event_{event_id}/{config_name}')
    model = FaultDetector()
    model.load_models(model_path)

    # Load the time features and add them to the data if available (for the conditional autoencoders)
    if (model_path / 'ts_features.txt').exists():
        with open(model_path / 'ts_features.txt', 'r') as f:
            ts_features = f.read().splitlines()
        data = add_cyclic_time_features(data, ts_features)

    bias, _, _ = model.run_root_cause_analysis(data, track_losses=False, track_bias=False)
    return calculate_mean_arcana_importances(bias).sort_values(ascending=False)


def calculate_earliness(criticality_threshold: int, report_ts: int | pd.Timestamp, criticality: pd.Series,
                        min_detection_time: pd.Timedelta = pd.Timedelta(hours=24)
                        ) -> Tuple[int | pd.Timestamp | None, float]:
    """Calculate the detection time and earliness score:

    E = max(0, min(1, (report_ts - detection_timestamp) / min_detection_time))

    The earliness score is 1 if the fault is detected at least min_detection_time before the report and 0 if the
    fault is detected after the report or not detected at all. Between min_detection_time before the report and the
    report timestamp, the earliness score linearly decreases to 0.

    Args:
        criticality_threshold (int): Threshold for determining whether the event is detected.
        report_ts (int | pd.Timestamp): Timestamp of the report.
        criticality (pd.Series): Series containing the criticality pd.Series of each event.
        min_detection_time (pd.Timedelta, optional): Minimum detection time. Defaults to pd.Timedelta(hours=24).

    Returns:
        A tuple containing the detection time and earliness score. If not detected, the detection time is None and
        the earliness score is 0.
    """

    crit_threshold_reached = criticality[criticality >= criticality_threshold]
    if crit_threshold_reached.empty:
        detection_time = None
        earliness = 0
        return detection_time, earliness

    detection_timestamp = crit_threshold_reached.sort_index(ascending=True).index[0]
    detection_time = report_ts - detection_timestamp
    # max(earliness, 0) to handle detection after the fault is known
    earliness = max(min(1, detection_time / min_detection_time), 0)
    return detection_time, earliness
