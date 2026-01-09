from typing import List, Tuple
from pathlib import Path
from copy import deepcopy
import logging
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


def train_or_get_model(event_id: int, dataset: PreDistDataset, manufacturer: int, config_name: str, conf: Config,
                       bottleneck_ratio: float, load_from_file: bool, ts_features_orig: List[str] | None
                       ) -> Tuple[int, FaultDetectionResult]:
    """Processes a single event: loads data, trains/loads model, and predicts."""

    # Configure logging
    logger = logging.getLogger('energy_fault_detector')
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    logger.info(f'Processing event {event_id} for manufacturer {manufacturer}...')

    # Local copy of ts_features and configuration to avoid mutation issues in parallel
    ts_features = ts_features_orig.copy() if ts_features_orig else None
    local_conf = deepcopy(conf)

    # Get specific event data
    data = dataset.get_event_data(manufacturer, event_id)

    # Create a new model or load from file
    model_path = Path(f'./models/m{manufacturer}/event_{event_id}/{config_name}')

    if model_path.exists() and load_from_file:
        model = FaultDetector()
        model.load_models(model_path)
        if (model_path / 'ts_features.txt').exists():
            with open(model_path / 'ts_features.txt', 'r') as f:
                ts_features = f.read().splitlines()
    else:
        # Prepare data and config
        train_data = data['train_data']
        bottleneck = calculate_bottleneck(train_data, local_conf, bottleneck_ratio)
        local_conf['train']['autoencoder']['params']['code_size'] = bottleneck
        if ts_features:
            train_data = add_cyclic_time_features(train_data, ts_features)

        model = FaultDetector(local_conf, model_directory=model_path)
        model_data = model.fit(train_data, data['train_normal_flag'], save_models=True, overwrite_models=True)
        if ts_features:
            # save the time features as well
            with open(Path(model_data.model_path) / 'ts_features.txt', 'w') as f:
                f.write('\n'.join(ts_features))

    # Predict
    test_data = data['test_data']
    if ts_features:
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


def calculate_bottleneck(df: pd.DataFrame, config: Config, ratio: float = 0.75) -> int:
    """Calculates code_size relative to input dimensions, accounting for exclusions/conditions."""
    ae_params = config['train']['autoencoder']['params']
    cond_features = ae_params.get('conditional_features', [])

    # Exclude conditions and existing data_preprocessor exclusions
    input_dim = len(df.columns) - len([c for c in cond_features if c in df.columns])

    # Check for manual feature exclusions in config
    dp_params = config['train'].get('data_preprocessor', {}).get('params', {})
    excluded = dp_params.get('features_to_exclude', [])
    input_dim -= len([e for e in excluded if e in df.columns])

    return max(1, round(input_dim * ratio))


def find_optimal_threshold(true_anomalies: pd.Series,
                           max_criticalities: pd.Series,
                           thresholds: np.ndarray = np.arange(1, 100),
                           k: int = 5) -> int:
    """Finds the threshold maximizing reliability (Event-wise F0.5) using CV.

    Args:
        true_anomalies (pd.Series): Series indicating whether each event is an anomaly. 1 = anomaly, 0 = normal.
        max_criticalities (pd.Series): Series containing the maximum criticality of each event.
        thresholds (np.ndarray, optional): Array of thresholds to evaluate. Defaults to np.arange(1, 100).
        k (int, optional): Number of folds for CV. Defaults to 5.

    Returns:
        Optimal criticality threshold.
    """
    y_true = true_anomalies.values
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    results = []
    for t in thresholds:
        fold_f05 = []
        y_pred = (max_criticalities >= t).astype(int).values

        for train_idx, val_idx in skf.split(y_true, y_true):
            score = fbeta_score(y_true[val_idx], y_pred[val_idx], beta=0.5, zero_division=0)
            fold_f05.append(score)

        results.append({'threshold': t, 'mean_f05': np.mean(fold_f05)})

    best_t = max(results, key=lambda x: x['mean_f05'])['threshold']
    return best_t


def get_arcana_importances(manufacturer: int, event_id: int, config_name: str, data: pd.DataFrame) -> pd.Series:

    model_path = Path(f'models/m{manufacturer}/event_{event_id}/{config_name}')
    model = FaultDetector()
    model.load_models(model_path)
    if (model_path / 'ts_features.txt').exists():
        with open(model_path / 'ts_features.txt', 'r') as f:
            ts_features = f.read().splitlines()
        data = add_cyclic_time_features(data, ts_features)
    bias, _, _ = model.run_root_cause_analysis(data, track_losses=False, track_bias=False)
    return calculate_mean_arcana_importances(bias).sort_values(ascending=False)


def calculate_earliness(criticality_threshold: int, report_ts: int | pd.Timestamp, criticality: pd.Series,
                        min_detection_time: pd.Timedelta = pd.Timedelta(hours=24)
                        ) -> Tuple[int | pd.Timestamp | None, float]:
    """Calculate the detection time and earliness score.

    Args:
        criticality_threshold (int): Threshold for determining whether the event is detected.
        report_ts (int | pd.Timestamp): Timestamp of the report.
        criticality (pd.Series): Series containing the criticality of each event.
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
    # max(earliness, 0) to handle detection after fault is known
    earliness = max(min(1, detection_time / min_detection_time), 0)
    return detection_time, earliness
