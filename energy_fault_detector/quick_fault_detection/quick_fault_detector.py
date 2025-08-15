"""Quick energy fault detection, to try out the EnergyFaultDetector model on a specific dataset."""

import os
import logging
from typing import List, Optional, Tuple

import pandas as pd

from energy_fault_detector._logs import setup_logging
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.utils.analysis import create_events
from energy_fault_detector.root_cause_analysis.arcana_utils import calculate_mean_arcana_importances
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult

from energy_fault_detector.quick_fault_detection.data_loading import load_train_test_data
from energy_fault_detector.quick_fault_detection.configuration import select_config
from energy_fault_detector.quick_fault_detection.output import generate_output_plots, output_info

setup_logging(os.path.join(os.path.dirname(__file__), '..', 'logging.yaml'))
logger = logging.getLogger('energy_fault_detector')


def quick_fault_detector(csv_data_path: str, csv_test_data_path: Optional[str] = None,
                         train_test_column_name: Optional[str] = None, train_test_mapping: Optional[dict] = None,
                         time_column_name: Optional[str] = None, status_data_column_name: Optional[str] = None,
                         status_mapping: Optional[dict] = None,
                         status_label_confidence_percentage: Optional[float] = 0.95,
                         features_to_exclude: Optional[List[str]] = None, angle_features: Optional[List[str]] = None,
                         automatic_optimization: bool = True, enable_debug_plots: bool = False,
                         min_anomaly_length: int = 18, save_dir: Optional[str] = None
                         ) -> Tuple[FaultDetectionResult, pd.DataFrame]:
    """Analyzes provided data using an autoencoder based approach for identifying anomalies based on a learned normal
    behavior. Anomalies are then aggregated to events and further analyzed.
    Runs the entire fault detection module chain in one function call. Sections of this function call are:
    1. Data Loading and verification
    2. Config selection and optimization
    3. AnomalyDetector training
    4. AnomalyDetector prediction
    5. Event aggregation
    6. ARCANA-Analysis of detected events
    7. Visualization of output

    Args:
        csv_data_path (str): Path to a csv-file containing tabular data which must contain training data for the
            autoencoder. This data can also contain test data for evaluation, but in this case train_test_column and
            optionally train_test_mapping must be provided.
        csv_test_data_path (Optional str): Path to a csv file containing test data for evaluation. If test data is
            provided in both ways (i.e. via csv_test_data_path and in csv_data_path + train_test_column) then both test
            data sets will be fused into one. Default is None.
        train_test_column_name (Optional str): Name of the column which specifies which part of the data in
            csv_data_path is training data and which is test data. If this column does not contain boolean values or
            values which can be cast into boolean values, then train_test_mapping must be provided.
            True values are interpreted as training data. Default is None.
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
        status_mapping (Optional dict): Dictionary which defines a mapping of all non-boolean values in the
            status_data_column to booleans. Keys of the dictionary must be values from status_data_column, and they must
            have a datatype which can be cast to the datatype of train_test_column. Values of this dictionary must be
            booleans or at least castable to booleans. True values are interpreted as normal status. Default is None.
        status_label_confidence_percentage (Optional float): Specifies how sure the user is that the provided status 
            labels and derived normal_indexes are correct. This determines the quantile for quantile threshold method.
            Default is 0.95.
        features_to_exclude (Optional[List[str]]): List of column names which are present in the csv-files but which
            should be ignored for this failure detection run. Default is None.
        angle_features (Optional[List[str]]): List of column names which represent angle-features. This enables a
            specialized preprocessing of angle features, which might otherwise hinder the failure detection process.
            Default is None.
        automatic_optimization (bool): If True, an automatic hyperparameter optimization is done based on the dimension
            of the provided dataset. Default is True.
        enable_debug_plots (bool): If True advanced information for debugging is added to the result plots.
            Default is False.
        min_anomaly_length (int): Minimal number of consecutive anomalies (i.e. data points with an anomaly score above
            the FaultDetector threshold) to define an anomaly event.
        save_dir (Optional[str]): Directory to save the output plots. If not provided, the plots are not saved.
            Defaults to None.

    Returns:
        Tuple(FaultDetectionResult, pd.DataFrame): FaultDetectionResult object with the following DataFrames:

            - predicted_anomalies: DataFrame with a column 'anomaly' (bool).
            - reconstruction: DataFrame with reconstruction of the sensor data with timestamp as index.
            - deviations: DataFrame with reconstruction errors.
            - anomaly_score: DataFrame with anomaly scores for each timestamp.
            - bias_data: DataFrame with ARCANA results with timestamp as index. None if ARCANA was not run.
            - arcana_losses: DataFrame containing recorded values for all losses in ARCANA. None if ARCANA was not run.
            - tracked_bias: List of DataFrames. None if ARCANA was not run.

        and the detected anomaly events as dataframe.
    """
    logger.info('Starting Automated Energy Fault Detection and Identification.')
    logger.info('Loading Data...')
    train_data, train_normal_index, test_data = load_train_test_data(csv_data_path=csv_data_path,
                                                                     csv_test_data_path=csv_test_data_path,
                                                                     train_test_column_name=train_test_column_name,
                                                                     train_test_mapping=train_test_mapping,
                                                                     time_column_name=time_column_name,
                                                                     status_data_column_name=status_data_column_name,
                                                                     status_mapping=status_mapping)
    logger.info('Selecting suitable config...')
    config = select_config(train_data=train_data, normal_index=train_normal_index,
                           status_label_confidence_percentage=status_label_confidence_percentage,
                           features_to_exclude=features_to_exclude, angles=angle_features,
                           automatic_optimization=automatic_optimization)
    logger.info('Training a Normal behavior model.')
    anomaly_detector = FaultDetector(config=config)
    anomaly_detector.fit(sensor_data=train_data, normal_index=train_normal_index)
    logger.info('Evaluating Test data based on the learned normal behavior.')
    prediction_results = anomaly_detector.predict(sensor_data=test_data, root_cause_analysis=False)
    anomalies = prediction_results.predicted_anomalies['anomaly']
    # Find anomaly events
    event_meta_data, event_data_list = create_events(sensor_data=test_data,
                                                     boolean_information=anomalies,
                                                     min_event_length=min_anomaly_length)
    arcana_mean_importance_list = []
    arcana_loss_list = []
    for i in range(len(event_meta_data)):
        logger.info(f'Analyzing anomaly events ({i + 1} of {len(event_meta_data)}).')
        event_data = event_data_list[i]
        arcana_mean_importances, arcana_losses = analyze_event(
            anomaly_detector=anomaly_detector,
            event_data=event_data,
            track_losses=enable_debug_plots)
        arcana_mean_importance_list.append(arcana_mean_importances)
        if len(arcana_losses) > 0:
            arcana_loss_list.append(arcana_losses)
    logger.info('Generating Output Graphics.')
    logger.info(output_info)
    generate_output_plots(anomaly_detector=anomaly_detector, train_data=train_data, normal_index=train_normal_index,
                          test_data=test_data, arcana_losses=arcana_loss_list,
                          arcana_mean_importances=arcana_mean_importance_list,
                          event_meta_data=event_meta_data, save_dir=save_dir)
    return prediction_results, event_meta_data


def analyze_event(anomaly_detector: FaultDetector, event_data: pd.DataFrame, track_losses: bool
                  ) -> Tuple[pd.Series, pd.DataFrame]:
    """ Runs root cause analysis for detected anomaly events.

    Args:
        anomaly_detector (FaultDetector): trained AnomalyDetector instance.
        event_data (pd.DataFrame): data from a detected anomaly event
        track_losses (bool): If True ARCANA-losses are tracked. 

    Returns:
        importances (pd.Series): Series of importance values for every feature in event_data.
        tracked_losses (pd.DataFrame): Potentially empty DataFrame containing recorded ARCANA losses.
    """
    bias, tracked_losses, _ = anomaly_detector.run_root_cause_analysis(sensor_data=event_data,
                                                                       track_losses=track_losses,
                                                                       track_bias=False)
    importances_mean = calculate_mean_arcana_importances(bias_data=bias)
    return importances_mean, tracked_losses
