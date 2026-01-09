
import pathlib
import logging
from typing import Union, List

import pandas as pd
from sklearn.decomposition import PCA

from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.config import Config
from energy_fault_detector.quick_fault_detection.optimization import automatic_hyper_opt

logger = logging.getLogger('energy_fault_detector')
here = pathlib.Path(__file__).parent.resolve()


def select_config(train_data: pd.DataFrame, normal_index: Union[pd.Series, None],
                  status_label_confidence_percentage: float, features_to_exclude: Union[List[str], None],
                  angles: Union[List[str], None], automatic_optimization: bool) -> Config:
    """ Selects a suitable config for the given data. The config is determined based on the data dimension and a
    PCA. If automatic optimization is True an optuna hyperparameter optimization is done for the autoencoder.

    Args:
        train_data (pd.DataFrame): Dataframe containing numerical values
        normal_index (Union[pd.Series, None]): Contains boolean information about which rows of train_data are normal
            and which contain anomalous behavior.
        status_label_confidence_percentage (Optional float): Determines the quantile for quantile threshold method.
        features_to_exclude (Union[List[str], None]): List of column names of train_data which should be ignored.
        angles (Union[List[str], None]): List of column names of angle features in train_data which need
            specialized preprocessing.
        automatic_optimization (bool): If True an optuna hyperparameter optimization is done for the autoencoder.

    Returns:
        Config: (optimized) config object for the AnomalyDetector.
    """
    config_path = here / '..' / 'base_config.yaml'
    config = Config(config_filename=str(config_path))
    config = update_preprocessor_config(config=config, features_to_exclude=features_to_exclude, angles=angles)
    config = update_threshold_config(config=config, quantile=status_label_confidence_percentage)
    pca_code_size = int(PCA(n_components=0.99).fit(train_data.values).n_components_)
    if automatic_optimization:
        logger.info('Optimizing Hyperparameters... (this can take some time)')
        autoencoder_params = automatic_hyper_opt(config=config, train_data=train_data,
                                                 normal_index=normal_index, pca_code_size=pca_code_size,
                                                 num_trials=10)
    else:
        # Rework if there are implemented autoencoder models which do not have the parameter 'code_size'.
        prepped_train_data, _, _ = FaultDetector(config).preprocess_train_data(sensor_data=train_data,
                                                                               normal_index=normal_index)
        autoencoder_params = {'code_size': pca_code_size, 'layers': [max(prepped_train_data.shape[1], 20),
                                                                     max(int(prepped_train_data.shape[1] / 2), 10),
                                                                     max(int(prepped_train_data.shape[1] / 4), 5),
                                                                     ]}
    config = update_autoencoder_config(config=config, autoencoder_params=autoencoder_params)
    return config


def update_preprocessor_config(config: Config, features_to_exclude: Union[List[str], None],
                               angles: Union[List[str], None]) -> Config:
    """ Updates data preprocessor parameters in the config with user given metadata for features.

    Args:
        config (Config): Config object for an AnomalyDetector.
        features_to_exclude (Union[List[str], None]): List of column names which should be ignored.
        angles (Union[List[str], None]): List of column names of angle features which need
            specialized preprocessing.

    Returns:
        Config: Updated config object.
    """

    if features_to_exclude is not None:
        if config['train']['data_preprocessor'].get('params'):
            # old data preprocessing configuration style
            config['train']['data_preprocessor']['params']['features_to_exclude'] = features_to_exclude
        else:
            # new configuration style
            steps = config['train']['data_preprocessor'].setdefault('steps', [])
            column_selector_found = False
            for step in steps:
                if step['name'] == 'column_selector':
                    step['params']['features_to_exclude'] = features_to_exclude
                    column_selector_found = True
                    break
            if not column_selector_found:
                steps.append({'name': 'column_selector', 'params': {'features_to_exclude': features_to_exclude}})
    if angles is not None:
        if config['train']['data_preprocessor'].get('params'):
            # old data preprocessing configuration style
            config['train']['data_preprocessor']['params']['angles'] = angles
        else:
            # new configuration style
            steps = config['train']['data_preprocessor'].setdefault('steps', [])
            angle_transformer_found = False
            for step in steps:
                if step['name'] == 'angle_transformer':
                    step['params']['angles'] = angles
                    angle_transformer_found = True
                    break
            if not angle_transformer_found:
                steps.append({'name': 'angle_transformer', 'params': {'angles': angles}})
    return config


def update_autoencoder_config(config: Config, autoencoder_params: dict) -> Config:
    """ Updates autoencoder parameters in the config with the new (optimized) parameters.

    Args:
        config (Config): Config object for an AnomalyDetector.
        autoencoder_params (dict): Dictionary defining new parameter values for the autoencoder

    Returns:
        Config: Updated config object.
    """
    config['train']['autoencoder']['params'].update(autoencoder_params)
    return config


def update_threshold_config(config: Config, quantile: float) -> Config:
    """ Updates threshold parameters in the config.

    Args:
        config (Config): Config object for an AnomalyDetector.
        quantile: Quantile for the quantile threshold method (must be a float between 0 and 1)

    Returns:
        Config: Updated config object.
    """
    if quantile > 1:
        logger.warning(f'Quantile for reconstruction error threshold was specified as {quantile} which larger than 1.'
                       f'To avoid an exception quantile will be set to the default value 0.95')
        quantile = 0.95
    if quantile < 0:
        logger.warning(f'Quantile for reconstruction error threshold was specified as {quantile} which smaller than 0.'
                       f'To avoid an exception quantile will be set to the default value 0.95')
        quantile = 0.95
    config['train']['threshold_selector']['params']['quantile'] = quantile
    return config
