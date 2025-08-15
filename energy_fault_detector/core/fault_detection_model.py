"""Template for anomaly detection models"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List, Tuple
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from energy_fault_detector.config import Config
from energy_fault_detector import registry
from energy_fault_detector.core import Autoencoder, AnomalyScore, ThresholdSelector
from energy_fault_detector.core.model_factory import ModelFactory
from energy_fault_detector.core.fault_detection_result import ModelMetadata, FaultDetectionResult
from energy_fault_detector.data_preprocessing import DataPreprocessor
from energy_fault_detector._logs import setup_logging
from energy_fault_detector.data_splitting.data_splitter import BlockDataSplitter

setup_logging(os.path.join(os.path.dirname(__file__), '..', 'logging.yaml'))
logger = logging.getLogger('energy_fault_detector')

DATA_PREP_DIR = 'data_preprocessor'
AUTOENCODER_DIR = 'autoencoder'
THRESHOLD_DIR = 'threshold_selector'
SCORE_DIR = 'anomaly_score'

DataType = Union[pd.DataFrame, np.ndarray, List]


class NoTrainingData(Exception):
    """Raised when no training data is available."""


class FaultDetectionModel(ABC):
    """Template for fault detection models. Train and predicts should be implemented.

    Args:
        config: a Config object with anomaly detection configuration.
        model_directory: directory with models.
        model_subdir: if given, the subdirectory of the model objects. If None,
            the current datetime (formatted yyyyMMdd_HHmmSS) is used.
            Complete path to models:

                - train method: model_directory/model_subdir/asset_id
                - tune method: model_directory/model_subdir/asset_id_tune_method
    """

    def __init__(self, config: Optional[Config] = None, model_directory: str = 'models',
                 model_subdir: Optional[Any] = None):
        self.config: Optional[Config] = config
        self.model_directory: str = model_directory

        self.anomaly_score: Optional[AnomalyScore] = None
        self.autoencoder: Optional[Autoencoder] = None
        self.threshold_selector: Optional[ThresholdSelector] = None
        self.data_preprocessor: Optional[DataPreprocessor] = None

        # add timestamps at which models were saved to a list in order to be able to identify which model was created
        # by this instance
        self.save_timestamps: List[str] = []
        self.model_subdir_name: str = str(model_subdir) if model_subdir is not None else None

        # build models
        self._model_factory: Optional[ModelFactory] = ModelFactory(config) if config else None

    def _init_models(self):
        """Initialize models."""

        if self._model_factory is None:
            raise ValueError('No models can be initialized. Did you provide a configuration?')

        logger.info('Initialize models..')
        self.anomaly_score = self._model_factory.anomaly_score
        self.autoencoder = self._model_factory.autoencoder
        self.threshold_selector = self._model_factory.threshold_selector
        self.data_preprocessor = self._model_factory.data_preprocessor

    @abstractmethod
    def fit(self, sensor_data: pd.DataFrame, normal_index: pd.Series = None, asset_id: Union[int, str] = None,
            **kwargs) -> ModelMetadata:
        """Fit models on the given sensor_data and save them locally and return the metadata.

        Args:
            asset_id: asset ID of the asset for which the model should be trained.
            sensor_data: pandas DataFrame with the sensor data to use.
                The time stamp should be the index and the sensor values as columns.
            normal_index: a pandas Series indicating normal behaviour as boolean with the timestamp as index.

        Returns:
            ModelMetadata object.
        """

    def train(self, sensor_data: pd.DataFrame, normal_index: pd.Series = None, asset_id: Union[int, str] = None,
              **kwargs) -> ModelMetadata:
        """Same as the `fit`-method."""
        return self.fit(sensor_data=sensor_data, normal_index=normal_index, asset_id=asset_id, **kwargs)

    @abstractmethod
    def predict(self, sensor_data: pd.DataFrame, model_path: Optional[str] = None, asset_id: Union[int, str] = None
                ) -> FaultDetectionResult:
        """Predict with given models for specific asset ID. Return dictionary with results.

        Args:
            asset_id: asset ID of the asset for which the model must be applied.
            model_path: path to the models to be applied.
            sensor_data: pandas DataFrame with the sensor data to use.
                The time stamp should be the index and the sensor values as columns.

        Returns:
            FaultDetectionResult object.
        """

    def train_val_split(self, x: DataType) -> Tuple[DataType, DataType]:
        """Split data in train and validation data.

        Args:
            x: dataframe/list/array to split

        Returns:
            tuple: x_train, x_val
        """

        data_splitter_params = self.config.data_split_params

        data_splitter_type = data_splitter_params.get('type')
        if (data_splitter_type is None) or (data_splitter_type in ['DataSplitter', 'BlockDataSplitter', 'blocks']):
            train_data, val_data = BlockDataSplitter(
                train_block_size=data_splitter_params.get('train_block_size'),
                val_block_size=data_splitter_params.get('val_block_size'),
            ).split(x=x)
        else:  # 'sklearn', 'train_test_split'
            shuffle = data_splitter_params.get('shuffle')
            train_data, val_data = train_test_split(
                x,
                test_size=data_splitter_params['validation_split'],
                shuffle=shuffle if shuffle is not None else False
            )

        return train_data, val_data

    def save_models(self, model_name: Union[str, int] = None, overwrite: bool = False) -> Tuple[str, str]:
        """Save the model objects.

        Args:
            model_name (optional str): model name, will be the directory in which the model files are saved.
                If not provided, the model files will be saved in self.model_directory / self.model_subdir
            overwrite (optional bool): If True existing folders can be overwritten. Default: False

        Returns:
            The full path to the saved models and the timestamp of the function call in string format.
        """
        if not os.path.isdir(self.model_directory):
            os.mkdir(self.model_directory)

        current_datetime: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_timestamps.append(current_datetime)
        if self.model_subdir_name is None:
            model_subdir = current_datetime
        else:
            model_subdir = self.model_subdir_name

        sub_dir_path = os.path.join(self.model_directory, model_subdir)
        os.makedirs(sub_dir_path, exist_ok=True)

        model_dir = sub_dir_path if model_name is None else os.path.join(sub_dir_path, str(model_name))
        os.makedirs(model_dir, exist_ok=True)

        if self.model_subdir_name is not None and not overwrite:
            # if subdir is set by the user and models are saved more than once the original save would be overwritten
            # in case of overwrite = False this must be prevented by adding the current datetime as unique identifier to
            # the path
            model_dir = os.path.join(sub_dir_path, current_datetime)
            os.makedirs(model_dir, exist_ok=True)

        self.data_preprocessor.save(os.path.join(model_dir, DATA_PREP_DIR), overwrite=overwrite)
        self.autoencoder.save(os.path.join(model_dir, AUTOENCODER_DIR), overwrite=overwrite)
        self.threshold_selector.save(os.path.join(model_dir, THRESHOLD_DIR), overwrite=overwrite)
        self.anomaly_score.save(os.path.join(model_dir, SCORE_DIR), overwrite=overwrite)
        self.config.write_config(os.path.join(model_dir, 'config.yaml'), overwrite=overwrite)
        return os.path.abspath(model_dir), current_datetime

    def load_models(self, model_path: str, load_threshold: bool = True) -> None:
        """Load saved models given the model path.

        Args:
            model_path: Path to the model files.
            load_threshold: if True, then the threshold model will be loaded
        """

        data_prep_dir = os.path.join(model_path, DATA_PREP_DIR)
        self.data_preprocessor = self._load_pickled_model(
            model_type='data_preprocessor',
            model_directory=data_prep_dir
        )
        self.autoencoder = self._load_pickled_model(
            model_type='autoencoder',
            model_directory=os.path.join(model_path, AUTOENCODER_DIR)
        )
        if load_threshold:
            self.threshold_selector = self._load_pickled_model(
                model_type='threshold_selector',
                model_directory=os.path.join(model_path, THRESHOLD_DIR)
            )
        self.anomaly_score = self._load_pickled_model(
            model_type='anomaly_score',
            model_directory=os.path.join(model_path, SCORE_DIR)
        )
        # for backwards compatibility - check whether config was saved:
        if os.path.exists(os.path.join(model_path, 'config.yaml')):
            self.config = Config(os.path.join(model_path, 'config.yaml'))
            self._model_factory = ModelFactory(self.config)

    @staticmethod
    def _load_pickled_model(model_type: str, model_directory: str):
        """Load a pickled model of given type, using file name (which is the class name)."""
        model_class_name = os.listdir(model_directory)[0].split('.')[0]
        if model_type != 'data_preprocessor':
            model_class = registry.get(model_type, model_class_name)
        else:
            model_class = DataPreprocessor
        model_instance = model_class()
        model_instance.load(model_directory)
        return model_instance
