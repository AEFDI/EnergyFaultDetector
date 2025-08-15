"""Main fault detection class"""

import logging
from typing import Optional, Any, Tuple, List
from datetime import datetime
import os

import pandas as pd
import numpy as np
from tensorflow.keras.backend import clear_session

from energy_fault_detector.core.fault_detection_model import FaultDetectionModel
from energy_fault_detector.threshold_selectors import AdaptiveThresholdSelector
from energy_fault_detector.data_preprocessing.data_preprocessor import DataPreprocessor
from energy_fault_detector.data_preprocessing.data_clipper import DataClipper
from energy_fault_detector.root_cause_analysis import Arcana
from energy_fault_detector.config import Config
from energy_fault_detector._logs import setup_logging
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult, ModelMetadata

setup_logging(os.path.join(os.path.dirname(__file__), 'logging.yaml'))
logger = logging.getLogger('energy_fault_detector')


class FaultDetector(FaultDetectionModel):
    """Main class for fault detection in renewable energy assets and power grids.

    Args:
        config (Optional[Config]): a Config object with fault detection configuration. Defaults to None.
            If None, the models need to be loaded from a path using the `load_models` method.
        model_directory (str, optional): directory with models. Defaults to 'output'.
        model_subdir (Optional[Any], optional): if given, the subdirectory of the model objects. Defaults to None.
            If None, the current datetime (formatted yyyyMMdd_HHmmSS) is used.
            Complete path to models:

                - train method: model_directory/model_subdir/trained_model
                - tune method: model_directory/model_subdir/tune_method

    Attributes:
        anomaly_score: AnomalyScore object
        autoencoder: Autoencoder object
        threshold_selector: ThresholdSelector object
        data_preprocessor: DataPreprocessorPipeline object
    """

    def __init__(self, config: Optional[Config] = None, model_directory: str = 'output',
                 model_subdir: Optional[Any] = None):
        super().__init__(config=config, model_directory=model_directory, model_subdir=model_subdir)
        if config is None:
            logger.info('No configuration set - load models and config from path with the `load_models` method.')
        else:
            self._init_models()

    def preprocess_train_data(self, sensor_data: pd.DataFrame, normal_index: pd.Series, fit_preprocessor: bool = True
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """ Preprocesses the training data using the configured data_preprocessor

        Args:
            sensor_data (pd.DataFrame): unprocessed training data
            normal_index (pd.Series): unprocessed normal index
            fit_preprocessor (bool, optional): if True the preprocessor is fitted. If False the preprocessor is not
                fitted and the user has to provide a ready-to-use preprocessor by loading models before calling this
                function.
        Returns: tuple of (pd.Dataframe, pd.Dataframe, pd.Series)
            x_prepped (pd.DataFrame): preprocessed normal training data
            x: ordered training data (unprocessed)  # needed for _fit_threshold
            y: ordered normal_index (unprocessed)  # needed for _fit_threshold
        """
        x = sensor_data.sort_index()
        if normal_index is not None:
            y = normal_index.sort_index()
        else:
            # assume only 'normal behaviour' in x
            y = pd.Series(np.full(len(x), True), index=x.index)

        if self.config.data_clipping:
            logger.info('Clip data before scaling.')
            data_clipper = DataClipper(**self.config.data_clipping_params)
            data_clipper.fit(x=x)
            x = data_clipper.transform(x)

        x_normal = x[y]  # filter normal before data prep
        if fit_preprocessor:
            logger.info('Fit preprocessor pipeline.')
            self.data_preprocessor.fit(x_normal)

        x_prepped = self.data_preprocessor.transform(x_normal)
        return x_prepped, x, y

    def fit(self, sensor_data: pd.DataFrame, normal_index: pd.Series = None, save_models: bool = True,
            overwrite_models: bool = False, fit_autoencoder_only: bool = False, fit_preprocessor: bool = True,
            **kwargs) -> ModelMetadata:
        """Fit models on the given sensor_data and save them locally and return the metadata.

        Args:
            sensor_data (pd.DataFrame): DataFrame with the sensor data of one asset for a specific time window.
                The timestamp should be the index and the sensor values as columns.
            normal_index (Optional[pd.Series]): Series indicating normal behavior as boolean with the timestamp as
                index.
                Optional; if not provided, assumes all sensor_data represents normal behavior.
            save_models (bool, optional): Whether to save models. Defaults to True.
            overwrite_models (bool, optional): If True, existing model directories can be overwritten. Defaults to
                False.
            fit_autoencoder_only (bool, optional): If True, only fit the data preprocessor and autoencoder objects.
                Defaults to False.
            fit_preprocessor (bool, optional): If True, the preprocessor is fitted. Defaults to True.

        Returns:
            ModelMetadata: metadata of the trained model: model_date, model_path, model reconstruction errors
            of the training and validation data.
        """
        clear_session()
        model_path = None  # default value (will be overwritten by self._save if the models are saved).
        x_prepped, x, y = self.preprocess_train_data(sensor_data=sensor_data, normal_index=normal_index,
                                                     fit_preprocessor=fit_preprocessor)
        train_recon_error, val_recon_error = None, None
        x_train, x_val = self.train_val_split(x_prepped)
        logger.info('Train autoencoder.')
        self.autoencoder.fit(x=x_train, x_val=x_val, verbose=self.config.verbose)

        train_recon_error = self.autoencoder.get_reconstruction_error(x_train)
        if x_val is not None:
            if len(x_val) > 0:
                val_recon_error = self.autoencoder.get_reconstruction_error(x_val)

        if not fit_autoencoder_only:
            self._fit_threshold(x=x, y=y, x_val=x_val, fit_on_validation=self.config.fit_threshold_on_val)

        # save the models
        if save_models:
            model_path, model_date = self.save_models(model_name='trained_model', overwrite=overwrite_models)
        else:
            model_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ModelMetadata(
            model_date=model_date,
            model_path=model_path,
            train_recon_error=train_recon_error,
            val_recon_error=val_recon_error
        )

    def tune(self, sensor_data: pd.DataFrame, normal_index: pd.Series,
             pretrained_model_path: str = None, new_learning_rate: float = 0.0001, tune_epochs: int = 10,
             tune_method: str = 'full', save_models: bool = True, overwrite_models: bool = False,
             data_preprocessor: DataPreprocessor = None) -> ModelMetadata:
        """FaultDetector finetuning via the following methods:
            'full' (all autoencoder weights + threshold and anomaly-score scaling will be adapted),
            'decoder' (only decoder weights + threshold will be adapted),
            'threshold' (only the threshold and anomaly-score scaling is adapted)

        Notes: Parameters tune_epochs and new_learning_rate should be chosen carefully while considering
            potential overfitting issues depending on the similarity of the tuning data and the training data.

        Args:
            sensor_data (pd.DataFrame): DataFrame with the sensor data of one asset for a specific time window.
                The timestamp should be the index and the sensor values as columns.
            normal_index (pd.Series): Series indicating normal behavior as boolean with the timestamp as index.
            pretrained_model_path (Optional[str], optional): Path to pretrained model. If None, assumes attributes
                data_preprocessor, autoencoder, anomaly_score, and threshold_selector contain fitted instances.
            tune_epochs (int, optional): Number of epochs to fine-tune. Defaults to 10.
            new_learning_rate (float, optional): Learning rate to tune the autoencoder with. Defaults to 0.0001.
            tune_method (str, optional): Possible options:
                'full' (all autoencoder weights + threshold and anomaly-score scaling will be adapted),
                'decoder' (only decoder weights + threshold will be adapted),
                'threshold' (only the threshold and anomaly-score scaling is adapted)
                Defaults to 'full'.
            save_models (bool, optional): Whether to save models. Defaults to True.
            overwrite_models (bool, optional): If True, existing model directories can be overwritten. Defaults to
                False.
            data_preprocessor (Optional[DataPreprocessor], optional): Optional prefitted data preprocessor. Useful
                when using a generic preprocessor for all models.

        Returns:
            ModelMetadata: metadata of the trained model with model_date, model_path, model reconstruction errors
            of the training and validation data.
        """

        if tune_method not in ['threshold', 'decoder', 'full']:
            raise ValueError(f'Unknown tune method {tune_method}.')

        if tune_method == 'threshold' and self.config.fit_threshold_on_val:
            logger.warning('Fine-tuning using only validation data for threshold does not make sense if only the'
                           ' threshold is tuned! Setting fit_threshold_on_val to False.')
            self.config['train']['threshold_selector']['fit_on_val'] = False

        if pretrained_model_path is not None:
            self.load_models(model_path=pretrained_model_path, load_threshold=True)
        else:
            if self.autoencoder is None:
                raise ValueError('No models loaded and no pretrained_model_path provided!')

        clear_session()
        x = sensor_data.sort_index()
        if normal_index is not None:
            y = normal_index.sort_index()
        else:
            # assume only 'normal behaviour' in x
            y = pd.Series(np.full(len(x), True), index=x.index)

        x_normal = x[y.values]  # use the boolean array, so it works for multi-index series/dataframes as well
        if data_preprocessor is not None:
            self.data_preprocessor = data_preprocessor

        train_recon_error = None
        val_recon_error = None
        x_prepped = self.data_preprocessor.transform(x_normal)
        x_train, x_val = self.train_val_split(x_prepped)
        if tune_method != 'threshold':
            if tune_method == 'full':
                self.autoencoder.tune(x=x_train, x_val=x_val, learning_rate=new_learning_rate, tune_epochs=tune_epochs,
                                      verbose=self.config.verbose)
            else:  # tune_method == 'decoder':
                self.autoencoder.tune_decoder(x=x_train, x_val=x_val, learning_rate=new_learning_rate,
                                              tune_epochs=tune_epochs, verbose=self.config.verbose)

            train_recon_error = self.autoencoder.get_reconstruction_error(x_train)
            val_recon_error = self.autoencoder.get_reconstruction_error(x_val) if len(x_val) > 0 else None

        # tune/fit threshold
        self._fit_threshold(x=x, y=y, x_val=x_val, fit_on_validation=self.config.fit_threshold_on_val)

        model_path = None
        if save_models:
            model_name = tune_method + '_tuned_model'
            model_path, model_date = self.save_models(model_name=model_name, overwrite=overwrite_models)
        else:
            model_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ModelMetadata(
            model_path=model_path,
            model_date=model_date,
            train_recon_error=train_recon_error,
            val_recon_error=val_recon_error
        )

    def predict(self, sensor_data: pd.DataFrame, model_path: Optional[str] = None,
                root_cause_analysis: bool = False, track_losses: bool = False,
                track_bias: bool = False) -> FaultDetectionResult:
        """Predict with given models for a specific asset

        Args:
            sensor_data (pd.DataFrame): DataFrame with the sensor data of one asset for a specific time window.
                The timestamp should be the index and the sensor values as columns.
            model_path (Optional[str], optional): Path to the models to be applied. If None, assumes the attributes
                data_preprocessor, autoencoder, anomaly_score, and threshold_selector contain fitted instances.
            root_cause_analysis (bool, optional): Whether to run ARCANA. Defaults to False.
            track_losses (bool, optional): Optional; if True, ARCANA losses will be tracked over the iterations. Defaults to False.
            track_bias (bool, optional): Optional; if True, ARCANA bias will be tracked over the iterations. Defaults to False.

        Returns:
            FaultDetectionResult: with the following DataFrames:
                - predicted_anomalies: DataFrame with a column 'anomaly' (bool).
                - reconstruction: DataFrame with reconstruction of the sensor data with timestamp as index.
                - deviations: DataFrame with reconstruction errors.
                - anomaly_score: DataFrame with anomaly scores for each timestamp.
                - bias_data: DataFrame with ARCANA results with timestamp as index. None if ARCANA was not run.
                - arcana_losses: DataFrame containing recorded values for all losses in ARCANA. None if ARCANA was not run.
                - tracked_bias: List of DataFrames. None if ARCANA was not run.
        """

        x = sensor_data.sort_index()

        if model_path is not None:
            self.load_models(model_path=model_path, load_threshold=True)
        else:
            if self.data_preprocessor is None:
                raise ValueError('No models loaded and no model_path provided!')

        x_prepped = self.data_preprocessor.transform(x).sort_index()
        column_order = x_prepped.columns

        if hasattr(self.autoencoder, 'conditional_features'):
            x_predicted = self.autoencoder.predict(x_prepped, return_conditions=True, verbose=self.config.verbose)
            x_predicted = x_predicted[column_order]
        else:
            x_predicted = self.autoencoder.predict(x_prepped, verbose=self.config.verbose)
        recon_error = self.autoencoder.get_reconstruction_error(x_prepped)

        # inverse transform predictions so they are comparable to the raw data
        df_inverse_scaled_pred = self.data_preprocessor.inverse_transform(x_predicted)

        scores = self.anomaly_score.transform(recon_error)
        predicted_anomalies = self.predict_anomalies(scores, x_prepped)
        predicted_anomalies = pd.DataFrame(data=predicted_anomalies, columns=['anomaly'], index=scores.index)

        if root_cause_analysis:
            logger.info('Run root cause analysis.')
            df_arcana_bias, arcana_losses, tracked_bias = self.run_root_cause_analysis(sensor_data=sensor_data,
                                                                                       track_losses=track_losses,
                                                                                       track_bias=track_bias)
        else:
            df_arcana_bias = None
            arcana_losses = None
            tracked_bias = None

        return FaultDetectionResult(
            predicted_anomalies=predicted_anomalies,
            reconstruction=df_inverse_scaled_pred,
            recon_error=recon_error,
            anomaly_score=pd.DataFrame(data=scores, index=x_predicted.index, columns=['value']),
            bias_data=df_arcana_bias,
            arcana_losses=arcana_losses,
            tracked_bias=tracked_bias
        )

    def predict_anomaly_score(self, sensor_data: pd.DataFrame) -> pd.Series:
        """Predict the anomaly score."""

        x_prepped = self.data_preprocessor.transform(sensor_data)
        recon_error = self.autoencoder.get_reconstruction_error(x_prepped)
        scores = self.anomaly_score.transform(recon_error)
        return scores

    def predict_anomalies(self, scores: pd.Series, x_prepped: pd.DataFrame = None) -> pd.Series:
        """Predict anomalies based on anomaly scores."""

        if self.threshold_selector.__class__.__name__ == 'AdaptiveThresholdSelector':
            predicted_anomalies, _ = self.threshold_selector.predict(x=scores, scaled_ae_input=x_prepped)
        else:
            predicted_anomalies = self.threshold_selector.predict(scores)

        return predicted_anomalies

    def run_root_cause_analysis(self, sensor_data: pd.DataFrame, track_losses: bool = False,
                                track_bias: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
        """Run ARCANA

        Args:
            sensor_data: pandas DataFrame containing the sensor data which should be analyzed.
            track_losses: optional bool. If True the arcana losses will be tracked over the iterations
            track_bias: optional bool. If True the arcana bias will be tracked over the iterations

        Returns: Tuple of (pd.DataFrame, pd.DataFrame, List[pd.DataFrame])
            df_arcana_bias: pandas dataframe containing the arcana bias.
            arcana_losses: dictionary containing loss names as keys and lists of loss values as values.
            tracked_bias: list of pandas dataframe containing the arcana bias recorded over the iterations.
        """

        x_prepped = self.data_preprocessor.transform(sensor_data)
        if self.config is None:
            # backwards compatibility, old models did not save config, just use default parameters
            rca = Arcana(model=self.autoencoder)
        else:
            rca = Arcana(model=self.autoencoder, **self.config.arcana_params)

        df_arcana_bias, arcana_losses, tracked_bias = rca.find_arcana_bias(x=x_prepped,
                                                                           track_losses=track_losses,
                                                                           track_bias=track_bias)
        return df_arcana_bias, arcana_losses, tracked_bias

    def _fit_threshold(self, x: pd.DataFrame, y: pd.Series, x_val: pd.DataFrame, fit_on_validation: bool = False
                       ) -> None:
        """Fit AnomalyScore and ThresholdSelector objects."""

        # Fit score object only on normal data (all training + validation data)
        x_prepped_all = self.data_preprocessor.transform(x)
        deviations = self.autoencoder.get_reconstruction_error(x_prepped_all)
        y_ = y.loc[deviations.index]
        self.anomaly_score.fit(deviations[y_.values])  # use series values for compatibility with a multi-index

        scores = self.anomaly_score.transform(deviations)
        if fit_on_validation:
            logger.info('Fit threshold on validation data')
            if x_val is None:
                logger.warning('No validation data available, fit threshold on full training data.')
                x_val = x

            x_val_all = x_prepped_all.sort_index().loc[x_val.index.min():]  # including known anomalies
            re_val = self.autoencoder.get_reconstruction_error(x_val_all)
            scores = self.anomaly_score.transform(re_val)

        logger.info('Fit threshold.')
        # fit threshold - x_prepped and labels are filtered based on scores (all or validation data only) used
        if isinstance(self.threshold_selector, AdaptiveThresholdSelector):
            self.threshold_selector.fit(scaled_ae_input=x_prepped_all.loc[scores.index],
                                        anomaly_score=scores,
                                        normal_index=y.loc[scores.index])
        else:
            self.threshold_selector.fit(x=scores, y=y.loc[scores.index])
