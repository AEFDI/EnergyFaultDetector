"""Main fault detection class"""

import logging
from typing import Optional, Tuple, List
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from energy_fault_detector.core.fault_detection_model import FaultDetectionModel
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult, ModelMetadata
from energy_fault_detector.data_preprocessing.data_preprocessor import DataPreprocessor
from energy_fault_detector.data_preprocessing.data_clipper import DataClipper
from energy_fault_detector.config import Config
from energy_fault_detector.threshold_selectors import FbetaSelector, FDRSelector, AdaptiveThresholdSelector

logger = logging.getLogger('energy_fault_detector')


class FaultDetector(FaultDetectionModel):
    """Main class for fault detection in renewable energy assets and power grids.

    Args:
        config (Optional[Config]):  Config object with fault detection configuration. Defaults to None.
            If None, the models need to be loaded from a path using the `load_models` method.
        model_directory (str, optional): Directory to save models to. Defaults to 'fault_detector_model'.

    Attributes:
        anomaly_score: AnomalyScore object.
        autoencoder: Autoencoder object.
        threshold_selector: ThresholdSelector object.
        data_preprocessor: DataPreprocessorPipeline object.
        save_timestamps: a list of string timestamps indicating when the model was saved.
    """

    def __init__(self, config: Optional[Config] = None, model_directory: str | Path = 'fault_detector_model'):
        super().__init__(config=config, model_directory=model_directory)

    def preprocess_train_data(self, sensor_data: pd.DataFrame, normal_index: pd.Series, fit_preprocessor: bool = True
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Preprocesses the training data using the configured data_preprocessor.
        If categorical features are declared in config file, data is split into categorical and numerical features.
        Boolean features are treated the same as numerical features. Encoded categorical features are scaled or not
        depending on the config flag.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: Preprocessed training data after filtering only normal index, 
            original sensor data (clipped if data clipping is enabled), and normal index.
        """

        x = sensor_data.sort_index()
        if normal_index is not None:
            y = normal_index.sort_index()
        else:
            y = pd.Series(np.full(len(x), True), index=x.index)

        if not x.loc[x.index.duplicated()].empty or not y.loc[y.index.duplicated()].empty:
            raise ValueError('There are duplicated indices in the input dataframe `sensor_data` and/or in the '
                             '`normal_index`, please check your input data.')

        # TODO: add list of relevant features to config and protect relevant features (if present) as well.
        # Determine which features to protect based on config flag
        protect = self.config.protect_conditional_features if self.config else True
        protected_features = (
            self.autoencoder.conditional_features or []
        ) if protect else []

        # Data clipping (outlier clipping)
        if self.config.data_clipping:
            logger.debug('Clip data before scaling.')
            clipper_params = self.config.data_clipping_params.copy()
            if protected_features:
                existing_exclusions = clipper_params.get('features_to_exclude', [])
                clipper_params['features_to_exclude'] = list(set(existing_exclusions + protected_features))
                logger.debug(f'Excluding conditional features from clipping: {protected_features}')
            data_clipper = DataClipper(**clipper_params)
            data_clipper.fit(x=x)
            x = data_clipper.transform(x)

        x_normal = x[y.values]
        # Data preprocessing pipeline
        if fit_preprocessor:
            logger.info('Fit preprocessor pipeline.')
            fit_params = {}
            for step_name in self.data_preprocessor.named_steps.keys():
                if 'column_selector' in step_name or 'low_unique_value_filter' in step_name:
                    fit_params[f'{step_name}__protected_features'] = protected_features
            if protected_features:
                fit_params['protected_features'] = protected_features
            self.data_preprocessor.fit(x_normal, **fit_params)

        x_prepped = self.data_preprocessor.transform(x_normal)
        x_prepped = x_prepped.astype(self.config.dtype)

        if x_prepped.empty:
            raise ValueError(
                "Preprocessed training data is empty after filtering for normal behaviour and applying the "
                "preprocessing pipeline. Check that 'normal_index' marks enough rows as normal and that the "
                "preprocessor (e.g. ForwardFillImputer, ColumnSelector) is not dropping all rows."
            )

        return x_prepped, x, y

    def fit(self, sensor_data: pd.DataFrame, normal_index: pd.Series = None, save_models: bool = True,
            overwrite_models: bool = False, fit_autoencoder_only: bool = False, fit_preprocessor: bool = True,
            **kwargs) -> ModelMetadata:
        """Fit models on the given sensor_data and save them locally and return the metadata."""

        try:
            from keras.backend import clear_session
        except ImportError:
            logger.warning('Could not import tensorflow.keras.backend.clear_session(). Please install tensorflow.')
            raise

        if normal_index is None and isinstance(self.threshold_selector, (FbetaSelector, FDRSelector)):
            raise ValueError("FbetaSelector/FDRSelector require a `normal_index` with both True and False values. "
                             "If your data does not have labels, consider a different threshold selector, such as "
                             "QuantileThresholdSelector or AdaptiveThresholdSelector.")

        non_numeric = sensor_data.select_dtypes(exclude='number').columns.tolist()
        if non_numeric and fit_preprocessor:
                logger.warning("Fitting preprocessor on non-numeric columns. Boolean columns will be treated as "
                               "floats during imputation. Non-declared categorical features will be dropped.")
        clear_session()

        # Preprocess data and fit autoencoder
        model_path = None
        x_prepped, x, y = self.preprocess_train_data(
            sensor_data=sensor_data, normal_index=normal_index, fit_preprocessor=fit_preprocessor
        )

        # Resolve declared conditions against preprocessed data
        self._resolve_conditional_features(x_prepped)

        # Check conditionals: available conditions in original data and surviving conditions during preprocessing.
        if self.autoencoder.is_conditional:
            configured = self.autoencoder.conditional_features or []
            # Uses nested for loop in case categorical cols have been encoded already and contain the original
            # conditional feature name as a substring.
            available = [declared_condition for declared_condition in configured
                         if any(declared_condition in col for col in sensor_data.columns)]
            missing = [declared_condition for declared_condition in configured if declared_condition not in available]
            if missing:
                logger.warning(f"Declared conditions not found in sensor_data will be ignored: "
                               f"{sorted(missing)}. Using: {available or 'none'}")

            if not self.config.protect_conditional_features:
                # Uses nested for loop in case categorical cols have been encoded already and contain the original
                # conditional feature name as a substring.
                surviving = [declared_condition for declared_condition in available
                             if any(declared_condition in col for col in x_prepped.columns)]
                dropped_by_pipeline = set(available or []) - set(surviving)
                
                if dropped_by_pipeline:
                    logger.warning(f"Declared conditions dropped by preprocessing pipeline: "
                                f"{sorted(dropped_by_pipeline)}. Remaining: {surviving or 'none'}")

        train_recon_error, val_recon_error = None, None
        x_train, x_val = self.train_val_split(x_prepped)
        logger.info('Train autoencoder.')
        self.autoencoder.fit(x=x_train, x_val=x_val)

        train_recon_error = self.autoencoder.get_reconstruction_error(x_train)
        if x_val is not None:
            if len(x_val) > 0:
                val_recon_error = self.autoencoder.get_reconstruction_error(x_val)

        if not fit_autoencoder_only:
            self._fit_threshold(x=x, y=y, x_val=x_val, fit_on_validation=self.config.fit_threshold_on_val)

        if save_models:
            model_path, model_date = self.save(overwrite=overwrite_models)
        else:
            model_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ModelMetadata(
            model_date=model_date,
            model_path=model_path,
            train_recon_error=train_recon_error,
            val_recon_error=val_recon_error
        )

    def tune(self, sensor_data: pd.DataFrame, normal_index: Optional[pd.Series] = None,
             pretrained_model_path: Optional[str] = None, new_learning_rate: float = 0.0001, tune_epochs: int = 10,
             tune_method: str = 'full', save_models: bool = True, overwrite_models: bool = False,
             data_preprocessor: Optional[DataPreprocessor] = None) -> ModelMetadata:
        """FaultDetector finetuning via the following methods:
            'full' (all autoencoder weights + threshold and anomaly-score scaling will be adapted),
            'decoder' (only decoder weights + threshold will be adapted),
            'threshold' (only the threshold and anomaly-score scaling is adapted)

        Notes: Parameters tune_epochs and new_learning_rate should be chosen carefully while considering
            potential overfitting issues depending on the similarity of the tuning data and the training data.

        Args:
            sensor_data (pd.DataFrame): DataFrame with the sensor data of one asset for a specific time window.
                The timestamp should be the index and the sensor values as columns.
            normal_index (pd.Series, optional): Series indicating normal behavior as boolean with the timestamp as
                index. If not provided, it is assumed all data in `sensor_data` represents normal behaviour.
                Defaults to None.
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
        if self.config is None:
            raise ValueError("Cannot tune without a configuration. Load models using FaultDetector.load() "
                             "or provide pretrained_model_path.")

        if tune_method not in ['threshold', 'decoder', 'full']:
            raise ValueError(f'Unknown tune method {tune_method}.')

        fit_on_val = self.config.fit_threshold_on_val
        if tune_method == 'threshold' and fit_on_val:
            logger.warning('Fine-tuning using only validation data for threshold does not make sense if only the'
                           ' threshold is tuned! Setting fit_threshold_on_val to False.')
            fit_on_val = False

        if pretrained_model_path is not None:
            self._load_from_path(model_path=pretrained_model_path)
        else:
            if self.autoencoder is None:
                raise ValueError('No models loaded and no pretrained_model_path provided!')

        try:
            from keras.backend import clear_session
        except ImportError:
            logger.warning('Could not import tensorflow.keras.backend.clear_session(). Please install tensorflow.')
            raise

        x = sensor_data.sort_index()
        if normal_index is not None:
            y = normal_index.sort_index()
        else:
            # assume only 'normal behaviour' in x
            y = pd.Series(np.full(len(x), True), index=x.index)

        fit_preprocessor = True
        if data_preprocessor is not None:
            self.data_preprocessor = data_preprocessor
            fit_preprocessor = False

        train_recon_error = None
        val_recon_error = None
        x_prepped, x, y = self.preprocess_train_data(sensor_data=sensor_data, normal_index=normal_index,
                                                     fit_preprocessor=fit_preprocessor)
        self._resolve_conditional_features(x_prepped)

        x_train, x_val = self.train_val_split(x_prepped)
        if tune_method != 'threshold':
            logger.info('Tune autoencoder.')
            if tune_method == 'full':
                self.autoencoder.tune(x=x_train, x_val=x_val, learning_rate=new_learning_rate, tune_epochs=tune_epochs)
            else:  # tune_method == 'decoder':
                self.autoencoder.tune_decoder(x=x_train, x_val=x_val, learning_rate=new_learning_rate,
                                              tune_epochs=tune_epochs)

            train_recon_error = self.autoencoder.get_reconstruction_error(x_train)
            val_recon_error = (
                self.autoencoder.get_reconstruction_error(x_val) if len(x_val) > 0 else None
            )

        # tune/fit threshold
        self._fit_threshold(x=x, y=y, x_val=x_val, fit_on_validation=fit_on_val)

        model_path = None
        if save_models:
            model_name = tune_method + '_tuned_model'
            model_path, model_date = self.save(model_name=model_name, overwrite=overwrite_models)
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
            track_losses (bool, optional): Optional; if True, ARCANA losses will be tracked over the iterations.
                Defaults to False.
            track_bias (bool, optional): Optional; if True, ARCANA bias will be tracked over the iterations.
                Defaults to False.

        Returns:
            FaultDetectionResult: with the following DataFrames:
                - predicted_anomalies: Series with detected anomalies (bool).
                - reconstruction: DataFrame with reconstruction of the sensor data with timestamp as index.
                - recon_error: DataFrame with reconstruction errors.
                - anomaly_score: Series with anomaly scores for each timestamp.
                - bias_data: DataFrame with ARCANA results with timestamp as index. None if ARCANA was not run.
                - arcana_losses: DataFrame containing recorded values for all losses in ARCANA. None if ARCANA was not run.
                - tracked_bias: List of DataFrames. None if ARCANA was not run.
        """

        x = sensor_data.sort_index()

        if model_path is not None:
            self._load_from_path(model_path=model_path)
            self.model_directory = model_path
        else:
            if self.data_preprocessor is None:
                raise ValueError('No models loaded and no model_path provided!')
            logger.debug('No model_path provided; using existing model instances.')

        if not x.loc[x.index.duplicated()].empty:
            raise ValueError('There are duplicated indices in the input dataframe `sensor_data`.')

        x_prepped = self.data_preprocessor.transform(x).sort_index()
        x_prepped = x_prepped.astype(self.config.dtype)
        # Validate conditional features at predict time
        if self.autoencoder.is_conditional:
            required = self.autoencoder.conditional_features or []
            missing_at_predict = [f for f in required if f not in x_prepped.columns]
            if missing_at_predict:
                raise ValueError(f"Conditional features required by the trained model are missing from "
                                 f"sensor_data: {sorted(missing_at_predict)}. The model was trained with "
                                 f"these features and cannot predict without them.")

        column_order = x_prepped.columns

        if self.autoencoder.is_conditional:
            x_predicted = self.autoencoder.predict(x_prepped, return_conditions=True)
            x_predicted = x_predicted[column_order]
            main_cols = [c for c in column_order if c not in self.autoencoder.conditional_features]
            recon_error = self.autoencoder.get_reconstruction_error(x=x_prepped, reconstruction=x_predicted[main_cols])
        else:
            x_predicted = self.autoencoder.predict(x_prepped)
            recon_error = self.autoencoder.get_reconstruction_error(x_prepped, reconstruction=x_predicted)

        # inverse transform predictions, so they are comparable to the raw data
        reconstruction = self.data_preprocessor.inverse_transform(x_predicted)

        scores = self.anomaly_score.transform(recon_error)
        predicted_anomalies = self.predict_anomalies(scores, x_prepped)

        if root_cause_analysis:
            logger.info('Run root cause analysis..')
            df_arcana_bias, arcana_losses, tracked_bias = self.run_root_cause_analysis(sensor_data=sensor_data,
                                                                                       track_losses=track_losses,
                                                                                       track_bias=track_bias)
        else:
            df_arcana_bias = None
            arcana_losses = None
            tracked_bias = None

        return FaultDetectionResult(
            predicted_anomalies=predicted_anomalies,
            reconstruction=reconstruction,
            recon_error=recon_error,
            anomaly_score=scores,
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
            predicted_anomalies, _ = self.threshold_selector.predict(
                x=scores, scaled_ae_input=x_prepped.loc[scores.index])
        else:
            predicted_anomalies = self.threshold_selector.predict(scores)

        if not isinstance(predicted_anomalies, pd.Series):
            predicted_anomalies = pd.Series(predicted_anomalies, index=scores.index)

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
        try:
            from energy_fault_detector.root_cause_analysis.arcana import Arcana
        except ImportError:
            logger.warning('Could not import energy_fault_detector.root_cause_analysis.arcana import Arcana. '
                           'Please ensure tensorflow is installed.')
            raise

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
        """Fit AnomalyScore and ThresholdSelector objects.
        
        Args:
            x: pandas DataFrame with the sensor data.
            y: pandas Series with the labels indicating whether each sample is normal (True) or anomalous (False).
            x_val: pandas DataFrame with the validation sensor data.
            fit_on_validation: bool indicating whether to fit the threshold on the validation data.
        """

        x_prepped_all = self.data_preprocessor.transform(x)
        deviations = self.autoencoder.get_reconstruction_error(x_prepped_all)
        y_ = y.loc[deviations.index]
        # Fit score object only on normal data (all training + validation data)
        self.anomaly_score.fit(deviations[y_.values])  # use series values for compatibility with a multi-index

        scores = self.anomaly_score.transform(deviations)
        if fit_on_validation:
            logger.debug('Fit threshold on validation data')
            if x_val is None:
                logger.warning('No validation data available, fit threshold on full training data.')
                x_val = x

            # Select all prepped data whose index is in the validation set
            val_mask = x_prepped_all.index.isin(x_val.index)
            x_val_all = x_prepped_all[val_mask]
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

    def _resolve_conditional_features(self, sensor_data: pd.DataFrame) -> List[str]:
        """Resolve which declared conditions are actually available in the data.

        If all conditions are missing and the autoencoder is a ConditionalAE,
        falls back to MultilayerAutoencoder. For sequence models, simply clears the
        conditional_features list (they handle None gracefully).

        Args:
            sensor_data: Input DataFrame to check column availability.

        Returns:
            List of available conditional feature names (may be empty).
        """
        configured = self.autoencoder.conditional_features or []
        if not configured:
            return []

        # TODO: this can produce undesired results if any column in the original data contains a substring of a declared
        #  conditional feature name.
        available = [col for col in sensor_data.columns if any(declared_condition in col for declared_condition in configured)]

        if not available:
            self._fallback_if_no_conditionals()
        else:
            self.autoencoder.conditional_features = available
            self.autoencoder.is_conditional = True

        return available

    def _fallback_if_no_conditionals(self) -> None:
        """Handle the case where all conditional features are unavailable.

        For ConditionalAE: replaces with MultilayerAutoencoder (same architecture params).
        For sequence models: just clears conditional_features (they work without).
        """
        from energy_fault_detector.autoencoders.conditional_autoencoder import ConditionalAE
        from energy_fault_detector.autoencoders.multilayer_autoencoder import MultilayerAutoencoder
        from energy_fault_detector.autoencoders.sequence_autoencoder import SequenceAutoencoder

        if isinstance(self.autoencoder, SequenceAutoencoder):
            logger.warning("All conditional features are unavailable. "
                           "Sequence model will proceed without conditional inputs.")
            self.autoencoder.conditional_features = None
            self.autoencoder.is_conditional = False

        elif isinstance(self.autoencoder, ConditionalAE):
            logger.warning("All conditional features are unavailable. "
                           "Falling back from ConditionalAE to MultilayerAutoencoder.")
            ae_params = dict(self.config['train']['autoencoder'].get('params', {}))
            ae_params.pop('conditional_features', None)
            self.autoencoder = MultilayerAutoencoder(**ae_params)

        else:
            self.autoencoder.conditional_features = None
            self.autoencoder.is_conditional = False
