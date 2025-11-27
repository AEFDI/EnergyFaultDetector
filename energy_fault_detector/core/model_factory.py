
from typing import Union, Dict

from energy_fault_detector.registration import registry
from energy_fault_detector.config import Config
from energy_fault_detector.data_preprocessing import DataPreprocessor
from energy_fault_detector.core import (
    Autoencoder, AnomalyScore, ThresholdSelector
)

ModelType = Union[Autoencoder, AnomalyScore, ThresholdSelector, DataPreprocessor]


class ModelFactory:
    def __init__(self, config: Config) -> None:
        """
        Initialize the ModelFactory with a given configuration.

        Args:
            config (Config): Configuration object containing model parameters.
        """
        self.config = config
        self._models: Dict[str, ModelType] = {}
        self._initialize_models()

    def _initialize_models(self) -> None:
        """
        Initialize all models based on the configuration if they haven't been initialized yet.
        This method populates the _models dictionary with instances of the models.
        """

        # Retrieve training configuration
        train_dict = self.config['train']

        # data preprocessor
        self._models['data_preprocessor'] = DataPreprocessor(
            steps=train_dict.get('data_preprocessor', {}).get('steps'),
            **train_dict.get('data_preprocessor', {}).get('params', {})
        )

        # Loop through each model type and initialize the corresponding model
        model_types = ['autoencoder', 'anomaly_score', 'threshold_selector']
        for model_type in model_types:
            model_params = train_dict.get(model_type)
            model_class = registry.get(model_type, model_params['name'])

            self._models[model_type] = model_class(**model_params.get('params', {}))

    @property
    def data_preprocessor(self) -> DataPreprocessor:
        """
        Get the data preprocessor model.

        Returns:
            DataPreprocessor: The initialized data preprocessor model.
        """
        return self._models.get('data_preprocessor')

    @property
    def autoencoder(self) -> Autoencoder:
        """
        Get the autoencoder model.

        Returns:
            Autoencoder: The initialized autoencoder model.
        """
        return self._models.get('autoencoder')

    @property
    def anomaly_score(self) -> AnomalyScore:
        """
        Get the anomaly score model.

        Returns:
            AnomalyScore: The initialized anomaly score model.
        """
        return self._models.get('anomaly_score')

    @property
    def threshold_selector(self) -> ThresholdSelector:
        """
        Get the threshold selector model.

        Returns:
            ThresholdSelector: The initialized threshold selector model.
        """
        return self._models.get('threshold_selector')
