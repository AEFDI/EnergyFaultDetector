
from typing import Union, Dict, TYPE_CHECKING

from energy_fault_detector.registration import registry
from energy_fault_detector.config import Config
from energy_fault_detector.data_preprocessing import DataPreprocessor
from energy_fault_detector.core import AnomalyScore, ThresholdSelector

if TYPE_CHECKING:
    from energy_fault_detector.core.autoencoder import Autoencoder

ModelType = Union["Autoencoder", AnomalyScore, ThresholdSelector, DataPreprocessor]


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
        # runtime imports
        from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder
        from energy_fault_detector.autoencoders.seq2seq_autoencoder import Seq2SeqAutoencoder
        from energy_fault_detector.autoencoders.seq2one_autoencoder import Seq2OneAutoencoder

        train_dict = self.config["train"]

        # Data preprocessor
        data_prep_conf = train_dict.get("data_preprocessor", {}) or {}
        self._models["data_preprocessor"] = DataPreprocessor(
            steps=data_prep_conf.get("steps"),
            **data_prep_conf.get("params", {}),
        )

        # autoencoder
        ae_params = train_dict.get("autoencoder")
        ae_class = registry.get("autoencoder", ae_params["name"])
        ae_kwargs = dict(ae_params.get("params", {}))

        # If this is a sequence AE, build the SequenceDatasetBuilder from config
        if issubclass(ae_class, (Seq2SeqAutoencoder, Seq2OneAutoencoder)):
            builder_conf = ae_kwargs.pop("sequence_builder", None)
            if builder_conf is None:
                raise ValueError(
                    f"sequence_builder config is required for sequence autoencoder {ae_class.__name__}"
                )

            ts_freq = builder_conf["ts_freq"]  # already np.timedelta64 thanks to _parse_timedelta
            sequence_builder = SequenceDatasetBuilder(
                sequence_length=builder_conf["sequence_length"],
                ts_freq=ts_freq,
                overlap=builder_conf.get("overlap", 0),
                pad_incomplete=builder_conf.get("pad_incomplete", False),
                pad_value=builder_conf.get("pad_value", 0.0),
            )

            ae_kwargs["sequence_builder"] = sequence_builder

        self._models["autoencoder"] = ae_class(**ae_kwargs)

        # anomaly_score
        score_params = train_dict.get("anomaly_score")
        score_class = registry.get("anomaly_score", score_params["name"])
        self._models["anomaly_score"] = score_class(**score_params.get("params", {}))

        # threshold_selector
        thresh_params = train_dict.get("threshold_selector")
        thresh_class = registry.get("threshold_selector", thresh_params["name"])
        self._models["threshold_selector"] = thresh_class(**thresh_params.get("params", {}))

    @property
    def data_preprocessor(self) -> DataPreprocessor:
        """
        Get the data preprocessor model.

        Returns:
            DataPreprocessor: The initialized data preprocessor model.
        """
        return self._models.get('data_preprocessor')

    @property
    def autoencoder(self) -> "Autoencoder":
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
