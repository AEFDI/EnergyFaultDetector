"""Configuration object for anomaly detection.

The :class:`Config` class is a Pydantic ``BaseModel`` that validates YAML/dict
input.  Top-level sections (``train``, ``root_cause_analysis``, ``dtype``) are
typed fields accessible via normal attribute access
(``config.train.autoencoder.params``).  File I/O (``Config('file.yaml')``,
``write_config``) is built into the constructor and model methods.

All sections have sensible defaults — ``Config()`` with no arguments produces a
ready-to-use configuration (MultilayerAutoencoder, RMSE anomaly score, quantile
threshold selector, default preprocessing pipeline, sklearn data splitter with
10% validation split).
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError, model_validator

logger = logging.getLogger('energy_fault_detector')


class InvalidConfigFile(Exception):
    """Raise when the configuration file is not valid."""


def _format_validation_error(exc: ValidationError) -> InvalidConfigFile:
    """Convert a Pydantic ``ValidationError`` into an :class:`InvalidConfigFile`."""
    lines = []
    for err in exc.errors():
        loc = '.'.join(str(p) for p in err['loc'])
        lines.append(f"{loc}: {err['msg']}")
    return InvalidConfigFile('Configuration is not valid: ' + '; '.join(lines))


# --- Section models ----------------------------------------------------------
# extra='allow' lets unknown keys survive at every nested level (matching the
# former Cerberus allow_unknown=True).  DataPreprocessorConfig uses
# extra='forbid' because its schema explicitly disallowed unknown keys.


class AnomalyScoreConfig(BaseModel):
    """``train.anomaly_score`` section."""

    model_config = ConfigDict(extra='allow', validate_assignment=True)
    name: str = 'rmse'
    params: Dict[str, Any] = {}

    @model_validator(mode='after')
    def _none_params_to_empty(self) -> 'AnomalyScoreConfig':
        if self.params is None:
            self.params = {}
        return self


class AutoencoderConfig(BaseModel):
    """``train.autoencoder`` section.

    ``params`` is an opaque dict consumed by the registry/model factory
    (includes autoencoder-specific keys like ``sequence_builder``).
    """

    model_config = ConfigDict(extra='allow', validate_assignment=True)
    name: str = 'default'
    params: Dict[str, Any] = {}
    verbose: Optional[int] = None

    @model_validator(mode='after')
    def _none_params_to_empty(self) -> 'AutoencoderConfig':
        if self.params is None:
            self.params = {}
        return self


class DataPreprocessorConfig(BaseModel):
    """``train.data_preprocessor`` section (``extra='forbid'``).

    After validation, ``steps`` is always a list (converted from legacy
    ``params`` if needed, or defaulted to ``[]``).  The ``params`` field is
    cleared after conversion so that ``write_config`` produces the ``steps``
    format.  An empty ``steps`` list means "use the DataPreprocessor's built-in
    default pipeline".
    """

    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    params: Optional[Dict[str, Any]] = None
    steps: Optional[List[Dict[str, Any]]] = []

    @model_validator(mode='before')
    @classmethod
    def _none_to_empty(cls, data: Any) -> Any:
        """Treat ``data_preprocessor: null`` (or missing) as an empty config."""
        if data is None:
            return {}
        return data

    @model_validator(mode='after')
    def _convert_params_to_steps(self) -> 'DataPreprocessorConfig':
        if self.steps and self.params:
            warnings.warn(
                "Both 'data_preprocessor.steps' and 'data_preprocessor.params' provided in config; "
                "'data_preprocessor.steps' take precedence and 'data_preprocessor.params' are ignored. "
                "Note: 'data_preprocessor.params' will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.params = None
        elif self.params and not self.steps:
            warnings.warn(
                "Using deprecated 'data_preprocessor.params' to create a DataPreprocessor. "
                "Please update to 'steps'; 'data_preprocessor.params' will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.steps = _data_preprocessor_params_to_steps(self.params)
            self.params = None
        if self.steps is None:
            self.steps = []
        return self


class ThresholdSelectorConfig(BaseModel):
    """``train.threshold_selector`` section."""

    model_config = ConfigDict(extra='allow', validate_assignment=True)
    name: str = 'quantile'
    fit_on_val: bool = False
    params: Dict[str, Any] = {}

    @model_validator(mode='after')
    def _none_params_to_empty(self) -> 'ThresholdSelectorConfig':
        if self.params is None:
            self.params = {}
        return self


class DataClippingConfig(BaseModel):
    """``train.data_clipping`` section."""

    model_config = ConfigDict(extra='allow', validate_assignment=True)
    lower_percentile: Optional[float] = None
    upper_percentile: Optional[float] = None
    features_to_exclude: Optional[List[str]] = None
    features_to_clip: Optional[List[str]] = None


class DataSplitterConfig(BaseModel):
    """``train.data_splitter`` section.

    ``type`` selects block-style vs sklearn-style splitting.  The remaining
    fields are type-specific options; irrelevant ones are ignored by the
    splitter implementation.

    Defaults: ``sklearn`` with ``validation_split=0.1`` and ``shuffle=False``,
    i.e. the last 10% of the data (in original order) is used as a validation
    set.  This enables early stopping and ``fit_on_val`` out of the box.
    """

    model_config = ConfigDict(extra='allow', validate_assignment=True)
    type: Literal['DataSplitter', 'BlockDataSplitter', 'blocks', 'sklearn',
                  'train_test_split', 'train_val_split'] = 'sklearn'
    train_block_size: Optional[int] = None
    val_block_size: Optional[int] = None
    validation_split: Optional[float] = 0.1
    shuffle: Optional[bool] = False


class TrainConfig(BaseModel):
    """``train`` section.

    All sub-sections have defaults; ``TrainConfig()`` produces a ready-to-use
    configuration.
    """

    model_config = ConfigDict(extra='allow', validate_assignment=True)
    autoencoder: AutoencoderConfig = AutoencoderConfig()
    anomaly_score: AnomalyScoreConfig = AnomalyScoreConfig()
    data_preprocessor: DataPreprocessorConfig = DataPreprocessorConfig()
    threshold_selector: ThresholdSelectorConfig = ThresholdSelectorConfig()
    data_clipping: Optional[DataClippingConfig] = None
    data_splitter: DataSplitterConfig = DataSplitterConfig()
    protect_conditional_features: bool = False


class RootCauseAnalysisConfig(BaseModel):
    """``root_cause_analysis`` section.

    The YAML keeps all ARCANA parameters flat (``alpha``, ``num_iter``, …) at the
    section level.  The :attr:`params` property returns them as a dict for easy
    unpacking (``Arcana(**config.root_cause_analysis.params)``).
    """

    model_config = ConfigDict(extra='allow', validate_assignment=True)
    alpha: Optional[float] = None
    init_x_bias: Optional[str] = None
    num_iter: Optional[int] = None
    epsilon: Optional[float] = None
    verbose: Optional[bool] = None
    max_sample_threshold: Optional[int] = None

    @property
    def params(self) -> Dict[str, Any]:
        """Return all non-None RCA parameters as a dict (including extras)."""
        result = {k: v for k, v in self.__dict__.items() if v is not None and not k.startswith('_')}
        extra = getattr(self, '__pydantic_extra__', None)
        if extra:
            result.update({k: v for k, v in extra.items() if v is not None})
        return result


# --- Extra validation helpers ------------------------------------------------


def _ae_params(model: 'Config') -> Dict[str, Any]:
    """Return the autoencoder params dict."""
    return model.train.autoencoder.params


def _validate_early_stopping(model: 'Config') -> None:
    """Check whether early_stopping settings are consistent with the data splitter."""
    params = _ae_params(model)
    early_stopping = params.get('early_stopping', False)

    splitter = model.train.data_splitter
    block_types = {'DataSplitter', 'BlockDataSplitter', 'blocks'}

    if splitter.type in block_types:
        val_block_size = splitter.val_block_size or 0
        validation = val_block_size > 0
    else:
        validation_split = splitter.validation_split or 0.0
        if not isinstance(validation_split, float):
            validation_split = 0.0
        validation = 0 < validation_split < 1

    if early_stopping and not validation:
        msg = ('Configuration is not valid: If early_stopping is enabled either validation_split or '
               'val_block_size must be given. If validation_split is used, it must be a float >0 and <1.')
        raise InvalidConfigFile(msg)


def _validate_sequence_fit_on_val(model: 'Config') -> None:
    """Warn if ``fit_on_val=True`` with a sequence model and ``shuffle=True``.

    Sequence models need temporally contiguous validation data.  Shuffling
    breaks that contiguity, and the sequence builder will drop windows that cross
    data gaps — potentially leaving no valid windows for threshold fitting.
    """
    params = _ae_params(model)
    has_sequence_builder = "sequence_builder" in params
    fit_on_val = model.train.threshold_selector.fit_on_val
    shuffle = model.train.data_splitter.shuffle or False

    if has_sequence_builder and fit_on_val and shuffle:
        warnings.warn(
            "Using 'fit_on_val=True' with a sequence model and 'shuffle=True' may result in "
            "non-contiguous validation timestamps. The sequence builder will drop windows that "
            "cross data gaps, potentially leaving no valid windows for threshold fitting. "
            "Consider setting 'shuffle: false' or 'fit_on_val: false'.",
            UserWarning,
            stacklevel=2,
        )


# --- DataPreprocessor params->steps translation ------------------------------


def _data_preprocessor_params_to_steps(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate old ``data_preprocessor.params`` into a ``steps`` specification.

    Args:
        params: Legacy params dict.

    Returns:
        List of step-spec dicts to be passed to the DataPreprocessor, e.g.::

            [
                {"name": "duplicate_to_nan", "params": {...}},
                {"name": "counter_diff_transformer", "params": {...}},
                ...
            ]
    """

    p = params or {}
    steps: List[Dict[str, Any]] = []

    # 0. DuplicateValuesToNan
    if p.get("include_duplicate_value_to_nan", False):
        steps.append({
            "name": "duplicate_to_nan",
            "params": {
                "value_to_replace": p.get("value_to_replace", 0),
                "n_max_duplicates": p.get("n_max_duplicates", 144),
                "features_to_exclude": p.get("duplicate_features_to_exclude"),
            },
        })

    # 1. CounterDiffTransformer
    counter_cols = p.get("counter_columns_to_transform", []) or []
    if counter_cols:
        steps.append({
            "name": "counter_diff_transformer",
            "params": {
                "counters": counter_cols,
                "compute_rate": False,
                "reset_strategy": "zero",
            },
        })

    # 2. ColumnSelector
    if p.get("include_column_selector", True):
        steps.append({
            "name": "column_selector",
            "params": {
                "max_nan_frac_per_col": p.get("max_nan_frac_per_col", 0.05),
                "features_to_exclude": p.get("features_to_exclude"),
            },
        })

    # 3. LowUniqueValueFilter
    if p.get("include_low_unique_value_filter", True):
        steps.append({
            "name": "low_unique_value_filter",
            "params": {
                "min_unique_value_count": p.get("min_unique_value_count", 2),
                "max_col_zero_frac": p.get("max_col_zero_frac", 1.0),
            },
        })

    # 4. AngleTransformer
    angles = p.get("angles", []) or []
    if angles:
        steps.append({
            "name": "angle_transformer",
            "params": {"angles": angles},
        })

    # 5. SimpleImputer
    imputer_params: Dict[str, Any] = {"strategy": p.get("imputer_strategy", "mean")}
    if imputer_params["strategy"] == "constant":
        imputer_params["fill_value"] = p.get("imputer_fill_value", None)
    steps.append({"name": "simple_imputer", "params": imputer_params})

    # 6. Scaler
    scale = p.get("scale", "standardize")
    if scale in ["standardize", "standard", "standardscaler"]:
        steps.append({"name": "scaler", "step_name": "scaler", "params": {"scaler_type": "standard"}})
    else:
        steps.append({"name": "scaler", "step_name": "scaler", "params": {"scaler_type": "minmax"}})

    return steps


# --- Config (top-level model) ------------------------------------------------


class Config(BaseModel):
    """Configuration for the anomaly-detection pipeline.

    Loads and validates a YAML configuration file (or an inline dict) using
    Pydantic models.  Top-level sections (``train``, ``root_cause_analysis``,
    ``dtype``) are accessible via normal attribute access
    (e.g. ``config.train.autoencoder.params``).

    All sections have defaults — ``Config()`` with no arguments produces a
    ready-to-use configuration.

    Example:

        .. code-block:: python

            from energy_fault_detector.config import Config

            # defaults
            config = Config()

            # from YAML file
            config = Config('config.yaml')

            # from dict
            config = Config({'train': {...}})

    Args:
        config_filename: Path to a YAML configuration file, or an inline
            configuration dictionary.
        config_dict: Inline configuration dictionary (alternative to passing
            a dict as the first positional argument).
    """

    model_config = ConfigDict(extra='ignore', validate_default=True, validate_assignment=True)
    train: TrainConfig = TrainConfig()
    root_cause_analysis: Optional[RootCauseAnalysisConfig] = None
    dtype: Literal['float32', 'float64'] = 'float32'

    _configuration_file: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode='after')
    def _run_extra_checks(self) -> 'Config':
        """Run post-validation checks."""
        _validate_early_stopping(self)
        _validate_sequence_fit_on_val(self)
        return self

    def __init__(self, config_filename: str | Path | Dict[str, Any] = None,
                 config_dict: Dict[str, Any] = None, **kwargs: Any) -> None:
        if isinstance(config_filename, dict):
            config_dict = config_filename
            config_filename = None

        if config_filename is not None:
            with open(config_filename, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data is None:
                raise InvalidConfigFile('The configuration file is empty!')
            self._log_unknown_keys(data)
            try:
                super().__init__(**data)
            except ValidationError as exc:
                raise _format_validation_error(exc) from exc
            object.__setattr__(self, '_configuration_file', str(config_filename))
        elif config_dict is not None:
            self._log_unknown_keys(config_dict)
            try:
                super().__init__(**config_dict)
            except ValidationError as exc:
                raise _format_validation_error(exc) from exc
        else:
            super().__init__(**kwargs)

    @staticmethod
    def _log_unknown_keys(data: Any) -> None:
        if isinstance(data, dict):
            known_keys = set(Config.model_fields.keys())
            for key in list(data.keys()):
                if key not in known_keys:
                    logger.info('Key `%s` is an unknown field and will be ignored.', key)

    def __repr__(self) -> str:
        return repr(self.model_dump(exclude_none=True))

    # --- serialization --------------------------------------------------------

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of the config.

        .. deprecated::
            Use :meth:`model_dump` (Pydantic's built-in) or direct attribute
            access instead.
        """
        warnings.warn(
            "Config.config_dict is deprecated; use Config.model_dump() or "
            "direct attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model_dump(exclude_none=True)

    # --- write / save --------------------------------------------------------

    def write_config(self, file_name: Optional[str] = None, overwrite: bool = False) -> None:
        """Write the configuration to a YAML file.

        Args:
            file_name: Target path. If ``None``, uses the original configuration file.
            overwrite: If ``False``, raises :class:`FileExistsError` when the file
                already exists.
        """
        if file_name is None and self._configuration_file is None:
            raise ValueError('No file name given and no known configuration file to overwrite.')

        file_name = file_name if file_name is not None else self._configuration_file
        if Path(file_name).exists() and not overwrite:
            raise FileExistsError(f'File {file_name} already exists and overwrite is set to False.')

        conf_dict = self.model_dump(exclude_none=True)

        with open(file_name, 'w', encoding='utf-8') as f:
            yaml.safe_dump(conf_dict, f)

    def save(self, file_name: str, overwrite: bool = False) -> None:
        """Save the configuration to a YAML file. Wrapper for :meth:`write_config`."""
        self.write_config(file_name, overwrite)
