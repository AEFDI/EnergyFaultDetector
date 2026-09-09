"""Configuration object for anomaly detection.

The configuration is defined by a set of Pydantic models (section models plus a
top-level :class:`ConfigModel`) that validate the YAML/dict input.  The public
:class:`Config` class is a thin wrapper around :class:`ConfigModel` that adds
file I/O and serialization.  Model fields (``train``, ``predict``,
``root_cause_analysis``, ``dtype``) are accessible via normal attribute access
on the :class:`Config` wrapper (delegated to the underlying model through
``__getattr__``).
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

logger = logging.getLogger('energy_fault_detector')


class InvalidConfigFile(Exception):
    """Raise when the configuration file is not valid."""


def _format_timedelta(value: np.timedelta64) -> str:
    """Format an :class:`np.timedelta64` as a compact string like ``'30s'``.

    Args:
        value: timedelta to format.

    Returns:
        Compact string representation suitable for YAML round-tripping.
    """
    total_seconds = int(value.astype('timedelta64[s]').astype(int))
    return f"{total_seconds}s"


def _dump_value(value: Any) -> Any:
    """Recursively dump nested models/lists; plain dicts/lists are kept as-is."""
    if isinstance(value, BaseModel):
        return _model_to_config_dict(value)
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    return value


def _model_to_config_dict(model: BaseModel) -> Dict[str, Any]:
    """Serialize a validated Pydantic model back to a plain configuration dict.

    Reproduces the structure ``cerberus.Validator.normalized`` produced:

      * fields that are *required* (no default) are always included, even when
        their value is ``None`` (e.g. ``train.data_preprocessor`` which is
        required but nullable);
      * fields that are *optional* (have a default) are included only when their
        value is not ``None`` (falsy defaults such as ``False`` are kept because
        ``False is not None``);
      * unknown fields kept via ``extra='allow'`` are merged back in.

    Args:
        model: The validated configuration model.

    Returns:
        Plain (mutable) dictionary representation.
    """
    result: Dict[str, Any] = {}
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        if not field.is_required() and value is None:
            continue
        result[name] = _dump_value(value)
    extra = getattr(model, '__pydantic_extra__', None)
    if extra:
        result.update(extra)
    return result


# --- Section models ----------------------------------------------------------
# extra='allow' mirrors the former Cerberus Validator(allow_unknown=True) which
# let unknown keys survive at every nested level.  The sole exception is
# DataPreprocessorConfig (extra='forbid') because its Cerberus schema explicitly
# set allow_unknown: False.  Leaves (params, steps items) are left as plain
# mutable dicts -- the package never validated their internals with Cerberus
# either, so this is strict parity.


class AnomalyScoreConfig(BaseModel):
    """``train.anomaly_score`` section."""

    model_config = ConfigDict(extra='allow')
    name: str
    params: Optional[Dict[str, Any]] = None


class AutoencoderConfig(BaseModel):
    """``train.autoencoder`` section.

    ``params`` accepts a dict or a list (the Cerberus schema allowed both) and is
    left opaque: step/autoencoder-specific keys (including ``sequence_builder``)
    are not modeled here and are consumed by the registry/model factory.
    """

    model_config = ConfigDict(extra='allow')
    name: str
    params: Dict[str, Any] | List[Any]
    verbose: Optional[int] = None


class DataPreprocessorConfig(BaseModel):
    """``train.data_preprocessor`` section (``extra='forbid'``)."""

    model_config = ConfigDict(extra='forbid')
    params: Optional[Dict[str, Any]] = None
    steps: Optional[List[Dict[str, Any]]] = None


class ThresholdSelectorConfig(BaseModel):
    """``train.threshold_selector`` section."""

    model_config = ConfigDict(extra='allow')
    name: str
    fit_on_val: bool = False
    params: Optional[Dict[str, Any]] = None


class DataClippingConfig(BaseModel):
    """``train.data_clipping`` section."""

    model_config = ConfigDict(extra='allow')
    lower_percentile: Optional[float] = None
    upper_percentile: Optional[float] = None
    features_to_exclude: Optional[List[str]] = None
    features_to_clip: Optional[List[str]] = None


class DataSplitterConfig(BaseModel):
    """``train.data_splitter`` section.

    Reproduces the Cerberus ``dependencies`` rules: block-size fields require a
    block-style ``type`` and the split/shuffle fields require an sklearn-style
    ``type``.  The default for ``type`` is applied before the ``mode='after'``
    validator runs (Pydantic applies field defaults first), matching Cerberus
    which also applied the default during validation.
    """

    model_config = ConfigDict(extra='allow')
    type: Literal['DataSplitter', 'BlockDataSplitter', 'blocks', 'sklearn',
                  'train_test_split', 'train_val_split'] = 'BlockDataSplitter'
    train_block_size: Optional[int] = None
    val_block_size: Optional[int] = None
    validation_split: Optional[float] = None
    shuffle: Optional[bool] = None

    @model_validator(mode='after')
    def _check_dependencies(self) -> 'DataSplitterConfig':
        block_types = {'DataSplitter', 'BlockDataSplitter', 'blocks'}
        split_types = {'sklearn', 'train_test_split', 'train_val_split'}
        if self.train_block_size is not None and self.type not in block_types:
            raise ValueError(
                "train_block_size depends on type in ['DataSplitter', 'BlockDataSplitter', 'blocks']")
        if self.val_block_size is not None and self.type not in block_types:
            raise ValueError(
                "val_block_size depends on type in ['DataSplitter', 'BlockDataSplitter', 'blocks']")
        if self.validation_split is not None and self.type not in split_types:
            raise ValueError(
                "validation_split depends on type in ['sklearn', 'train_test_split', 'train_val_split']")
        if self.shuffle is not None and self.type not in {'sklearn', 'train_test_split'}:
            raise ValueError("shuffle depends on type in ['sklearn', 'train_test_split']")
        return self


class TrainConfig(BaseModel):
    """``train`` section."""

    model_config = ConfigDict(extra='allow')
    anomaly_score: AnomalyScoreConfig
    autoencoder: AutoencoderConfig
    data_preprocessor: Optional[DataPreprocessorConfig]  # required but nullable -> keep None
    threshold_selector: ThresholdSelectorConfig
    data_clipping: Optional[DataClippingConfig] = None
    data_splitter: Optional[DataSplitterConfig] = None
    protect_conditional_features: bool = False


class RootCauseAnalysisConfig(BaseModel):
    """``root_cause_analysis`` section."""

    model_config = ConfigDict(extra='allow')
    alpha: Optional[float] = None
    init_x_bias: Optional[str] = None
    num_iter: Optional[int] = None
    epsilon: Optional[float] = None
    verbose: Optional[bool] = None
    max_sample_threshold: Optional[int] = None


class CriticalityConfig(BaseModel):
    """``predict.criticality`` section."""

    model_config = ConfigDict(extra='allow')
    max_criticality: Optional[int] = None


class PredictConfig(BaseModel):
    """``predict`` section."""

    model_config = ConfigDict(extra='allow')
    criticality: Optional[CriticalityConfig] = None


# --- Extra validation helpers ------------------------------------------------


def _ae_params(model: 'ConfigModel') -> Optional[Dict[str, Any]]:
    """Return the autoencoder params dict, or ``None`` if not applicable.

    ``params`` can be a dict or a list; only dict params are relevant for the
    extra checks (``sequence_builder``, ``early_stopping`` live inside dicts).
    """
    if model.train is None or model.train.autoencoder is None:
        return None
    params = model.train.autoencoder.params
    if isinstance(params, dict):
        return params
    return None


def _parse_timedelta(model: 'ConfigModel') -> None:
    """Parse ``sequence_builder.ts_freq`` from a compact string to ``np.timedelta64``.

    Expects ``ts_freq`` under ``train.autoencoder.params.sequence_builder.ts_freq``
    and supports strings like ``'10m'``, ``'1h'``, ``'30s'``.
    """
    unit_map = {"min": "m", "sec": "s", "hr": "h", "hour": "h"}
    params = _ae_params(model)
    if params is None:
        return
    seq_builder = params.get("sequence_builder")
    if not isinstance(seq_builder, dict):
        return
    ts_freq = seq_builder.get("ts_freq")
    if isinstance(ts_freq, str):
        digits = "".join(ch for ch in ts_freq if ch.isdigit())
        unit = "".join(ch for ch in ts_freq if not ch.isdigit())
        unit = unit_map.get(unit, unit)
        if not digits or not unit:
            raise ValueError(
                f"Unexpected value for `ts_freq`: {ts_freq!r}. Expected format like '10m', '1h'.")
        seq_builder["ts_freq"] = np.timedelta64(int(digits), unit)


def _validate_early_stopping(model: 'ConfigModel') -> None:
    """Check whether early_stopping settings are consistent with the data splitter."""
    params = _ae_params(model)
    if params is None:
        return
    early_stopping = params.get('early_stopping', False)

    splitter = model.train.data_splitter
    validation_split = getattr(splitter, 'validation_split', 0.0) or 0.0
    val_block_size = getattr(splitter, 'val_block_size', 0) or 0

    if not isinstance(validation_split, float):
        validation_split = 0.0

    validation = 0 < validation_split < 1 or val_block_size > 0

    if early_stopping and not validation:
        msg = ('Configuration is not valid: If early_stopping is enabled either validation_split or '
               'val_block_size must be given. If validation_split is used, it must be a float >0 and <1.')
        raise InvalidConfigFile(msg)


def _validate_sequence_fit_on_val(model: 'ConfigModel') -> None:
    """Warn if ``fit_on_val=True`` with a sequence model and ``shuffle=True``."""
    params = _ae_params(model)
    if params is None:
        return
    has_sequence_builder = "sequence_builder" in params
    fit_on_val = model.train.threshold_selector.fit_on_val
    splitter = model.train.data_splitter
    shuffle = getattr(splitter, 'shuffle', False)

    if has_sequence_builder and fit_on_val and shuffle:
        warnings.warn(
            "Using 'fit_on_val=True' with a sequence model and 'shuffle=True' may result in "
            "non-contiguous validation timestamps. The sequence builder will drop windows that "
            "cross data gaps, potentially leaving no valid windows for threshold fitting. "
            "Consider setting 'shuffle: false' or 'fit_on_val: false'.",
            UserWarning,
            stacklevel=2,
        )


# --- Top-level model ---------------------------------------------------------


class ConfigModel(BaseModel):
    """Top-level Pydantic model for the anomaly-detection pipeline configuration.

    The top-level uses ``extra='ignore'`` (unknown top-level keys are dropped and
    logged) mirroring the former Cerberus behaviour.  Section models use
    ``extra='allow'`` (or ``extra='forbid'`` for the data preprocessor).
    """

    model_config = ConfigDict(extra='ignore', validate_default=True)
    train: Optional[TrainConfig] = None
    predict: Optional[PredictConfig] = None
    root_cause_analysis: Optional[RootCauseAnalysisConfig] = None
    dtype: Literal['float32', 'float64'] = 'float32'

    @model_validator(mode='after')
    def _run_extra_checks(self) -> 'ConfigModel':
        """Run the three post-validation checks formerly handled by Cerberus extras."""
        _parse_timedelta(self)
        _validate_early_stopping(self)
        _validate_sequence_fit_on_val(self)
        return self


# --- Config wrapper ----------------------------------------------------------


class Config:
    """Configuration for the anomaly-detection pipeline.

    Loads and validates a YAML configuration file (or an inline dict) using
    Pydantic models.  Top-level model fields (``train``, ``predict``,
    ``root_cause_analysis``, ``dtype``) are accessible via attribute access on
    the wrapper (e.g. ``config.train`` returns a :class:`TrainConfig` model).

    Example:

        .. code-block:: python

            from energy_fault_detector.config import Config

            # from YAML file
            config = Config.from_yaml('config.yaml')

            # from dict
            config = Config.from_dict({'train': {...}})

            # legacy constructor (still supported)
            config = Config('config.yaml')
            config = Config(config_dict={'train': {...}})

    Args:
        config_filename: Path to a YAML configuration file.
        config_dict: Inline configuration dictionary.
    """

    def __init__(self, config_filename: str | Path = None,
                 config_dict: Dict[str, Any] = None) -> None:
        self._model: Optional[ConfigModel] = None
        self._configuration_file: Optional[str] = None
        if config_filename is not None or config_dict is not None:
            self._configuration_file = str(config_filename) if config_filename else None
            self.read_config(config_dict=config_dict)

    # --- attribute delegation -------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying :class:`ConfigModel`.

        ``__getattr__`` is only called when the attribute is not found through
        normal lookup, so properties/methods defined on :class:`Config` take
        precedence.  Only model fields (``train``, ``predict``, etc.) are
        delegated.
        """
        model = self.__dict__.get('_model')
        if model is not None and name in type(model).model_fields:
            return getattr(model, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return self.config_dict.__repr__()

    # --- factory classmethods -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'Config':
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated configuration instance.

        Raises:
            InvalidConfigFile: If the file is empty or fails validation.
        """
        return cls(config_filename=path)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a configuration from an inline dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Validated configuration instance.

        Raises:
            InvalidConfigFile: If the dictionary is None or fails validation.
        """
        if config_dict is None:
            raise InvalidConfigFile('The configuration file is empty!')
        return cls(config_dict=config_dict)

    # --- loading / validation -------------------------------------------------

    def read_config(self, config_dict: Dict[str, Any] = None, part: str = None) -> None:
        """Read and validate the configuration from file or dict.

        Args:
            config_dict: Inline configuration dictionary.  If ``None``, reads
                from :attr:`_configuration_file`.
            part: If given, extract only this top-level section (legacy compat).
        """
        if config_dict is not None:
            data = config_dict
        elif self._configuration_file is not None:
            with open(self._configuration_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            raise InvalidConfigFile('The configuration file is empty!')

        if data is None:
            raise InvalidConfigFile('The configuration file is empty!')

        if part is not None and isinstance(data, dict) and part in data:
            data = data[part]

        self._load_and_validate(data)

    def _load_and_validate(self, data: Dict[str, Any]) -> None:
        """Validate *data* with Pydantic and store the resulting model."""
        # Drop and log unknown top-level keys (extra='ignore' discards them).
        if isinstance(data, dict):
            known_keys = set(ConfigModel.model_fields.keys())
            for key in list(data.keys()):
                if key not in known_keys:
                    logger.info('Key `%s` is an unknown field and will be ignored.', key)

        try:
            self._model = ConfigModel.model_validate(data)
        except ValidationError as exc:
            lines = []
            for err in exc.errors():
                loc = '.'.join(str(p) for p in err['loc'])
                lines.append(f"{loc}: {err['msg']}")
            raise InvalidConfigFile('Configuration is not valid: ' + '; '.join(lines)) from exc

        if not self.config_dict:
            raise InvalidConfigFile(f'The configuration file is empty for {type(self).__name__}.')

    # --- serialization --------------------------------------------------------

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Return a plain (mutable) dictionary representation of the config.

        The dict is rebuilt from the underlying Pydantic model on every call, so
        it always reflects the current model state.  Leaf dicts (``params``,
        ``steps``) are the same mutable objects stored on the model, so
        in-place mutations (``config.train.autoencoder.params.update(...)``)
        persist.
        """
        return _model_to_config_dict(self._model)

    # --- write / save --------------------------------------------------------

    def write_config(self, file_name: Optional[str] = None, overwrite: bool = False) -> None:
        """Write the configuration to a YAML file.

        Args:
            file_name: Target path. If ``None``, uses the original configuration file.
            overwrite: If ``False``, raises :class:`FileExistsError` when the file
                already exists.
        """
        from copy import deepcopy

        if file_name is None and self._configuration_file is None:
            raise ValueError('No file name given and no known configuration file to overwrite.')

        file_name = file_name if file_name is not None else self._configuration_file
        if Path(file_name).exists() and not overwrite:
            raise FileExistsError(f'File {file_name} already exists and overwrite is set to False.')

        conf_dict = deepcopy(self.config_dict)

        ae_params = conf_dict.get('train', {}).get('autoencoder', {}).get('params', {})
        seq_builder = ae_params.get('sequence_builder')
        if isinstance(seq_builder, dict):
            sb_ts_freq = seq_builder.get('ts_freq')
            if isinstance(sb_ts_freq, np.timedelta64):
                seq_builder['ts_freq'] = _format_timedelta(sb_ts_freq)

        with open(file_name, 'w', encoding='utf-8') as f:
            yaml.safe_dump(conf_dict, f)

    def save(self, file_name: str, overwrite: bool = False) -> None:
        """Save the configuration to a YAML file. Wrapper for :meth:`write_config`."""
        self.write_config(file_name, overwrite)

    def update_config(self, new_config_dict: Dict[str, Any]) -> None:
        """Update the configuration with *new_config_dict* and re-validate.

        Args:
            new_config_dict: Dictionary with new/updated configuration values.
        """
        merged = self.config_dict
        merged.update(new_config_dict)
        self._load_and_validate(merged)
        self._configuration_file = None

    # --- properties with real logic -------------------------------------------

    @property
    def data_preprocessor_steps(self) -> List[Dict[str, Any]]:
        """Get the data preprocessor steps.

        If ``data_preprocessor.steps`` is provided it takes precedence over the
        deprecated ``data_preprocessor.params`` style.
        """
        if self._model.train is None or self._model.train.data_preprocessor is None:
            return []
        dp = self._model.train.data_preprocessor
        params = dp.params or {}
        steps = dp.steps or []

        if steps and params:
            warnings.warn(
                "Both 'data_preprocessor.steps' and 'data_preprocessor.params' provided in config; "
                "'data_preprocessor.steps' take precedence and 'data_preprocessor.params' are ignored. "
                "Note: 'data_preprocessor.params' will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            return steps

        if params and not steps:
            warnings.warn(
                "Using deprecated 'data_preprocessor.params' to create a DataPreprocessor. "
                "Please update to 'steps'; 'data_preprocessor.params' will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            steps = _data_preprocessor_params_to_steps(params)

        return steps or []

    @property
    def arcana_params(self) -> Dict[str, Any]:
        """Get the ARCANA parameters as a plain dict (for ``**`` unpacking)."""
        if self._model.root_cause_analysis is None:
            return {}
        return _model_to_config_dict(self._model.root_cause_analysis)

    @property
    def data_split_params(self) -> Dict[str, Any]:
        """DataSplitter or train_test_split parameters as a plain dict."""
        if self._model.train is None or self._model.train.data_splitter is None:
            return {}
        return _model_to_config_dict(self._model.train.data_splitter)

    @property
    def data_clipping_params(self) -> Dict[str, Any]:
        """Data clipping parameters as a plain dict."""
        if self._model.train is None or self._model.train.data_clipping is None:
            return {}
        return _model_to_config_dict(self._model.train.data_clipping)


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
