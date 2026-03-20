"""Configuration object for anomaly detection"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Callable, Any
import warnings

from energy_fault_detector.config.base_config import BaseConfig, InvalidConfigFile

logger = logging.getLogger('energy_fault_detector')

# TODO: Clearer error/validation messages
# Consider pydantic or data classes instead of cerberus
TRAIN_SCHEMA = {
    'anomaly_score': {
        'type': 'dict',
        'required': True,
        'schema': {
            'name': {'type': 'string', 'required': True},
            'params': {'type': 'dict', 'required': False},
        }
    },
    'autoencoder': {
        'type': 'dict',
        'required': True,
        'schema': {
            'name': {'type': 'string', 'required': True},
            'params': {'type': ['dict', 'list'], 'required': True},
            'verbose': {'type': 'integer', 'required': False},
        }
    },
    'data_preprocessor': {
        'type': 'dict',
        'required': True,
        'allow_unknown': False,
        'nullable': True,  # if not specified, create default pipeline
        'schema': {
            'params': {'type': 'dict', 'required': False, 'nullable': True,},
            'steps': {
                'type': 'list',
                'required': False,
                'nullable': True,
                'schema': {
                    'type': 'dict',
                    'allow_unknown': True
                }
            },
        }
    },
    'threshold_selector': {
        'type': 'dict',
        'required': True,
        'schema': {
            'name': {'type': 'string', 'required': True},
            'fit_on_val': {'type': 'boolean', 'required': False, 'default': False},
            'params': {'type': 'dict', 'required': False},
        }
    },
    'data_clipping': {
        'type': 'dict',
        'required': False,  # no data clipping if not specified
        'schema': {
            'lower_percentile': {'type': 'float', 'required': False},
            'upper_percentile': {'type': 'float', 'required': False},
        }
    },
    'data_splitter': {
        'type': 'dict',
        'required': False,  # defaults if not specified
        'schema': {
            'type': {'type': 'string', 'required': False, 'default': 'BlockDataSplitter',
                     'allowed': ['DataSplitter', 'BlockDataSplitter', 'blocks', 'sklearn', 'train_test_split', 'train_val_split']},
            'train_block_size': {'type': 'integer', 'required': False, 'dependencies': {'type': ['DataSplitter', 'BlockDataSplitter', 'blocks']}},
            'val_block_size': {'type': 'integer', 'required': False, 'dependencies': {'type': ['DataSplitter', 'BlockDataSplitter', 'blocks']}},
            'validation_split': {'type': 'float', 'required': False, 'dependencies': {'type': ['sklearn', 'train_test_split', 'train_val_split']}},
            'shuffle': {'type': 'boolean', 'required': False, 'dependencies': {'type': ['sklearn', 'train_test_split']}},
        }
    },
}

ROOT_CAUSE_ANALYSIS_SCHEMA = {
    'alpha': {'type': 'float', 'required': False},
    'init_x_bias': {'type': 'string', 'required': False},
    'num_iter': {'type': 'integer', 'required': False},
    'epsilon': {'type': 'float', 'required': False},
    'verbose': {'type': 'boolean', 'required': False},
}

PREDICT_SCHEMA = {
    'criticality': {'type': 'dict', 'required': False, 'schema': {
        'max_criticality': {'type': 'integer', 'required': False}
    }}
}

CONFIG_SCHEMA = {
    'train': {'type': 'dict', 'schema': TRAIN_SCHEMA, 'required': False, 'allow_unknown': True},
    'predict': {'type': 'dict', 'schema': PREDICT_SCHEMA, 'required': False},
    'root_cause_analysis': {'type': 'dict', 'schema': ROOT_CAUSE_ANALYSIS_SCHEMA, 'required': False},
    'dtype': {'type': 'string', 'required': False, 'allowed': ['float32', 'float64']}
}


def _validate_early_stopping(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether early_stopping settings are ok."""

    train_config = config_dict.get('train', {})
    early_stopping = train_config.get('autoencoder', {}).get('params', {}).get('early_stopping', False)

    data_splitter = train_config.get('data_splitter', {})
    validation_split = data_splitter.get('validation_split', 0.0)
    val_block_size = data_splitter.get('val_block_size', 0)

    if not isinstance(validation_split, float):
        validation_split = 0.0

    validation = 0 < validation_split < 1 or val_block_size > 0

    if early_stopping and not validation:
        msg = f'Configuration is not valid: If early_stopping is enabled either validation_split or ' \
              f'val_block_size must be given. If validation_split is used, it must be a float >0 and <1.'
        raise InvalidConfigFile(msg)

    return config_dict


def _parse_timedelta(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Parse sequence_builder.ts_freq from a compact string (e.g. '10m') to np.timedelta64.

    Expects `ts_freq` under: train.autoencoder.params.sequence_builder.ts_freq
    and supports strings like '10m', '1h', '30s'.
    """
    train_cfg = config_dict.get("train", {})
    ae_cfg = train_cfg.get("autoencoder", {}) or {}
    params_cfg = ae_cfg.get("params", {}) or {}
    seq_builder_cfg = params_cfg.get("sequence_builder", {}) or {}
    ts_freq = seq_builder_cfg.get("ts_freq")

    if isinstance(ts_freq, str):
        # Parse '10m', '1h', etc.
        digits = "".join(ch for ch in ts_freq if ch.isdigit())
        unit = "".join(ch for ch in ts_freq if not ch.isdigit())
        if not digits or not unit:
            raise ValueError(f"Unexpected value for `ts_freq`: {ts_freq!r}. Expected format like '10m', '1h'.")
        import numpy as np  # noqa: F401
        value = int(digits)
        seq_builder_cfg["ts_freq"] = np.timedelta64(value, unit)
        params_cfg["sequence_builder"] = seq_builder_cfg
        ae_cfg["params"] = params_cfg
        train_cfg["autoencoder"] = ae_cfg
        config_dict["train"] = train_cfg

    return config_dict


class Config(BaseConfig):
    """Configuration class. Either config_filename or config_dict must be provided.
    Reads a yaml file with the anomaly detection configuration and sets corresponding settings.
    """

    def __init__(self, config_filename: str | Path = None, config_dict: Dict[str, Any] = None):
        super().__init__(config_filename=config_filename, config_dict=config_dict)
        self._schema = CONFIG_SCHEMA
        self._extra_validation_checks: List[Callable[[Dict], Dict]] = [_parse_timedelta,
                                                                       _validate_early_stopping]
        self.read_config()

    def __getitem__(self, item) -> Any:
        return self.config_dict[item]

    def __contains__(self, item):
        return item in self.config_dict.keys() or item == 'models'

    @property
    def data_preprocessor_steps(self) -> List[Dict[str, Any]]:
        """Get the data preprocessor steps."""
        data_prep = self.config_dict.get('train', {}).get('data_preprocessor', {}) or {}
        params = data_prep.get('params') or {}
        steps = data_prep.get('steps') or []

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
    def root_cause_analysis(self) -> bool:
        """Whether to run ARCANA."""
        return 'root_cause_analysis' in self.config_dict

    @property
    def arcana_params(self) -> Dict[str, Any]:
        """Get the ARCANA parameters."""
        return self.config_dict.get('root_cause_analysis', {})

    @property
    def data_split_params(self) -> Dict[str, Any]:
        """DataSplitter or train_test_split parameters."""
        return self.config_dict.get('train', {}).get('data_splitter', {})

    @property
    def data_clipping(self) -> bool:
        """Whether to clip training data."""
        return 'data_clipping' in self.config_dict.get('train', {})

    @property
    def data_clipping_params(self) -> Dict[str, Any]:
        """Data clipping parameters."""
        return self.config_dict.get('train', {}).get('data_clipping', {})

    @property
    def max_criticality(self) -> Optional[int]:
        """Max criticality value."""
        return self.config_dict.get('predict', {}).get('criticality', {}).get('max_criticality', 144)

    @property
    def fit_threshold_on_val(self) -> bool:
        """Whether to fit threshold on validation data only."""
        return self.config_dict.get('train', {}).get('threshold_selector', {}).get('fit_on_val', False)

    @property
    def dtype(self):
        """Data type, float32 by default."""
        return self.config_dict.get('dtype', 'float32')


def _data_preprocessor_params_to_steps(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate old data_preprocessor params into a 'steps' specification.

    Returns:
        List of step-spec dicts to be passed to the DataPreprocessor, e.g.:
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
        steps.append({"name": "standard_scaler", "step_name": "scaler"})
    else:
        steps.append({"name": "minmax_scaler", "step_name": "scaler"})

    return steps
