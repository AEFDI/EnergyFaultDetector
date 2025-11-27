"""Configuration object for anomaly detection"""

import logging
from typing import Dict, Optional, List, Callable, Any

from energy_fault_detector.config.base_config import BaseConfig, InvalidConfigFile

logger = logging.getLogger('energy_fault_detector')

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
            'time_series_sampler': {
                'type': 'dict',
                'required': False,
                'allow_unknown': True,
            },
        }
    },
    'data_preprocessor': {
        'type': 'dict',
        'required': True,
        'allow_unknown': False,
        'nullable': True,  # if not specfied, create default pipeline
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
            'type': {'type': 'string', 'required': False, 'default': 'DataSplitter',
                     'allowed': ['DataSplitter', 'BlockDataSplitter', 'blocks', 'sklearn', 'train_test_split']},
            'train_block_size': {'type': 'integer', 'required': False, 'dependencies': {'type': ['DataSplitter', 'BlockDataSplitter', 'blocks']}},
            'val_block_size': {'type': 'integer', 'required': False, 'dependencies': {'type': ['DataSplitter', 'BlockDataSplitter', 'blocks']}},
            'validation_split': {'type': 'float', 'required': False, 'dependencies': {'type': ['sklearn', 'train_test_split']}},
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
    """Parse timedelta string representation to numpy timedelta."""

    ts_freq = config_dict.get('train', {}).get('autoencoder', {}).get('time_series_sampler', {}).get('ts_freq')
    if ts_freq is not None:
        if isinstance(ts_freq, str):
            if (ts_freq.startswith('np.timedelta64(')
                    or ts_freq.startswith('numpy.timedelta64(')) and ts_freq.endswith(')'):
                import numpy  # needed for eval(ts_freq) in case ts_freq.startswith('numpy.timedelta64('))
                import numpy as np  # needed for eval(ts_freq) in case ts_freq.startswith('np.timedelta64(')
                config_dict['train']['autoencoder']['time_series_sampler']['ts_freq'] = eval(ts_freq)
            else:
                raise ValueError('Unexpected value for `ts_freq` (time_series_sampler params)')
    return config_dict


class Config(BaseConfig):
    """Configuration class. Either config_filename or config_dict must be provided.
    Reads a yaml files with the anomaly detection configuration and sets corresponding settings.
    """

    def __init__(self, config_filename: str = None, config_dict: Dict[str, Any] = None):

        # backwards compatibility
        if config_filename is None:
            if 'data' in config_dict:
                config_dict.pop('data')

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
    def verbose(self) -> int:
        """Verbosity Level of the Autoencoder."""
        return self.config_dict.get('train', {}).get('autoencoder', {}).get('verbose', 1)
