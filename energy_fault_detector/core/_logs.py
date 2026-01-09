"""Logging settings"""

import os
from pathlib import Path
import logging.config as logging_config

import yaml


def setup_logging(default_path: str | Path = 'logging.yaml', env_key: str = 'LOG_CFG') -> None:
    """Setup logging configuration

    Args:
        default_path (str or Path): default logging configuration file. Default is 'logging.yaml'
        env_key (str): Environment variable holding logging config file path (overrides default_path). Default is
            'LOG_CFG'
    """

    path = Path(default_path)
    value = os.getenv(env_key, None)
    if value:
        path = Path(value)

    try:
        with open(path, 'rt', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
            # check paths exist or create them:
            for _, handler in config['handlers'].items():
                filename = handler.get('filename')
                if filename:
                    # Resolve path and create parent directories if they don't exist
                    Path(filename).parent.mkdir(parents=True, exist_ok=True)

        logging_config.dictConfig(config)
    except Exception as e:
        raise ValueError(f"Error setting up logging: {e}") from e
