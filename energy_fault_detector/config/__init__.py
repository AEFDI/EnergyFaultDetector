"""Configuration of the fault-detection pipeline, loaded and validated from YAML."""

from energy_fault_detector.config.config import Config, InvalidConfigFile
from energy_fault_detector.config.quickstart_config import generate_quickstart_config

__all__ = [
    "Config",
    "InvalidConfigFile",
    "generate_quickstart_config",
]
