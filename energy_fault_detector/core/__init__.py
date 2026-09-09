"""Abstract base classes and shared infrastructure for the fault-detection pipeline.

Concrete implementations of these contracts live in their own subpackages
(autoencoders, anomaly_scores, threshold_selectors, data_preprocessing).
"""

from .anomaly_score import AnomalyScore
from .data_transformer import DataTransformer
from .threshold_selector import ThresholdSelector
from .fault_detection_result import FaultDetectionResult, ModelMetadata

__all__ = [
    "AnomalyScore",
    "DataTransformer",
    "ThresholdSelector",
    "FaultDetectionResult",
    "ModelMetadata",
]
