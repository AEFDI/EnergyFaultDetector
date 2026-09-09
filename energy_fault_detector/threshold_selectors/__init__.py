"""Threshold selectors that turn anomaly scores into normal/anomaly labels.

A :class:`~energy_fault_detector.core.ThresholdSelector` learns a threshold from
(optionally labelled) anomaly scores. Samples above the threshold are flagged as anomalous.
Each selector subclasses :class:`~energy_fault_detector.core.ThresholdSelector` and is selected
via the ``train.threshold_selector`` config key.
"""

from energy_fault_detector.threshold_selectors.fdr_threshold import FDRSelector
from energy_fault_detector.threshold_selectors.fbeta_threshold import FbetaSelector
from energy_fault_detector.threshold_selectors.quantile_threshold import QuantileThresholdSelector
from energy_fault_detector.threshold_selectors.adaptive_threshold import AdaptiveThresholdSelector

__all__ = [
    'FDRSelector',
    'FbetaSelector',
    'QuantileThresholdSelector',
    'AdaptiveThresholdSelector'
]
