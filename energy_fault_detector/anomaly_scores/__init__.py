"""Anomaly scores that reduce per-feature reconstruction errors to one score per sample.

After an autoencoder reconstructs its input, an :class:`~energy_fault_detector.core.AnomalyScore`
aggregates the per-feature reconstruction errors into a single scalar per sample.
Samples with high scores are candidate anomalies.

Each scorer subclasses :class:`~energy_fault_detector.core.AnomalyScore` and is selected
via the ``train.anomaly_score`` config key.
"""

from energy_fault_detector.anomaly_scores.mahalanobis_score import MahalanobisScore
from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore

__all__ = [
    "MahalanobisScore",
    "RMSEScore"
]
