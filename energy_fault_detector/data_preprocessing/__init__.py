"""Configurable preprocessing pipeline and outlier clipping for sensor data.

The :class:`~energy_fault_detector.data_preprocessing.data_preprocessor.DataPreprocessor` is an
sklearn-style ``Pipeline`` assembled from configurable :class:`~energy_fault_detector.core.DataTransformer`
steps (see ``DataPreprocessor.STEP_REGISTRY``).
The pipeline is configured via the ``train.data_preprocessor.steps`` config key.

"""

from energy_fault_detector.data_preprocessing.data_preprocessor import DataPreprocessor
from energy_fault_detector.data_preprocessing.data_clipper import DataClipper
