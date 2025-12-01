.. _configuration_guide:

Configuration
================================
This page explains how to configure training, prediction, and optional root cause analysis (ARCANA).

.. contents:: Table of Contents
   :depth: 3
   :local:

Quick start: minimal configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A minimal configuration that clips outliers (optional), imputes missing values, and scales features:

.. include:: basic_config.yaml
   :literal:

This setup:

- Applies DataClipper if specified.
- Builds a DataPreprocessor with:

  - ColumnSelector that drops columns with more than 20% NaNs.
  - LowUniqueValueFilter that removes constant/binary features by default.
  - SimpleImputer (mean) and StandardScaler (always added unless overridden).

- Trains a default autoencoder (with provided architecture, otherwise default values), with an RMSE anomaly score and a
  quantile threshold selector.
- Runs ARCANA with provided parameters, when calling :py:obj:`FaultDetector.predict(data, root_cause_analysis=True) <energy_fault_detector.fault_detector.FaultDetector.predict>`,
  otherwise, default ARCANA parameters are used (see :py:obj:`ARCANA docs <energy_fault_detector.root_cause_analysis.arcana.Arcana>`).

Detailed configuration
^^^^^^^^^^^^^^^^^^^^^^
Below is a more thorough configuration. It shows how to specify several preprocessing steps and more model parameters.

.. include:: advanced_config.yaml
   :literal:

For detailed documentation of the data preprocessor pipeline, refer to :py:obj:`DataPreprocessor <energy_fault_detector.data_preprocessing.data_preprocessor.DataPreprocessor>`.

Notes:

- :py:obj:`DataClipper <energy_fault_detector.data_preprocessing.data_clipper.DataClipper>` supports ``features_to_exclude`` or ``features_to_clip`` to control which features are clipped.
- :py:obj:`ColumnSelector <energy_fault_detector.data_preprocessing.column_selector.ColumnSelector>` supports ``features_to_exclude`` and ``features_to_select``. If ``features_to_select`` is provided, only those columns are kept.
- :py:obj:`AngleTransformer <energy_fault_detector.data_preprocessing.angle_transformer.AngleTransformer>` will convert provided angle features to sin/cos pairs.
- :py:obj:`CounterDiffTransformer <energy_fault_detector.data_preprocessing.counter_diff_transformer.CounterDiffTransformer>` normalizes counter-type features to differences or rates, with options for rollover and data gap handling.



Root cause analysis (ARCANA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``root_cause_analysis`` is provided, ARCANA will attempt to attribute anomalies to underlying biases using the
provided settings. For detailed documentation of ARCANA refer to the :py:obj:`ARCANA docs <energy_fault_detector.root_cause_analysis.arcana.Arcana>`.

Prediction options
^^^^^^^^^^^^^^^^^^
Under ``predict``, you can set:

- ``criticality.max_criticality``: cap the calculated criticality (anomaly counter) to this value.

Old params data preprocessing configuration (for older versions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Older configurations use params under ``train.data_preprocessor.params``.
These remain supported but are deprecated in favor of steps mode.

.. include:: old_config.yaml
   :literal:
