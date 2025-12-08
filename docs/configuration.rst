.. _configuration_guide:

Configuration
================================
This page explains how to configure training, prediction, and optional root cause analysis (ARCANA).

.. contents:: Table of Contents
   :depth: 3
   :local:

Quick start: minimal configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A minimal configuration that clips outliers, imputes missing values, and scales features:

.. include:: basic_config.yaml
   :literal:

This setup:

- Applies DataClipper if specified.
- Builds a DataPreprocessor with:

  - ColumnSelector that drops columns with more than 20% NaNs (configurable).
  - LowUniqueValueFilter that removes constant features by default (configurable).
  - SimpleImputer (mean) and a scaler (StandardScaler by default). If you do not add an imputer/scaler explicitly,
    the pipeline ensures mean-imputation and StandardScaler are added.

- Trains a default autoencoder (with provided architecture, otherwise default values), with an RMSE anomaly score and a
  quantile threshold selector.
- Runs ARCANA with provided parameters when calling :py:obj:`FaultDetector.predict(..., root_cause_analysis=True) <energy_fault_detector.fault_detector.FaultDetector.predict>`.
  If not provided, default ARCANA parameters are used (see :py:obj:`ARCANA docs <energy_fault_detector.root_cause_analysis.arcana.Arcana>`).

If you leave out the data_preprocessor configuration (i.e., ``data_preprocessor: {}``), a default preprocessing pipeline
is generated, which drops constant features, features where >5% of the data is missing, imputes remaining missing values
with the mean value and scales the data to zero mean and unit standard deviation.

Detailed configuration
^^^^^^^^^^^^^^^^^^^^^^
Below is a more thorough configuration. It shows how to specify preprocessing steps and more model parameters.

.. include:: advanced_config.yaml
   :literal:

DataPreprocessor specification
""""""""""""""""""""""""""""""
A steps-based preprocessing pipeline can be configured under ``train.data_preprocessor.steps``. Each step is a dict
with the following keys:

- ``name`` (str): the registered step name (see table below).
- ``enabled`` (bool, optional): default ``True``; set to ``False`` to skip a step.
- ``params`` (dict, optional): constructor arguments for the step.
- ``step_name`` (str, optional): custom key for the sklearn pipeline; useful if a step is repeated.

Allowed step names and aliases:

+-------------------------+-----------------------------------------------+------------------------------------------------+
| Step name               | Purpose                                       | Aliases                                        |
+=========================+===============================================+================================================+
| column_selector         | Drop columns with too many NaNs               | \-                                             |
+-------------------------+-----------------------------------------------+------------------------------------------------+
| low_unique_value_filter | Drop columns with low variance/many zeros     | \-                                             |
+-------------------------+-----------------------------------------------+------------------------------------------------+
| angle_transformer       | Convert angles to sin/cos pairs               | angle_transform                                |
+-------------------------+-----------------------------------------------+------------------------------------------------+
| counter_diff_transformer| Convert counters to differences/rates         | counter_diff, counter_diff_transform           |
+-------------------------+-----------------------------------------------+------------------------------------------------+
| simple_imputer          | Impute missing values                         | imputer                                        |
+-------------------------+-----------------------------------------------+------------------------------------------------+
| standard_scaler         | Standardize features (z-score)                | standardize, standardscaler, standard          |
+-------------------------+-----------------------------------------------+------------------------------------------------+
| minmax_scaler           | Scale to [0, 1]                               | minmax                                         |
+-------------------------+-----------------------------------------------+------------------------------------------------+
| duplicate_to_nan        | Replace consecutive duplicate values with NaN | duplicate_value_to_nan, duplicate_values_to_nan|
+-------------------------+-----------------------------------------------+------------------------------------------------+

For detailed documentation of the data preprocessor pipeline, refer to the
:py:obj:`DataPreprocessor <energy_fault_detector.data_preprocessing.data_preprocessor.DataPreprocessor>` docs.

Other training configuration sections
"""""""""""""""""""""""""""""""""""""

- Data clipping:
  :py:obj:`DataClipper <energy_fault_detector.data_preprocessing.data_clipper.DataClipper>` supports
  ``features_to_exclude`` and ``features_to_clip`` for fine-grained control.


- Data splitter (``train.data_splitter``):

  - ``type``: one of ``BlockDataSplitter`` (aliases: ``blocks``, ``DataSplitter``), or ``sklearn`` (alias ``train_test_split``).
  - For sklearn: ``validation_split`` (float in (0, 1)) and ``shuffle`` (bool).
  - For :py:obj:`BlockDataSplitter <energy_fault_detector.data_splitting.data_splitter.BlockDataSplitter>`: ``train_block_size`` and ``val_block_size``.
  - Early stopping guard: if ``train.autoencoder.params.early_stopping`` is true, you must either set a
    valid ``validation_split`` in (0, 1), or use :py:obj:`BlockDataSplitter <energy_fault_detector.data_splitting.data_splitter.BlockDataSplitter>`
    with a positive ``val_block_size``.


- Autoencoder (``train.autoencoder``):

  - ``name``: class name in the registry.
  - ``params``: architecture and training args (e.g., ``layers``, ``epochs``, ``learning_rate``, ``early_stopping``).
    Refer to the autoencoder class docs (:py:obj:`autoencoders <energy_fault_detector.autoencoders>`) for specific params and their defaults.

- Anomaly score (``train.anomaly_score``):

  - ``name``: score name (e.g., ``rmse``, ``mahalanobis``).
  - ``params``: score-specific parameters. Refer to the :py:obj:`anomaly_scores <energy_fault_detector.anomaly_scores>` docs.

- Threshold selector (``train.threshold_selector``):

  - ``name``: e.g., ``quantile``, ``fbeta``, etc.
  - ``fit_on_val``: fit the threshold on validation only.
  - ``params``: selector-specific parameters (e.g., ``quantile`` for the quantile selector).
    See the :py:obj:`threshold_selectors <energy_fault_detector.threshold_selectors>` docs for more info on the settings.

Prediction options
^^^^^^^^^^^^^^^^^^
Under ``predict``, you can set:

- ``criticality.max_criticality``: cap the calculated criticality (anomaly counter) to this value.


Root cause analysis (ARCANA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If ``root_cause_analysis`` is provided, ARCANA will attempt to attribute anomalies to specific features using the
provided settings. If not provided, default settings are used. For detailed documentation refer to
:py:obj:`ARCANA docs <energy_fault_detector.root_cause_analysis.arcana.Arcana>`.


Old params data preprocessing configuration (for older versions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Older configurations use params under ``train.data_preprocessor.params``.
These remain supported but are deprecated in favor of steps mode.
When both ``steps`` and legacy params are present, ``steps`` take precedence and legacy params are ignored with a warning.

.. include:: old_config.yaml
   :literal:
