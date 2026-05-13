Usage examples
================================
To see interactive demonstrations of the energy fault detection package,
refer to the example notebooks in the repository's notebooks folder.

.. contents:: Table of Contents
    :depth: 3
    :local:

Expected input data format
^^^^^^^^^^^^^^^^^^^^^^^^^^

Most examples in this documentation assume that your data is already loaded into
pandas objects with the following structure:

* ``sensor_data``: ``pd.DataFrame`` in **wide format**

  * index:

    - either a unique, sorted ``DatetimeIndex`` (no duplicate timestamps), or
    - a ``MultiIndex`` with one datetime-like level and one non-datetime grouping
      level (e.g. ``(asset_id, timestamp)``).

      **MultiIndex limitations:**

      - Sequence-based models require a single-device ``DatetimeIndex``.
        Select one group before passing to ``fit``/``predict``.
      - The `quick-fault-detctor` CLI expects single-device CSV files.

  * columns: one column per sensor / feature (numeric, or castable to numeric)

* ``normal_index`` (optional): ``pd.Series``

  * index: same as ``sensor_data.index``
  * values: boolean – ``True`` indicates normal operation, ``False`` indicates
    non‑normal operation (faults, maintenance, curtailment, etc.)

If you do not provide ``normal_index``, the models assume that all samples in
``sensor_data`` represent normal behaviour. In that case you cannot use
label-based threshold selectors such as :class:`FbetaSelector` or :class:`FDRSelector`,
but you can still use the quantile-based (default) or adaptive threshold.

For sequence-based models, a :class:`pandas.DatetimeIndex` or a compatible
``MultiIndex`` is required as described above; windows are built per group
(when a grouping level is present) and then concatenated.

Minimal end-to-end example
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example shows the full workflow in a few lines: load data,
create a configuration, train a model and predict.

.. code-block:: python

    import pandas as pd
    from energy_fault_detector import FaultDetector, Config
    from energy_fault_detector.config import generate_quickstart_config

    # 1. Load your data
    df = pd.read_csv("my_data.csv", parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()  # ensure sorted
    # Keep only numeric sensor columns
    sensor_data = df[["power", "wind_speed", "pitch"]]  # adapt to your dataset

    # Boolean normal_index: True = normal operation
    # This is optional; if omitted, all data is treated as normal
    normal_index = df["status"] == "normal"

    # 2. Generate and load a base config
    generate_quickstart_config(output_path="base_config.yaml")
    config = Config("base_config.yaml")

    # 3. Train a normal-behavior model
    fault_detector = FaultDetector(config=config, model_directory="fault_detector_model")
    model_meta = fault_detector.fit(sensor_data=sensor_data, normal_index=normal_index)

    # 4. Predict anomalies
    results = fault_detector.predict(sensor_data=sensor_data)

    anomalies = results.predicted_anomalies    # pd.Series[bool]
    scores = results.anomaly_score             # pd.Series[float]
    recon = results.reconstruction             # pd.DataFrame
    recon_error = results.recon_error          # pd.DataFrame

For more configuration options and details (e.g. updating at runtime and listing available model classes),
see :ref:`configuration_guide`.


Standard `FaultDetector` usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The main interface for the `energy-fault-detector` package is the :py:obj:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>` class, which
needs a configuration object :py:obj:`Config <energy_fault_detector.config.config.Config>`.

To create a new :py:obj:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>` model,
create a configuration, as described below in the :ref:`configuration` section, and run:

.. code-block:: python

    from energy_fault_detector import FaultDetector, Config

    config = Config('path/to/your/configuration/file.yaml')
    fault_detector = FaultDetector(config=config, model_directory='model_directory')

To train new models, you need to provide the input data and call the :py:obj:`FaultDetector.fit <energy_fault_detector.fault_detector.FaultDetector.fit>` method:

.. code-block:: python

    # get data from database / csv / API ...
    sensor_data = ...  # a pandas DataFrame with timestamp as index and numerical sensor values as columns
    normal_index = ...  # a pandas Series with timestamp as index and booleans indicating normal behaviour
    # NOTE: The normal_index is optional; it is used to select training data for the autoencoder.
    # If not provided, we assume all data represents normal behaviour.
    # If you do not have any labels, you cannot use the F-beta-score- and FDR-based thresholds.
    # In that case, use the quantile-based threshold (default) or AdaptiveThreshold.

    # If you do not use the models for time series, the index can also be a standard RangeIndex,
    # as long as the sensor_data DataFrame and the normal_index Series share the same index.

    model_data = fault_detector.fit(sensor_data=sensor_data, normal_index=normal_index, save_models=True)

    # to save model manually:
    # fault_detector.save_models('model_name')  # model_name is optional

The trained models are saved locally in the provided ``model_directory``. The :py:obj:`FaultDetector.fit <energy_fault_detector.fault_detector.FaultDetector.fit>` method returns a
:py:obj:`ModelMetadata <energy_fault_detector.core.fault_detection_result.ModelMetadata>` object with
the model metadata such as the model date and the model path.

To predict using the trained model, use the :py:obj:`FaultDetector.predict <energy_fault_detector.fault_detector.FaultDetector.predict>` method:

.. code-block:: python

    results = fault_detector.predict(sensor_data=test_sensor_data)

The result is a :py:obj:`FaultDetectionResult <energy_fault_detector.core.fault_detection_result.FaultDetectionResult>` object
with the following information:

* predicted_anomalies: pandas Series with the predicted anomalies (bool).
* reconstruction: pandas DataFrame with reconstruction of the sensor data with timestamp as index.
* deviations: pandas DataFrame with reconstruction errors.
* anomaly_score: pandas Series with anomaly scores for each timestamp.
* bias_data: pandas DataFrame with ARCANA results with timestamp as index. None if ARCANA was not run.
* arcana_losses: pandas DataFrame containing recorded values for all losses in ARCANA. None if ARCANA was not run.
* tracked_bias: List of pandas DataFrames. None if ARCANA was not run.

You can also create a :py:obj:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>` object and load
trained models using the :py:obj:`FaultDetector.load <energy_fault_detector.core.fault_detection_model.FaultDetectionModel.load>` class method.
In this case, you do not need to provide a ``model_path`` in the :py:obj:`predict <energy_fault_detector.fault_detector.FaultDetector.predict>` method.

.. code-block:: python

    from energy_fault_detector.fault_detector import FaultDetector

    fault_detector = FaultDetector.load('path_to_trained_models')

    # get data from database / csv / API ...
    sensor_data = ...
    results = fault_detector.predict(sensor_data=sensor_data)



Quick fault detection (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a one-command experiment on a CSV file, you can use the
:ref:`quick_fault_detector_cli` command-line interface:

.. code-block:: bash

    quick_fault_detector path/to/data.csv --options path/to/options.yaml

This runs the full pipeline (training, prediction, event aggregation and ARCANA) and produces plots and CSV outputs.

For details, see :doc:`quick_fault_detection`.

Sequence-based models
^^^^^^^^^^^^^^^^^^^^^

Besides the dense :py:class:`MultilayerAutoencoder <energy_fault_detector.autoencoders.multilayer_autoencoder.MultilayerAutoencoder>`,
the package also provides sequence-based autoencoders such as:

- :py:class:`LSTMSeq2OneAutoencoder <energy_fault_detector.autoencoders.lstm_seq2one_autoencoder.LSTMSeq2OneAutoencoder>`
- :py:class:`BidirectionalLSTMSeq2OneAutoencoder <energy_fault_detector.autoencoders.bidirectional_lstm_seq2one_autoencoder.BidirectionalLSTMSeq2OneAutoencoder>`
- :py:class:`CNNSeq2OneAutoencoder <energy_fault_detector.autoencoders.cnn_seq2one_autoencoder.CNNSeq2OneAutoencoder>`

These models operate on **windows** of time-series data and reconstruct the **last timestep** in each window.
They require:

- either a :class:`pandas.DatetimeIndex` on ``sensor_data`` or a MultiIndex with
  one datetime-like level and one non-datetime grouping level (e.g.
  ``(asset_id, timestamp)``); windows are built per group and concatenated,
- a ``sequence_builder`` section in the config (with ``sequence_length``, ``stride``, ``ts_freq``, etc.).

A full description and examples are given in :doc:`sequence_models`.


.. _configuration:

Configuration
^^^^^^^^^^^^^

The behaviour of the :py:class:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>`
is controlled by a YAML configuration, parsed by :py:class:`Config <energy_fault_detector.config.config.Config>`.
The config typically has:

- a ``train`` section:
  - ``data_preprocessor``: preprocessing pipeline (imputation, scaling, etc.),
  - ``autoencoder``: model type and training parameters,
  - ``anomaly_score``: how reconstruction errors are turned into scores,
  - ``threshold_selector``: how a score threshold is chosen,
  - ``data_splitter``: how training/validation sets are split,
  - optional ``data_clipping``: outlier clipping on training data only.
- an optional ``root_cause_analysis`` section for ARCANA.
- an optional ``predict`` section (e.g. criticality settings).

For most users, the easiest way to create a valid configuration is via :func:`generate_quickstart_config <energy_fault_detector.config.quickstart_config.generate_quickstart_config>`:

.. code-block:: python

   from energy_fault_detector.config.quickstart_config import generate_quickstart_config
   from energy_fault_detector.config import Config

   # Create a minimal, valid config file
   generate_quickstart_config(output_path="base_config.yaml")

   # Load and use it
   cfg = Config("base_config.yaml")
   fd = FaultDetector(config=cfg)

If you prefer to write the YAML yourself or need more control, see the :ref:`Configuration guide <configuration_guide>`
for a full reference and examples.


Root cause analysis with ARCANA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`FaultDetector.run_root_cause_analysis <energy_fault_detector.fault_detector.FaultDetector.run_root_cause_analysis>`
method runs the ARCANA algorithm on a trained model and returns per-feature bias information.

For a dedicated explanation and examples, see :ref:`arcana_docs`.

Evaluation and CARE-Score
^^^^^^^^^^^^^^^^^^^^^^^^^^

For evaluation, the package provides:

- :func:`energy_fault_detector.utils.analysis.create_events` to aggregate
  point-wise anomaly predictions into contiguous anomaly events, and
- :class:`energy_fault_detector.evaluation.care_score.CAREScore` to compute
  the CARE-Score for early fault detection (Coverage, Accuracy, Reliability,
  Earliness), as introduced in the CARE2Compare paper.

For now, we recommend using the example notebooks for a full
walkthrough of the evaluation workflow (event creation, criticality, CARE-Score
on benchmark datasets such as CARE2Compare and PreDist). A higher-level
evaluation helper/script may be added in a future version.

Creating new model classes
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can extend the framework by creating new model classes based on the templates in the
:py:obj:`core <energy_fault_detector.core>` module and registering the new classes.
Examples are shown in the notebook ``Example - Create new model classes.ipynb``.

Creating your own pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to create your own energy fault detection pipeline with the building blocks of this package,
you can import the data preprocessor, autoencoder, anomaly score and threshold selection classes as follows:

.. code-block:: python

    from energy_fault_detector.data_preprocessing import DataPreprocessor, DataClipper
    from energy_fault_detector.autoencoders import MultilayerAutoencoder
    from energy_fault_detector.anomaly_scores import MahalanobisScore
    from energy_fault_detector.threshold_selectors import FbetaSelector

This allows you to add additional steps or use different data preprocessing pipelines.

An example training pipeline (similar to the :py:obj:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>` class)
would be:

.. code-block:: python

    x = ...  # i.e. sensor data
    y = ...  # normal behaviour indicator

    x_normal = x[y]
    # fit data preprocessor on normal data
    data_preprocessor = DataPreprocessor(...)
    x_normal_prepped = data_preprocessor.fit_transform(x_normal)

    # fit autoencoder on normal data
    ae = MultilayerAutoencoder(...)
    ae.fit(x_normal_prepped)

    # create and fit score
    anomaly_score = MahalanobisScore(...)
    x_prepped = data_preprocessor.transform(x)

    # fit on normal data
    recon_error_normal = ae.get_reconstruction_error(x_normal_prepped)
    anomaly_score.fit(recon_error_normal)
    # get scores of all data points
    recon_error = ae.get_reconstruction_error(x_prepped)
    scores = anomaly_score.transform(recon_error)

    # set the threshold and get predictions to evaluate
    threshold_selector = FbetaSelector(beta=1.0)  # sets optimal threshold based on F1 score
    threshold_selector.fit(scores, y)
    # NOTE: the fit-method of the AdaptiveThreshold has slightly different arguments!
    anomalies = threshold_selector.predict(scores)

And the inference:

.. code-block:: python

    x = ...

    x_prepped = data_preprocessor.transform(x)
    x_recon = ae.predict(x_prepped)  # reconstruction
    x_recon_error = ae.get_reconstruction_error(x_prepped)
    scores = anomaly_score.transform(x_recon_error)
    anomalies = threshold_selector.predict(scores)  # boolean series indicating anomaly detected
