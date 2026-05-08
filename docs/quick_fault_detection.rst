.. _quick_fault_detector_cli:
Quick fault detection (CLI)
===========================

The ``quick_fault_detector`` command-line tool runs the complete fault detection pipeline on a CSV file:

- load and split data (train / test),
- configure a normal-behavior model (optionally optimize hyperparameters),
- train the autoencoder and threshold,
- predict anomalies on test data,
- aggregate anomalies into events,
- optionally run ARCANA on each event,
- save plots and CSV results to a directory.

Basic usage
-----------
.. code-block:: bash

   quick_fault_detector path/to/train_and_test.csv

Expected CSV format
-------------------

Internally, the CLI converts your CSV into:

- a training ``sensor_data`` DataFrame (numeric, wide format), and
- a ``normal_index`` Series indicating normal behaviour during training, and
- a test ``sensor_data`` DataFrame used for prediction.

This is done by :mod:`energy_fault_detector.quick_fault_detection.data_loading`.

Internally, the timestamp column is converted to a ``DatetimeIndex`` on the
training and test DataFrames before passing them to the models.

Your CSV should contain:

- **A timestamp column** (optional but recommended), e.g. ``time_stamp``

  - Parsed to a ``DatetimeIndex`` if ``time_column_name`` is provided in the options.
  - Timestamps are sorted and duplicate rows are dropped upstream by the user if needed.

- **An optional train/test column**, e.g. ``train_test``

  - If ``train_test_column_name`` is given, it is converted to a boolean mask
    (True = training, False = test).
  - If the column is not already boolean, the mapping in ``train_test_mapping`` is applied.

- **An optional status column**, e.g. ``status`` or ``status_type_id``

  - If ``status_data_column_name`` is given, it is converted to a boolean mask
    (True = normal, False = non‑normal / faulty / maintenance).
  - If the column is not boolean, the mapping in ``status_mapping`` is applied.
  - This becomes the ``normal_index`` used for training.
  - If no status column is provided, **all training samples are assumed normal**, and a warning is logged.

- **All other columns** are treated as sensor features:

  - Numeric columns are used directly.
  - Non‑numeric columns that can be cast to numeric are converted (e.g. strings of numbers).
  - Remaining non‑numeric columns are ignored for the model input.

Conceptually:

- Training data: rows where the train/test mask is True (or all rows if no split is provided).
- Test data: rows where the train/test mask is False, plus any separate test file provided via
  ``csv_test_data_path`` (these are concatenated).

Example layout::

    time_stamp,train_test,status,power,wind_speed,pitch,asset_id
    2024-01-01 00:00:00,train,production,  500,  7.3,  2.0, 1234
    2024-01-01 00:10:00,train,production,  520,  7.5,  2.1, 1234
    2024-01-02 00:00:00,prediction,error,  100,  3.0, 15.0, 1234

With an options YAML like::

    time_column_name: "time_stamp"
    train_test_column_name: "train_test"
    train_test_mapping:
      train: true
      prediction: false
    status_data_column_name: "status"
    status_mapping:
      production: true
      service: false
      error: false


Options
-------
You can pass an options YAML file to control how data is interpreted:

.. code-block:: bash

   quick_fault_detector path/to/data.csv --options path/to/options.yaml

The options are defined in :mod:`energy_fault_detector.main` and correspond to the :class:`Options <energy_fault_detector.main.Options>`
dataclass. An example options file:

.. code-block:: yaml

   csv_test_data_path: "path/to/separate_test.csv"
   train_test_column_name: "train_test"      # True = training data
   train_test_mapping:                       # mapping if train test column is not boolean
     train: true
     prediction: false
   time_column_name: "time_stamp"
   status_data_column_name: "status"        # True = normal behaviour
   status_mapping:                          # mapping if status column is not boolean
     production: true
     service: false
     error: false
   status_label_confidence_percentage: 0.95
   min_anomaly_length: 18
   features_to_exclude:
     - do_not_use_this_feature
   angle_features:
     - wind_direction
   automatic_optimization: true
   enable_debug_plots: false

The underlying helper functions are implemented in:

- :mod:`energy_fault_detector.quick_fault_detection.data_loading`
- :mod:`energy_fault_detector.quick_fault_detection.configuration`
- :mod:`energy_fault_detector.quick_fault_detection.quick_fault_detector`

Output
------
The CLI writes:

- a combined results figure ``results.png``,
- CSV files for :class:`FaultDetectionResult <energy_fault_detector.core.fault_detection_result.FaultDetectionResult>`
  via ``FaultDetectionResult.save()``,
- an ``events.csv`` file with aggregated anomaly events.

See the notebook
``Example - Quick Fault Detection.ipynb`` for an interactive walkthrough.
