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
