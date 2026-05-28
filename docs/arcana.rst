.. _arcana_docs:

ARCANA: Root cause analysis
===========================

ARCANA (:mod:`energy_fault_detector.root_cause_analysis.arcana`) is a gradient-based root cause
analysis method for autoencoders, described in:

  *Autoencoder-based anomaly root cause analysis for wind turbines.*\
  Energy and AI, 2021. https://doi.org/10.1016/j.egyai.2021.100065

The idea is to find a small correction ``x_bias`` for datapoints with high reconstruction errors that makes the
autoencoder reconstruct "normal" behavior again. Large bias for a feature indicates that this feature contributes
strongly to the detected anomaly.

Using ARCANA via FaultDetector
------------------------------

The :py:meth:`FaultDetector.run_root_cause_analysis <energy_fault_detector.fault_detector.FaultDetector.run_root_cause_analysis>`
method runs ARCANA on a trained model:

.. code-block:: python

   from energy_fault_detector import FaultDetector, Config

   # Configure and fit a FaultDetector model
   config = Config("base_config.yaml")
   fd = FaultDetector(config=config)
   fd.fit(sensor_data=train_data, normal_index=train_normal_index)

   # Run ARCANA on some period of interest
   bias_data, arcana_losses, tracked_bias = fd.run_root_cause_analysis(
       sensor_data=test_data,
       track_losses=True,
       track_bias=True,
   )

   # bias_data: DataFrame with ARCANA bias per timestamp and feature
   # arcana_losses: DataFrame with loss values over iterations (optional)
   # tracked_bias: list of bias snapshots every N iterations (optional)

Interpreting ARCANA results
---------------------------

- ``bias_data`` is a DataFrame of the same shape as the scaled input.
  Each value indicates how much the corresponding feature was "corrected"
  (in model input units) to remove the anomaly.

- Large absolute values in a column indicate that this feature is likely
  responsible for the anomaly.

The helper functions in
:mod:`energy_fault_detector.root_cause_analysis.arcana_utils` provide
convenient aggregations, e.g.:

.. code-block:: python

   from energy_fault_detector.root_cause_analysis.arcana_utils import (
       calculate_mean_arcana_importances,
   )

   importances = calculate_mean_arcana_importances(bias_data)
   print(importances.sort_values(ascending=False).head())
   # returns a Series with mean relative importance per feature

For visualisation helpers, see
:mod:`energy_fault_detector.utils.visualisation`, in particular
``plot_arcana_mean_importances`` and ``plot_arcana_importance_series``.

Full Root Cause Analysis Workflow
---------------------------------
The typical workflow:

    1. Predict a time series of anomaly scores using the FaultDetector.
    2. Extract anomaly events.
    3. Analyze those events with ARCANA (i.e. calculate the ARCANA bias for each event) and convert ARCANA bias data to importances.
    4. Visualise the importances and interpret events.

Step 1: Predict a time series of anomaly scores

.. code-block:: python

    results = detector.predict(sensor_data=test_data)
    scores = results.anomaly_score           # Series[float]
    anomalies = results.predicted_anomalies  # Series[bool]

Step 2: Extract anomaly events
Aggregate consecutive anomalies into events using create_events:

.. code-block:: python
    from energy_fault_detector.utils.analysis import create_events

    event_meta, event_data_list = create_events(
        sensor_data=test_data,
        boolean_information=anomalies,  # True = anomaly
        min_event_length=18  # number of timesteps
    )

    # event_meta: DataFrame with ['start', 'end', 'duration']
    # event_data_list: list of DataFrames, one per event

Notes:
    If there are many events it is often times useful to focus on the events with the highest duration.

Step 3: Analyse events with ARCANA and calculate importances
Run ARCANA per event via FaultDetector.run_root_cause_analysis:

.. code-block:: python
    from energy_fault_detector.root_cause_analysis.arcana_utils import (
        calculate_mean_arcana_importances,
    )

    arcana_mean_importances_per_event = []
    arcana_losses_per_event = []  # Loss tracking is optional and only useful for evaluations of ARCANA parameter settings.

    for event_data in event_data_list:
        bias, arcana_losses, tracked_bias = detector.run_root_cause_analysis(
            sensor_data=event_data,
            track_losses=True,
            track_bias=False,
        )
        arcana_losses_per_event.append(arcana_losses)  # Optional if ARCANA parameter optimization is planned

        # Step 4: convert bias to importances
        mean_importances = calculate_mean_arcana_importances(bias_data=bias)
        arcana_mean_importances_per_event.append(mean_importances)

Step 4: Visualise ARCANA importances

.. code-block:: python
    from energy_fault_detector.utils.visualisation import (
        plot_arcana_mean_importances
    )
    import matplotlib.pyplot as plt

    # Bar plot of mean importances for one event
    fig, ax = plot_arcana_mean_importances(
        importances=mean_importances,
        top_n_features=10
    )
