.. _arcana_docs:

ARCANA: Root cause analysis
===========================

ARCANA (:mod:`energy_fault_detector.root_cause_analysis.arcana`) is a gradient-based root cause
analysis method for autoencoders, described in:

  *Autoencoder-based anomaly root cause analysis for wind turbines.*\
  Energy and AI, 2021. https://doi.org/10.1016/j.egyai.2021.100065

The idea is to find a small correction ``x_bias`` to the input data that makes the autoencoder
reconstruct "normal" behavior again. Large bias for a feature indicates that this feature contributes
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
