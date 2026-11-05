.. _criticality:

Criticality
===========

The criticality is a cumulative counter derived from the point-wise anomaly predictions of a model. It is used to
convert raw per-timestamp anomaly decisions into a *persistence-aware* signal, so that alarms are raised only when
anomalous behaviour is sustained over time. This avoids raising alarms for isolated anomalies caused by noise or by
potential data-quality issues.

The reference implementation is
:func:`calculate_criticality <energy_fault_detector.utils.analysis.calculate_criticality>` in
:mod:`energy_fault_detector.utils.analysis`. The same calculation is available directly on a prediction result via
:meth:`FaultDetectionResult.criticality <energy_fault_detector.core.fault_detection_result.FaultDetectionResult.criticality>`.

How it works
------------

For every timestamp the counter is updated by one of three rules, depending on the predicted anomaly and on the
operating status (``normal_idx``):

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Situation
     - Change
     - Reason
   * - Anomaly predicted, status normal
     - ``+1``
     - Sustained anomalous behaviour during normal operation increases criticality.
   * - No anomaly predicted, status normal
     - ``-1``
     - Normal behaviour decreases criticality again.
   * - Any prediction, status **not** normal
     - ``0``
     - Timestamps with non-normal status (maintenance, shutdown, ...) are ignored.

The counter is bounded below by ``0`` and above by ``max_criticality``. If ``normal_idx`` is not provided, all
timestamps are treated as normal operation.

.. note::
   The status information (``normal_idx``) can be used to ignore alarms which otherwise inflate the
   counter during periods in which anomalous behavior is not meaningful (e.g. during maintenance)

Parameters
----------

.. code-block:: python

   from energy_fault_detector.utils.analysis import calculate_criticality

   criticality = calculate_criticality(
       anomalies,                 # pd.Series[bool] over time
       normal_idx=None,           # optional pd.Series[bool], True = normal operation
       init_criticality=0,        # starting value of the counter
       max_criticality=1000,      # upper bound of the counter
   )

- ``anomalies``: boolean :class:`pandas.Series` of predicted anomalies, indexed by timestamp or by a ``MultiIndex``
  such as ``(asset_id, timestamp)``.
- ``normal_idx``: optional boolean :class:`pandas.Series` with normal-operation status. It is reindexed to
  ``anomalies``; missing values are treated as normal (``True``).
- ``init_criticality``: initial value of the counter. Used if persistence is desired across multiple calls to the function (e.g. when processing a long time series in separatechunks).
- ``max_criticality``: maximum value the counter can reach.

The function returns a :class:`pandas.Series` with the criticality value per timestamp and the same index as the
input. For a ``MultiIndex``, the criticality is computed independently per group (e.g. per asset).

On a prediction result you can call the method directly:

.. code-block:: python

   results = fault_detector.predict(sensor_data=test_sensor_data)

   criticality = results.criticality()                                   # all timestamps treated as normal
   criticality = results.criticality(normal_idx=test_status, max_criticality=144)

Interpreting criticality: raising alarms
----------------------------------------

Criticality is interpreted relative to an application-specific **criticality threshold**:

- as long as ``criticality < threshold``, no alarm is raised,
- once ``criticality >= threshold``, an alarm is raised (the anomaly is treated as significant).

Because the counter rises by one per anomalous timestamp, the threshold can be read as the minimum number of consecutive
anomalous timestamps required to raise an alarm. With a fixed sampling frequency this corresponds directly to a time
window:

.. code-block:: text

   alarm_time_window = criticality_threshold * time_frequency

For example, with a time frequency of ``10 minutes`` and a criticality threshold of ``6``:

.. math::

   6 \times 10\,\text{min} = 60\,\text{min}

so at least **one hour** of anomalous behaviour is needed to trigger an alarm.

The threshold therefore encodes two things at once:

- **How long** anomalous behaviour must persist before it is considered actionable.
- **How tolerant** the application is to occasional anomalies. A low threshold reacts quickly but will also alarm on
  isolated random anomalies; a high threshold is slower but more robust.

Choosing a threshold
^^^^^^^^^^^^^^^^^^^^

The correct threshold depends on the application context and on user expectations; there is no single default.
Consider:

- **Time frequency** of the data and the desired alarm latency. The same physical time window maps to different
  thresholds for different sampling rates.
- **Expected random anomalies** from noise or data-quality problems. These can be estimated from real anomalies used for validation
  and or from validation data with normal status: run the model on validation data known to be normal, 
  observe how the criticality behaves, and choose a threshold that random fluctuations do not reach. 
  The more (expected) spurious anomalies per unit of time, the higher the threshold should be.
- **Severity and cost** of a missed fault versus a false alarm.

A simple procedure:

#. Compute the criticality on validation data that has normal status.
#. Observe the maximum criticality reached by random anomalies (noise / data quality).
#. Set the threshold well above that level, and additionally require the desired minimum persistence time.

Choosing ``max_criticality``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``max_criticality`` caps the counter and therefore controls how long it takes for an active alarm to be reset
after the system returns to normal behaviour. Once the counter has saturated at ``max_criticality``, it must be
decremented back towards the threshold before the alarm is closed. Because the counter decreases by one per normal
timestamp, the time to reset from saturation is at most:

.. code-block:: text

   max_reset_time = (max_criticality - criticality_threshold) * time_frequency

Continuing the example above (``10 minutes`` resolution) with a threshold of ``6`` and ``max_criticality = 12``:

.. math::

   (12 - 6) \times 10\,\text{min} = 60\,\text{min}

so at most **one hour** of consecutive normal behaviour is needed for the counter to fall from saturation back below the
threshold and for the alarm to close.

Choose ``max_criticality`` relative to your threshold so that the time to reset matches what the application can
tolerate. Setting it so that the reset time is at most **1 hour** or **1 day** is often a good default.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Time frequency
     - Threshold
     - Alarm needs
     - ``max_criticality``
     - Alarm closes after
   * - 10 min
     - 6
     - 1 h anomalous
     - 12
     - 1 h normal (from saturation)
   * - 10 min
     - 6
     - 1 h anomalous
     - 150
     - ~1 day normal (from saturation)
   * - 1 min
     - 60
     - 1 h anomalous
     - 1500
     - ~1 day normal (from saturation)

.. note::
   Both the threshold and ``max_criticality`` are expressed in *number of timestamps*, not in physical time. When you
   change the time frequency of your data, recompute them so that the intended physical time windows are preserved.

Configuration
-------------

When using the :class:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>` framework, the maximum
criticality can be set in the YAML configuration:

.. code-block:: yaml

   predict:
     criticality:
       max_criticality: 144   # default: one day at 10 min resolution

The criticality *threshold* is not part of this configuration: it is an application decision that you apply to the
returned criticality series, e.g.:

.. code-block:: python

   criticality = results.criticality(normal_idx=test_status, max_criticality=config.max_criticality)
   alarms = criticality >= 6

See :ref:`configuration_guide` for the full configuration reference.

Visualisation
-------------

Criticality can be plotted together with the anomaly score using
:func:`plot_score_with_threshold <energy_fault_detector.utils.visualisation.plot_score_with_threshold>`:

.. code-block:: python

   from energy_fault_detector.utils.visualisation import plot_score_with_threshold

   plot_score_with_threshold(
       model=fault_detector,
       data=test_sensor_data,
       normal_index=test_status,
       show_criticality=True,
       max_criticality=144,
   )

Relation to evaluation (CARE score)
-----------------------------------

The same criticality concept is used by the
:class:`CAREScore <energy_fault_detector.evaluation.care_score.CAREScore>` to decide whether a complete event is
treated as detected: an event is classified as anomalous when its maximum criticality reaches
``criticality_threshold``. See :doc:`care2compare_guide` and :doc:`care2compare_faq`. As with alarm raising, this
threshold should be chosen based on the application context and time frequency, and tuned for the dataset at hand.
