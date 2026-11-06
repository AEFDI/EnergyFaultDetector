.. _criticality:

Criticality and alarms
======================

The criticality is a cumulative counter derived from the point-wise anomaly predictions of a model. It is used to
convert raw per-timestamp anomaly decisions into a *persistence-aware* signal, so that alarms are raised only when
anomalous behaviour is sustained over time. This avoids raising unnecessary alarms for isolated anomalies caused by
noise or by potential data-quality issues.

The package provides two building blocks in :mod:`energy_fault_detector.utils.analysis`:

- :func:`calculate_criticality <energy_fault_detector.utils.analysis.calculate_criticality>` computes the raw
  criticality counter.
- :func:`calculate_alarms <energy_fault_detector.utils.analysis.calculate_alarms>` builds on it to derive
  **alarm events** (start, end, duration) from the criticality.

Both are exposed on a prediction result:

- :meth:`FaultDetectionResult.criticality <energy_fault_detector.core.fault_detection_result.FaultDetectionResult.criticality>`
  returns only the criticality series (the basic counter, with an explicit ``max_criticality``).
- :meth:`FaultDetectionResult.criticality_alarms <energy_fault_detector.core.fault_detection_result.FaultDetectionResult.criticality_alarms>`
  returns both the alarm events and the criticality series, with the threshold and reset window derived from the
  time frequency.

How the criticality works
-------------------------

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
   The status information (normal_idx) can be used to ignore alarms which otherwise inflate the counter during periods in which anomalous 
   behavior is not meaningful (e.g. during maintenance).

Deriving alarms with ``calculate_alarms``
-----------------------------------------

:func:`calculate_alarms <energy_fault_detector.utils.analysis.calculate_alarms>` computes the criticality and turns
every timestamp at which the criticality reaches the threshold into an alarm. Contiguous alarm timestamps are
aggregated into alarm events.

.. code-block:: python

   from energy_fault_detector.utils.analysis import calculate_alarms

   alarm_events, criticality = calculate_alarms(
       anomalies,                    # pd.Series[bool] over time
       normal_idx=None,              # optional pd.Series[bool], True = normal operation
       init_criticality=0,           # starting value of the counter
       criticality_threshold=None,   # None -> derived from the time frequency (see below)
       alarm_reset_time_minutes=60,  # reset window used to derive max_criticality
   )

``alarm_events`` is a :class:`pandas.DataFrame` with the columns ``start``, ``end`` and ``duration`` (and an
additional ``group`` column for a ``MultiIndex``). ``criticality`` is the :class:`pandas.Series` of the counter over
time.

If ``criticality_threshold`` is not given, it is derived from the time frequency of the data:

- ``criticality_threshold = int(1 hour / time_frequency)`` — the number of timestamps needed to accumulate one hour
  of anomalous behaviour.
- If the time frequency is larger than one hour (or cannot be inferred, e.g. for a ``RangeIndex``), the threshold
  defaults to ``1``.

``max_criticality`` is derived from the threshold and ``alarm_reset_time_minutes``, which has a default value of ``60``:

- ``max_criticality = criticality_threshold + int(alarm_reset_time_minutes / time_frequency)``
- If the time frequency is larger than ``alarm_reset_time_minutes`` (or cannot be inferred),
  ``max_criticality = criticality_threshold + 1``.

The time frequency is inferred as the median difference between consecutive timestamps, using the datetime level of a
``MultiIndex`` when present. For MultiIndex data the criticality and the alarm events are computed independently per
group (e.g. per asset).

On a prediction result you can call the method directly. It returns the same ``(alarm_events, criticality)`` tuple:

.. code-block:: python

   results = fault_detector.predict(sensor_data=test_sensor_data)

   alarm_events, criticality = results.criticality_alarms()   # defaults derived from the data
   alarm_events, criticality = results.criticality_alarms(
       normal_idx=test_status,
       criticality_threshold=6,
       alarm_reset_time_minutes=60,
   )

For the plain criticality counter (without alarm event detection) use the legacy
:meth:`FaultDetectionResult.criticality <energy_fault_detector.core.fault_detection_result.FaultDetectionResult.criticality>`
method, which returns only the ``criticality`` series:

.. code-block:: python

   criticality = results.criticality(normal_idx=test_status, max_criticality=144)

Interpreting criticality: raising alarms
----------------------------------------

Criticality is interpreted relative to an application-specific **criticality threshold**:

- as long as ``criticality < threshold``, no alarm is raised,
- once ``criticality >= threshold``, an alarm is raised (the anomaly is treated as significant).

Because the counter rises by one per anomalous timestamp, the threshold can be read as the minimum number of
consecutive anomalous timestamps required to raise an alarm. With a fixed sampling frequency this corresponds directly
to a time window:

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
- **Expected random anomalies** from noise or data-quality problems. These can be estimated from real anomalies used
  for validation and/or from validation data with normal status: run the model on validation data known to be normal,
  observe how the criticality behaves, and choose a threshold that random fluctuations do not reach. The more
  (expected) spurious anomalies per unit of time, the higher the threshold should be.
- **Severity and cost** of a missed fault versus a false alarm.

A simple procedure:

#. Compute the criticality on validation data that has normal status.
#. Observe the maximum criticality reached by random anomalies (noise / data quality).
#. Set the threshold well above that level, and additionally require the desired minimum persistence time.

Choosing ``alarm_reset_time_minutes``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``alarm_reset_time_minutes`` controls how long it takes for an active alarm to be reset after the system returns to
normal behaviour, by determining ``max_criticality``. Once the counter has saturated at ``max_criticality``, it must be
decremented back below the threshold before the alarm is closed. Because the counter decreases by one per normal
timestamp, the time to reset from saturation is at most:

.. code-block:: text

   max_reset_time = (max_criticality - criticality_threshold) * time_frequency
                  = alarm_reset_time_minutes

Continuing the example above (``10 minutes`` resolution) with a threshold of ``6`` and
``alarm_reset_time_minutes = 60``, ``max_criticality`` becomes ``6 + int(60 / 10) = 12``:

.. math::

   (12 - 6) \times 10\,\text{min} = 60\,\text{min}

so at most **one hour** of consecutive normal behaviour is needed for the counter to fall from saturation back below
the threshold and for the alarm to close.

Setting the reset time to at most **1 hour** or **1 day** is often a good default, depending on the application
context and user preferences.

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
   The threshold and ``max_criticality`` are expressed in *number of timestamps*, not in physical time. When you
   change the time frequency of your data, recompute them so that the intended physical time windows are preserved.
   ``calculate_alarms`` does this automatically for the defaults.

Alarm event output
^^^^^^^^^^^^^^^^^^

Each alarm event contains the first and last timestamp at which the criticality was at or above the threshold:

.. code-block:: python

   alarm_events, criticality = calculate_alarms(
       anomalies, criticality_threshold=6, alarm_reset_time_minutes=60
   )
   print(alarm_events)

   #                 start                 end        duration
   # 0 2024-01-01 01:10:00 2024-01-01 03:10:00 0 days 02:00:00

The ``duration`` is the difference between ``end`` and ``start``. For a ``MultiIndex`` input an additional
``group`` column identifies the asset or device.

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
       criticality_threshold=6,
       alarm_reset_time_minutes=60,
   )

Relation to evaluation (CARE score)
-----------------------------------

The same criticality concept is used by the
:class:`CAREScore <energy_fault_detector.evaluation.care_score.CAREScore>` to decide whether a complete event is
treated as detected: an event is classified as anomalous when its maximum criticality reaches
``criticality_threshold``. See :doc:`care2compare_guide` and :doc:`care2compare_faq`. As with alarm raising, this
threshold should be chosen based on the application context and time frequency, and tuned for the dataset at hand.
