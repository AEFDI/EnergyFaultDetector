CARE2Compare FAQ
================

.. note::

    This page summarizes recurring interpretation questions and caveats around the
    CARE2Compare dataset and the CARE score as supported in EnergyFaultDetector.
    For official release notes and dataset updates, also consult the Zenodo record
    and the companion publication:

    CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data.
    Data. 2024; 9(12):138. https://doi.org/10.3390/data9120138

General label semantics
-----------------------

What is the difference between ``status_type_id`` and ``event_label``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``status_type_id`` is a timestamp-level label that describes the recorded operating mode of the turbine.

``event_label`` is an event-level label indicating whether the prediction section of an event dataset
contains an anomalous event.

They serve different purposes:

- ``status_type_id`` helps interpret turbine operation and filter data,
- ``event_label`` is the target for event-level anomaly evaluation.


Should models predict ``status_type_id`` or ``event_label``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

That depends on the goal. For early fault detection, you want to predict the ``event_label``. On the timestamp level,
this means that you will want to detect anomalies within the provided ``event_start`` and ``event_end`` for events with
``event_label == 'anomaly'`` and no anomalies if the ``event_label == 'normal'``.

The ``status_type_id`` only indicates whether the wind turbine has an operationally normal state or not. Once a fault is
known, this state is often already nor normal. Therefore, it is not interesting to detect whether the state is
anomalous, but it is interesting to find anomalies during expected normal operation before a fault is known or becomes
critical.

For early fault detection, a practical strategy is often:

- predict anomaly or normality at timestamp level,
- then derive an event-level decision from those timestamp-level predictions.

This is also consistent with CARE-style evaluation, where pointwise predictions can be aggregated into
event-wise decisions.


Why can anomalous events contain timestamps with ``status_type_id = 0``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The anomaly window between ``event_start`` and ``event_end`` represents an estimated anomaly time frame
leading up to a fault. During that time, the operator may still have considered the turbine to be in normal
operation. Therefore, timestamps with ``status_type_id = 0`` can appear inside anomalous events.

This is particularly important for early fault detection, because these timestamps are often the most relevant
ones from the operator's perspective.


How do the status labels of Wind Farm A differ from those of Wind Farms B and C?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For Wind Farms B and C, the status labels reflect anonymized operator-recorded turbine states.

For Wind Farm A, the status labels were derived differently and do not provide the same interpretation value
for prediction-time evaluation. The status labels are based on available failure-log information and should mainly be
used for filtering training data. For prediction-time evaluation, they should largely be ignored.


How should ``status_type_id`` be interpreted for Wind Farms B and C?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Wind Farms B and C, ``status_type_id`` is an anonymized version of operator-provided SCADA status
information.

This means a timestamp with ``status_type_id = 0`` reflects that the operator considered the turbine to be in
normal operation at that time, even if later retrospective analysis identified that timestamp as part of an
anomaly window.


Training and preprocessing
--------------------------

Should abnormal timestamps be removed from the training data?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Usually yes, if you are training a normal-behavior model.

The dataset provides ``status_type_id`` so users can filter training data. In practice, filtering by status is
an intended use of the dataset.

Depending on the model and the goal, it can also be reasonable to apply additional cleaning such as power-curve-based
filtering.


Has the dataset already been preprocessed?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No. The data was anonymized, but it has not undergone full preprocessing such as:

- missing-value removal,
- invalid-measurement filtering,
- normalization or scaling.

Additional preprocessing is generally required before modeling.
Please check the dataset description and dataset README on `Zenodo <https://doi.org/10.5281/zenodo.14958989>`_
for details on known data quality issues.


What preprocessing is typically needed?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This depends on the wind farm and the model, but common steps include:

- feature selection,
- NaN imputation,
- angle transformation,
- filtering invalid measurements,
- scaling.

An example is found in the notebook `CARE to Compare.ipynb <https://github.com/AEFDI/EnergyFaultDetector/blob/main/notebooks/CARE%20to%20Compare/CARE%20to%20Compare.ipynb>`_


Does using the dataset require domain knowledge?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For purely statistical anomaly detection experiments, general data analysis methods may be sufficient up to a
point. However, domain knowledge becomes increasingly important for:

- feature selection,
- validating detected anomalies,
- interpreting event behavior,
- root-cause-oriented analysis.


Evaluation
----------

Which timestamps are used for pointwise evaluation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For CARE-style pointwise measures, timestamps with normal status are the important ones.

We consider a timestamp to be correctly detected as anomalous, if the timestamp is part of an event with
``event_label == 'anomaly'``. It is correctly detected as normal behaviour if the timestamp is part of an event with
``event_label == 'normal'``.

Timestamps with abnormal status labels are excluded from pointwise evaluation, because it is not
useful to reward detection of a timestamp as anomalous when the operator already knew it was not in normal
operation. Note that we only apply this rule to wind farm B and C. For WF A, the status information can only be used
for filtering training data and not for evaluation.


How is the ground truth for Coverage interpreted?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coverage is a pointwise F-score computed on anomalous events.

The ground truth is as follows:

- timestamps with anomalous status are anomalous,
- timestamps between ``event_start`` and ``event_end`` of anomalous events are also considered anomalous,
- all other timestamps are considered normal.

For wind farm B and C the first type of true anomalies are ignored, because it is not
useful to reward detection of a timestamp as anomalous when the operator already knew it was not in normal
operation.


Why can true positives still exist after excluding abnormal-status timestamps?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because anomalous events can contain timestamps with normal status labels.

After excluding abnormal-status timestamps, the remaining timestamps may still lie inside the anomalous event
window. Those timestamps are still treated as anomalous ground truth for Coverage and can therefore produce
true positives.


How is Reliability computed?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reliability is an event-wise F-score. The model used should provide a decision for a complete event. This decision can
then be compared with the ``event_label``.

As an example, in the paper we used a criticality measure, which is accumulated over timestamp-level predictions:

- criticality increases when an anomaly is detected on a timestamp with normal status,
- criticality decreases when no anomaly is detected on a timestamp with normal status,
- criticality does not change on timestamps with abnormal status.

If the criticality reaches the threshold, the whole event is treated as predicted anomalous; otherwise it is
treated as predicted normal. Those event-level predictions are then compared with ``event_label`` values.


What is the default criticality threshold?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default threshold used in the CARE description and package discussions is ``72``.
You should of course tune this threshold for your dataset.


Why is Reliability not averaged like the other CARE components?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because Reliability is already computed as an event-wise score across events.

By contrast, Coverage, Accuracy, Earliness are first computed per event and then averaged over the relevant events.


Model training strategy
-----------------------

Should I train one model per turbine or one model per wind farm?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is no universally correct answer.

Possible strategies include:

- one model per turbine,
- one model per wind farm,
- hybrid strategies combining local and shared information.


For early fault detection, a practical strategy is often:

- predict anomaly or normality at timestamp level,
- then derive an event-level decision from those timestamp-level predictions.

This is also consistent with CARE-style evaluation, where pointwise predictions can be aggregated into
event-wise decisions.

In the dataset paper, individual autoencoder models per turbine were used.


Features and interpretation
---------------------------

Can the feature names be mapped to more detailed physical sensor identities?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because of anonymization agreements with data providers, no additional sensor metadata beyond the published
dataset description may be available.

The feature descriptions are the most detailed available for this dataset. If you load the data using the
:class:`energy_fault_detector.evaluation.care2compare.Care2CompareDataset`, the feature names are based on these
descriptions (instead of the enumerated feature names of the csv files).


Are all feature groups equally trustworthy?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are implausible values in some Min, Max, and Std features. For practical work,
Avg features are often the safest starting point, especially in Wind Farm B.

Please check the dataset description and dataset README on `Zenodo <https://doi.org/10.5281/zenodo.14958989>`_
for details on known data quality issues.


Root-cause analysis and ARCANA
------------------------------

Was ARCANA used to create the root-cause labels?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No. The root-cause information included in the dataset is based on operator feedback and service reports,
not on ARCANA-generated labels.


Benchmark reproducibility
-------------------------

Is the exact original benchmark code available?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not exactly.

The EnergyFaultDetector repository provides open-source implementations related to the CARE workflow,
including CARE score support and example notebooks. However, the exact original code used for the paper is
not available.


Can benchmark scores vary when reproduced?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. Even when following the provided examples and configurations, results can vary due to factors such as random
initialization of the models and later package updates.


Timestamps and anonymization
----------------------------

Can different event files still contain overlapping timestamps?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. Overlaps across different event CSV files are an expected side effect of the anonymization procedure. Event
files may therefore share timestamp ranges even when they represent different events.


Can I reconstruct the true chronological order of all events for one asset?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generally no. Because timestamps were anonymized per file, the true chronological order across event files is
not preserved.


See also
--------

- :doc:`care2compare_guide`
- :class:`energy_fault_detector.evaluation.care2compare.Care2CompareDataset`
- :class:`energy_fault_detector.evaluation.care_score.CAREScore`