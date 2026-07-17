CARE2Compare FAQ
================

.. note::

    This page summarizes recurring interpretation questions and caveats around the
    CARE2Compare dataset and the CARE score as supported in EnergyFaultDetector.
    For official release notes and dataset updates, also consult the Zenodo record
    and the companion publication.

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

That depends on the modeling approach.

For early fault detection, a practical strategy is often:

- predict anomaly or normality at timestamp level,
- then derive an event-level decision from those timestamp-level predictions.

This is also consistent with CARE-style evaluation, where pointwise predictions can be aggregated into
event-wise decisions.

Why can anomalous events contain timestamps with ``status_type_id = 0``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is expected.

The anomaly window between ``event_start`` and ``event_end`` represents an estimated anomaly time frame
leading up to a fault. During that time, the operator may still have considered the turbine to be in normal
operation. Therefore, timestamps with ``status_type_id = 0`` can appear inside anomalous events.

This is particularly important for early fault detection, because these timestamps are often the most relevant
ones from the operator's perspective.

Training and preprocessing
--------------------------

Should abnormal timestamps be removed from the training data?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Usually yes, if you are training a normal-behavior model.

The dataset provides ``status_type_id`` so users can filter training data. In practice, filtering by status is
an intended use of the dataset.

Depending on the model, it can also be reasonable to apply additional cleaning such as power-curve-based
filtering.

Has the dataset already been preprocessed?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No.

The data was anonymized, but it has not undergone full preprocessing such as:

- missing-value removal,
- invalid-measurement filtering,
- normalization or scaling.

Additional preprocessing is generally required before modeling.

What preprocessing is typically needed?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This depends on the wind farm and the model, but common steps include:

- feature selection,
- NaN imputation,
- angle transformation,
- filtering invalid measurements,
- scaling.

The EnergyFaultDetector package can be used as an example implementation of such preprocessing, but it is
not the only possible approach.

Does using the dataset require domain knowledge?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not always to the same extent.

For purely statistical anomaly detection experiments, general data analysis methods may be sufficient up to a
point. However, domain knowledge becomes increasingly important for:

- feature selection,
- validating detected anomalies,
- interpreting event behavior,
- root-cause-oriented analysis.

Pointwise CARE evaluation
-------------------------

Which timestamps are used for pointwise evaluation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For CARE-style pointwise measures, timestamps with normal status are the important ones.

In practice, timestamps with abnormal status labels are excluded from pointwise evaluation, because it is not
useful to reward detection of a timestamp as anomalous when the operator already knew it was not in normal
operation.

How is the ground truth for Coverage interpreted?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coverage is a pointwise F-score computed on anomalous events.

The intended interpretation is:

- timestamps with anomalous status are anomalous,
- timestamps between ``event_start`` and ``event_end`` of anomalous events are also considered anomalous,
- all other timestamps are considered normal.

At the same time, abnormal-status timestamps can be omitted from the pointwise evaluation set, so true
positives can still arise from timestamps with normal status inside the anomalous event window.

Why can true positives still exist after excluding abnormal-status timestamps?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because anomalous events can contain timestamps with normal status labels.

After excluding abnormal-status timestamps, the remaining timestamps may still lie inside the anomalous event
window. Those timestamps are still treated as anomalous ground truth for Coverage and can therefore produce
true positives.

Should samples with ``status_type_id = 0`` inside the event window be excluded from evaluation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No.

Those timestamps are often exactly the timestamps of interest for early fault detection. In CARE-style
evaluation, they are typically retained and used for pointwise evaluation.

Reliability and event decisions
-------------------------------

How is Reliability computed?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reliability is an event-wise F-score.

For each event dataset, a criticality measure is accumulated over timestamp-level predictions. In the described
CARE logic:

- criticality increases when an anomaly is detected on a timestamp with normal status,
- criticality decreases when no anomaly is detected on a timestamp with normal status,
- criticality does not change on timestamps with abnormal status.

If the criticality reaches the threshold, the whole event is treated as predicted anomalous; otherwise it is
treated as predicted normal. Those event-level predictions are then compared with ``event_label`` values.

What is the default criticality threshold?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default threshold used in the CARE description and package discussions is ``72``.

Why is Reliability not averaged like the other CARE components?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because Reliability is already computed as an event-wise score across events.

By contrast:

- Coverage,
- Accuracy,
- Earliness

are first computed per event and then averaged over the relevant events.

Wind Farm A special handling
----------------------------

Can ``status_type_id`` be used normally for Wind Farm A?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not in the same way as for Wind Farms B and C.

For Wind Farm A, the status labels are based on available failure-log information and should mainly be used
for filtering training data. For prediction-time CARE evaluation, they should largely be ignored.

Why is Wind Farm A different?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Wind Farms B and C, the status labels reflect anonymized operator-recorded turbine states.

For Wind Farm A, the status labels were derived differently and do not provide the same interpretation value
for prediction-time evaluation.

Are there known status-label issues in Wind Farm A?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, there have been reported issues in earlier dataset versions.

These included observations such as:

- problematic status assignments in some events,
- inconsistencies in the expected relation between status IDs 3 and 4,
- status ID 5 being an internal derived label that may later be removed from some descriptions.

When discussing such issues, it is best to make the statement version-specific.

Wind Farms B and C
------------------

How should ``status_type_id`` be interpreted for Wind Farms B and C?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Wind Farms B and C, ``status_type_id`` is an anonymized version of operator-provided SCADA status
information.

This means a timestamp with ``status_type_id = 0`` reflects that the operator considered the turbine to be in
normal operation at that time, even if later retrospective analysis identified that timestamp as part of an
anomaly window.

Model training strategy
-----------------------

Should I train one model per turbine or one model per wind farm?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is no universally correct answer.

Possible strategies include:

- one model per turbine,
- one model per wind farm,
- hybrid strategies combining local and shared information.

In the referenced benchmark work, individual autoencoder models per turbine were used.

Features and interpretation
---------------------------

Can the feature names be mapped to more detailed physical sensor identities?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not fully.

Because of anonymization agreements with data providers, no additional sensor metadata beyond the published
dataset description may be available.

Are all feature groups equally trustworthy?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not necessarily.

Version-specific notes indicate implausible values in some Min, Max, and Std features. For practical work,
Avg features are often the safest starting point, especially in Wind Farm B.

Root-cause analysis and ARCANA
------------------------------

Was ARCANA used to create the root-cause labels?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No.

ARCANA was used by the authors to analyze which features contribute to reconstruction error, i.e. for
model-dependent feature-importance analysis. The root-cause information included in the dataset is based on
operator feedback and service reports, not on ARCANA-generated labels.

Benchmark reproducibility
-------------------------

Is the exact original benchmark code available?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not exactly.

The EnergyFaultDetector repository provides open-source implementations related to the CARE workflow,
including CARE score support and example notebooks. However, the exact original code used for the paper is
not guaranteed to be available unchanged.

Can benchmark scores vary when reproduced?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes.

Even when following the provided examples and configurations, results can vary due to factors such as model
training randomness and later package updates.

Timestamps and anonymization
----------------------------

Are duplicate timestamps within a file expected?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No, duplicate timestamps within a single CSV file were reported as an issue and later fixed in newer dataset
versions.

Can different event files still contain overlapping timestamps?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes.

Overlaps across different event CSV files are an expected side effect of the anonymization procedure. Event
files may therefore share timestamp ranges even when they represent different benchmark events.

Can I reconstruct the true chronological order of all events for one asset?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generally no.

Because timestamps were anonymized per file, the true chronological order across event files is not preserved.

Does this create a data-leakage risk across events?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Potentially yes, if data from multiple events of the same asset are combined without care.

The intended benchmark usage is event-based. If you aggregate across events, you should be aware that
overlaps introduced by anonymization may make leakage hard to rule out completely.


See also
--------

- :doc:`care2compare_guide`
- :class:`energy_fault_detector.evaluation.care2compare.Care2CompareDataset`
- :class:`energy_fault_detector.evaluation.care_score.CAREScore`