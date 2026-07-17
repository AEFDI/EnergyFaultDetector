CARE2Compare guide
==================

.. note::

    This page documents how CARE2Compare support is exposed in EnergyFaultDetector.
    For generated API reference pages, see the corresponding module documentation.
    For interpretation caveats and recurring dataset questions, see :doc:`care2compare_faq`.

EnergyFaultDetector provides support for:

- loading CARE2Compare event datasets,
- accessing event metadata,
- formatting normal-operation masks based on ``status_type_id``,
- evaluating predictions with the CARE score.

Relevant classes include:

- :class:`energy_fault_detector.evaluation.care2compare.Care2CompareDataset`
- :class:`energy_fault_detector.evaluation.care_score.CAREScore`

Background
----------

The CARE2Compare dataset and CARE score are introduced in:

`CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data <https://doi.org/10.3390/data9120138>`_

In the package, CARE2Compare support is intended to help with two tasks:

#. loading event-based benchmark datasets,
#. evaluating anomaly predictions in a way that is aligned with the CARE benchmark.

Conceptual overview
-------------------

The dataset contains two related but distinct kinds of labels:

- ``status_type_id``:
  timestamp-level operating-status information,
- ``event_label``:
  event-level label indicating whether the prediction section contains an anomalous event.

These labels should not be treated as interchangeable.

In particular, anomalous events may still contain timestamps with ``status_type_id = 0``.
This is expected for CARE2Compare and reflects the difference between:

- the operator's recorded turbine status at a timestamp, and
- the retrospectively defined anomaly window used for early fault detection.

For details, see :doc:`care2compare_faq`.

Dataset loading
---------------

The :class:`energy_fault_detector.evaluation.care2compare.Care2CompareDataset` helper can be used to
load local CARE2Compare data or download the dataset automatically, depending on package configuration.

Typical usage:

.. code-block:: python

    from energy_fault_detector.evaluation.care2compare import Care2CompareDataset

    dataset = Care2CompareDataset(
        path="./CARE_To_Compare",
        download_dataset=False,
    )

    x_train, x_test = dataset.load_event_dataset(event_id=53)
    info = dataset.get_event_info(53)

The returned event metadata can then be used for evaluation, filtering, or reporting.

Formatted loading
-----------------

If you want a convenience split into values and normal-operation masks, use the formatted loader.

.. code-block:: python

    x_train, train_normal, x_test, test_normal = dataset.load_and_format_event_dataset(event_id=53)

In the package, the normal masks are derived from:

.. code-block:: text

    status_type_id == 0

This is mainly useful when training normal-behavior models or when applying pointwise evaluation logic.

Training-data filtering
-----------------------

For normal-behavior modeling, it is generally recommended to filter out timestamps that are clearly not
representative of normal operation.

In practice, this often means using ``status_type_id`` to exclude abnormal operating modes from the
training data.

However, CARE2Compare should not be understood as fully pre-cleaned data. Depending on the model and
wind farm, additional preprocessing may still be needed, such as:

- missing-value handling,
- invalid-measurement filtering,
- feature selection,
- angle transformation,
- scaling.

Example workflow
----------------

A typical workflow looks like this:

#. load an event dataset,
#. separate training and prediction sections,
#. derive or load model predictions for prediction timestamps,
#. evaluate those predictions with :class:`energy_fault_detector.evaluation.care_score.CAREScore`.

Example:

.. code-block:: python

    from energy_fault_detector.evaluation.care2compare import Care2CompareDataset
    from energy_fault_detector.evaluation.care_score import CAREScore

    dataset = Care2CompareDataset(path="./CARE_To_Compare", download_dataset=False)

    x_train, train_normal, x_test, test_normal = dataset.load_and_format_event_dataset(event_id=53)
    info = dataset.get_event_info(53)

    # Example placeholder prediction:
    # one boolean anomaly prediction per timestamp in the prediction section
    predicted_anomalies = [False] * len(x_test)

    scorer = CAREScore()

    scorer.evaluate_event(
        event_start=info["event_start"],
        event_end=info["event_end"],
        event_label=info["event_label"],
        predicted_anomalies=predicted_anomalies,
        normal_index=test_normal,
        event_id=53,
    )

    final_score = scorer.get_final_score()

CARE score overview
-------------------

The CARE score combines four aspects of detection quality:

- **Coverage**
- **Accuracy**
- **Reliability**
- **Earliness**

At a high level:

- Coverage and Earliness are computed on anomalous events,
- Accuracy is computed on normal events,
- Reliability is an event-wise score based on event decisions.

The package implementation follows the CARE benchmark logic provided in the publication and subsequent
package support. For detailed interpretation notes, see :doc:`care2compare_faq`.

Wind-farm-specific interpretation
---------------------------------

The meaning and usefulness of ``status_type_id`` differs slightly between wind farms.

For Wind Farms B and C, status labels reflect anonymized operator-provided status information and are
useful both for training-data filtering and for parts of CARE-style evaluation.

For Wind Farm A, the status labels should mainly be used for training-data filtering. For prediction-time
CARE evaluation, they should largely be ignored.

This distinction is important when interpreting evaluation results.

Known practical limitations
---------------------------

Users should be aware of several dataset and benchmark limitations:

- the dataset is anonymized,
- the data is not fully preprocessed for modeling,
- exact benchmark code from the original paper is not guaranteed to match the current package state,
- overlapping timestamps across different event files may occur due to anonymization,
- some version-specific data-quality notes apply.

These are discussed in more detail in :doc:`care2compare_faq`.

See also
--------

- :doc:`care2compare_faq`
- :class:`energy_fault_detector.evaluation.care2compare.Care2CompareDataset`
- :class:`energy_fault_detector.evaluation.care_score.CAREScore`
