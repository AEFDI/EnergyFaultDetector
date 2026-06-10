.. _advanced_usage:

Advanced usage and customization
================================

This page covers more advanced use cases:

- Creating new model classes that integrate with the :mod:`energy_fault_detector` registry
- Building your own pipeline from the building blocks (preprocessors, autoencoders, scores, thresholds)


Creating new model classes
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can extend the framework by creating new model classes based on the templates in the
:py:mod:`energy_fault_detector.core` module and registering the new classes.

Typical steps:

1. Implement a new class that inherits from one of the core base classes, such as:

   - :class:`~energy_fault_detector.core.autoencoder.Autoencoder`
   - :class:`~energy_fault_detector.core.anomaly_score.AnomalyScore`
   - :class:`~energy_fault_detector.core.threshold_selector.ThresholdSelector`
   - :class:`~energy_fault_detector.core.data_transformer.DataTransformer`

2. Register your class in :mod:`energy_fault_detector.registration` so it can be referenced
   by name from the YAML config:

   .. code-block:: python

      from energy_fault_detector.registration import register

      register(
          module_path="my_package.my_autoencoder.MyCustomAE",
          class_type="autoencoder",
          class_names=["MyCustomAE", "my_custom_ae"],
      )

3. Use the registered name in the config, for example:

   .. code-block:: yaml

      train:
        autoencoder:
          name: my_custom_ae
          params:
            # your model parameters here
            layers: [128, 64, 32]
            code_size: 16

Examples are shown in the notebook ``Example - Create new model classes.ipynb`` (see the
``notebooks/`` folder in the repository).


Creating your own pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to create your own energy fault detection pipeline using the building blocks of this package,
you can import the data preprocessor, autoencoder, anomaly score and threshold selection classes directly:

.. code-block:: python

    from energy_fault_detector.data_preprocessing import DataPreprocessor, DataClipper
    from energy_fault_detector.autoencoders import MultilayerAutoencoder
    from energy_fault_detector.anomaly_scores import MahalanobisScore
    from energy_fault_detector.threshold_selectors import FbetaSelector

This allows you to add additional steps or use different data preprocessing pipelines.

An example training pipeline (similar to the :class:`FaultDetector
<energy_fault_detector.fault_detector.FaultDetector>` class) would be:

.. code-block:: python

    x = ...  # sensor data as DataFrame
    y = ...  # normal behaviour indicator as boolean Series

    x_normal = x[y]

    # 1) Fit data preprocessor on normal data
    data_preprocessor = DataPreprocessor(...)
    x_normal_prepped = data_preprocessor.fit_transform(x_normal)

    # 2) Fit autoencoder on normal data
    ae = MultilayerAutoencoder(...)
    ae.fit(x_normal_prepped)

    # 3) Create and fit score
    anomaly_score = MahalanobisScore(...)
    x_prepped = data_preprocessor.transform(x)

    recon_error_normal = ae.get_reconstruction_error(x_normal_prepped)
    anomaly_score.fit(recon_error_normal)

    # 4) Compute scores for all data points
    recon_error = ae.get_reconstruction_error(x_prepped)
    scores = anomaly_score.transform(recon_error)

    # 5) Set the threshold and get predictions to evaluate
    threshold_selector = FbetaSelector(beta=1.0)  # sets optimal threshold based on F1 score
    threshold_selector.fit(scores, y)
    anomalies = threshold_selector.predict(scores)  # boolean Series indicating anomaly detected

Inference then looks like:

.. code-block:: python

    x_new = ...

    x_prepped = data_preprocessor.transform(x_new)
    x_recon = ae.predict(x_prepped)              # reconstruction
    x_recon_error = ae.get_reconstruction_error(x_prepped)
    scores = anomaly_score.transform(x_recon_error)
    anomalies = threshold_selector.predict(scores)

This pattern is useful when:

- you want full control over training and evaluation loops,
- you are integrating EnergyFaultDetector components into a larger existing pipeline,
- or you are experimenting with custom combinations of preprocessing, models, and thresholds.
