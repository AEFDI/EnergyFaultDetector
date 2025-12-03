Usage examples
================================
To see interactive demonstrations of the energy fault detection package,
refer to the example notebooks in the repository's notebooks folder.

.. toctree::
    :caption: Contents
    :glob:
    :maxdepth: 2

.. contents:: Table of Contents
    :depth: 3
    :local:


Energy Fault Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The main interface for the `energy-fault-detector` package is the :py:obj:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>` class, which
needs a configuration object :py:obj:`Config <energy_fault_detector.config.config.Config>`.

To create a new :py:obj:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>` model,
create a configuration, as described below in the :ref:`configuration` section, and run:

.. code-block:: python

    from energy_fault_detector import FaultDetector, Config

    config = Config('configs/basic_config.yaml')
    fault_detector = FaultDetector(config=config, model_directory='model_directory')

To train new models, you need to provide the input data and call the :py:obj:`FaultDetector.fit <energy_fault_detector.fault_detector.FaultDetector.fit>` method:

.. code-block:: python

    # get data from database / csv / API ...
    sensor_data = ...  # a pandas DataFrame with timestamp as index and numerical sensor values as columns
    normal_index = ...  # a pandas Series with timestamp as index and booleans indicating normal behaviour
    # NOTE: The normal_index is optional; it is used to select training data for the autoencoder.
    # If not provided, we assume all data represents normal behaviour.
    # If you do not have any labels, you cannot use th F-beta-score- and FDR-based thresholds.

    # If you do not use the models for time series, the index can also be a standard RangeIndex,
    # as long as the sensor_data DataFrame and the normal_index Series share the same index.

    model_data = fault_detector.fit(sensor_data=sensor_data, normal_index=normal_index, save_models=True)

    # to save model manually:
    # fault_detector.save_models('model_name')  # model_name is optional

The trained models are saved locally in the provided ``model_directory``. The :py:obj:`FaultDetector.fit <energy_fault_detector.fault_detector.FaultDetector.fit>` method returns a
:py:obj:`ModelMetadata <energy_fault_detector.core.fault_detection_result.ModelMetadata>` object with
the model metadata such as the model date and the model path.

To predict using the trained model, use the :py:obj:`FaultDetector.predict <energy_fault_detector.fault_detector.FaultDetector.predict>` method:

.. code-block:: python

    results = fault_detector.predict(sensor_data=test_sensor_data)

The result is a :py:obj:`FaultDetectionResult <energy_fault_detector.core.fault_detection_result.FaultDetectionResult>` object
with the following information:

* predicted_anomalies: pandas Series with the predicted anomalies (bool).
* reconstruction: pandas DataFrame with reconstruction of the sensor data with timestamp as index.
* deviations: pandas DataFrame with reconstruction errors.
* anomaly_score: pandas Series with anomaly scores for each timestamp.
* bias_data: pandas DataFrame with ARCANA results with timestamp as index. None if ARCANA was not run.
* arcana_losses: pandas DataFrame containing recorded values for all losses in ARCANA. None if ARCANA was not run.
* tracked_bias: List of pandas DataFrames. None if ARCANA was not run.

You can also create a :py:obj:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>` object and load
trained models using the :py:obj:`FaultDetector.load_models <energy_fault_detector.core.fault_detection_model.FaultDetectionModel.load_models>` method. In this case, you do not need to provide a ``model_path``
in the :py:obj:`predict <energy_fault_detector.fault_detector.FaultDetector.predict>` method.

.. code-block:: python

    from energy_fault_detector.fault_detector import FaultDetector

    fault_detector = FaultDetector()
    fault_detector.load_models('path_to_trained_models')

    # get data from database / csv / API ...
    sensor_data = ...
    results = fault_detector.predict(sensor_data=sensor_data)


.. _configuration:

Configuration
^^^^^^^^^^^^^

The training configuration is set with a ``yaml`` file which contains ``train`` specification with model settings, to
train new models and ``root_cause_analysis`` specification if you want to analyse the model predictions with the `ARCANA`
algorithm. An example:

.. include:: basic_config.yaml
   :literal:

If you leave out the data_preprocessor configuration (i.e., ``data_preprocessor: None``), as default preprocessing
pipeline is generated, which drops constant features, features where >5% of the data is missing, imputes remaining
missing values with the mean value and scales the data to zero mean and unit standard deviation.

See the :ref:`Configuration guide <configuration_guide>` for more details on the configuration file and options.

To update the configuration 'on the fly' (for example for hyperparameter optimization), you provide a new
configuration dictionary via the :py:obj:`Config.update_config <energy_fault_detector.config.config.Config.update_config>` method:

.. code-block:: python

  from energy_fault_detector.config import Config
  from copy import deepcopy

  config = Config('configs/base_config.yaml')

  # update some parameters:
  new_config_dict = deepcopy(config.config_dict)
  new_config_dict['train']['anomaly_score']['name'] = 'mahalanobis'
  config.update_config(new_config_dict)

  # or create a new configuration object and model
  new_model = FaultDetector(Config(config_dict=new_config_dict))

You can look up the names for the available model classes in the class registry:

.. code-block:: python

    from energy_fault_detector import registry

    registry.print_available_classes()


Evaluation
^^^^^^^^^^
Please check the example notebooks for evaluation examples.

Creating new model classes
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can extend the framework by creating new model classes based on the templates in the
:py:obj:`core <energy_fault_detector.core>` module and registering the new classes.
Examples are shown in the notebook ``Example - Create new model classes.ipynb``.

Creating your own pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to create your own energy fault detection pipeline with the building blocks of this package,
you can import the data preprocessor, autoencoder, anomaly score and threshold selection classes as follows:

.. code-block:: python

    from energy_fault_detector.data_preprocessing import DataPreprocessor, DataClipper
    from energy_fault_detector.autoencoders import MultilayerAutoencoder
    from energy_fault_detector.anomaly_score import MahalanobisScore
    from energy_fault_detector.threshold_selectors import FbetaSelector

This allows you to add additional steps or use different data preprocessing pipelines.

An example training pipeline (similar to the :py:obj:`FaultDetector <energy_fault_detector.fault_detector.FaultDetector>` class)
would be:

.. code-block:: python

    x = ...  # i.e. sensor data
    y = ...  # normal behaviour indicator

    x_normal = x[y]
    # fit data preprocessor on normal data
    data_preprocessor = DataPreprocessor(...)
    x_normal_prepped = data_preprocessor.fit_transform(x_normal)

    # fit autoencoder on normal data
    ae = MultilayerAutoencoder(...)
    ae.fit(x_normal_prepped)

    # create and fit score
    anomaly_score = MahalanobisScore(...)
    x_prepped = data_preprocessor.transform(x)

    # fit on normal data
    recon_error_normal = ae.get_reconstruction_error(x_normal_prepped)
    anomaly_score.fit(recon_error_normal)
    # get scores of all data points
    recon_error = ae.get_reconstruction_error(x_prepped)
    scores = anomaly_score.transform(recon_error)

    # set the threshold and get predictions to evaluate
    threshold_selector = FbetaSelector(beta=1.0)  # sets optimal threshold based on F1 score
    threshold_selector.fit(scores, y)
    # NOTE: the fit-method of the AdaptiveThreshold has slightly different arguments!
    anomalies = threshold_selector.predict(scores)

And the inference:

.. code-block:: python

    x = ...

    x_prepped = data_preprocessor.transform(x)
    x_recon = ae.predict(x_prepped)  # reconstruction
    x_recon_error = ae.get_reconstruction_error(x_prepped)
    scores = anomaly_score.transform(x_recon_error)
    anomalies = threshold_selector.predict(scores)  # boolean series indicating anomaly detected
