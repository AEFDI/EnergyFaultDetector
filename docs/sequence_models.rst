Sequence-based models
=====================

This section explains how to use the sequence-to-one autoencoders:

- :class:`~energy_fault_detector.autoencoders.LSTMSeq2OneAutoencoder`
- :class:`~energy_fault_detector.autoencoders.CNNSeq2OneAutoencoder`

These models take **windows** of time-series data and reconstruct the
**last timestep** of each window. They require a `DatetimeIndex` and a
sequence configuration in the YAML config.

When to use sequence models
---------------------------

Sequence-based autoencoders model a window of past timesteps and
reconstruct the **last timestep** in that window. They are useful when:

- your data is **time-correlated** and you expect the recent history to matter for the current timestep, and
- you want the model to be **causal** (only past data in the window),
- you want to take advantage of temporal context.

For purely static data (one row = one independent sample), the dense
:class:`~energy_fault_detector.autoencoders.MultilayerAutoencoder` is
usually sufficient.

The package provides:

- :py:class:`LSTMSeq2OneAutoencoder <energy_fault_detector.autoencoders.lstm_seq2one_autoencoder.LSTMSeq2OneAutoencoder>`
- :py:class:`CNNSeq2OneAutoencoder <energy_fault_detector.autoencoders.cnn_seq2one_autoencoder.CNNSeq2OneAutoencoder>`

Configuration
-------------

To enable a sequence model, set the autoencoder name and add a
``sequence_builder`` section inside ``train.autoencoder.params``:

.. code-block:: yaml

   train:
     autoencoder:
       name: LSTMSeq2OneAutoencoder
       verbose: 0
       params:
         sequence_builder:
           sequence_length: 36       # window length (number of time steps)
           stride: 1                 # step between window starts
           ts_freq: "30s"            # expected time delta between rows
           pad_incomplete: false
           pad_value: 0.0

         layers: [64, 32]
         dropout_rate: 0.0
         regularization: 0.01
         stateful: false
         learning_rate: 0.001
         batch_size: 32
         epochs: 15
         loss_name: "mean_squared_error"
         metrics: ["mean_absolute_error"]
         early_stopping: false
         noise: 0.0

Important points:

- ``sequence_length``: size of each input window, in number of rows.
- ``stride``: how far the window moves each step:
  - ``stride=1`` → heavily overlapping windows,
  - ``stride=sequence_length`` → disjoint windows.
- ``ts_freq``: expected sampling interval. The config parser converts
  strings like ``"30s"``, ``"10m"`` into ``numpy.timedelta64``.
- ``pad_incomplete``: if ``true``, the data is resampled to a regular
  grid at ``ts_freq`` and missing timestamps are filled with ``pad_value``.

Data requirements
-----------------

For sequence models, your ``sensor_data`` must be:

- a :class:`pandas.DataFrame` with a :class:`pandas.DatetimeIndex`,
- sorted in ascending time order,
- with numeric columns only (after the :class:`~energy_fault_detector.data_preprocessing.DataPreprocessor`).

Any gaps larger than ``ts_freq`` are treated as **data gaps**. Windows that cross such gaps are
dropped, so sequences always represent contiguous data.

Example usage
-------------

Training and prediction are identical to the dense autoencoder case:

.. code-block:: python

   import pandas as pd
   from energy_fault_detector import FaultDetector, Config

   df = pd.read_csv("my_timeseries.csv",
                    parse_dates=["timestamp"],
                    index_col="timestamp").sort_index()

   sensor_data = df[["power", "wind_speed", "pitch"]]
   normal_index = df["status"] == "normal"

   cfg = Config("lstm_seq2one_config.yaml")
   fd = FaultDetector(config=cfg)

   model_meta = fd.fit(sensor_data=sensor_data, normal_index=normal_index)

   results = fd.predict(sensor_data=sensor_data)

   # Each prediction is aligned to the **last timestamp** of its window:
   print(results.reconstruction.index[:5])


Conditional features
--------------------

Sequence models can use additional **conditional features** that are
fed to the model at each timestep but are not reconstructed. Configure them as:

.. code-block:: yaml

   train:
     autoencoder:
       name: LSTMSeq2OneAutoencoder
       params:
         sequence_builder:
           sequence_length: 36
           stride: 1
           ts_freq: "10m"
         conditional_features:
           - hour_of_day_sine
           - hour_of_day_cosine
           - day_of_week_sine
           - day_of_week_cosine

         # rest as before ...

Then add these columns to your DataFrame before training (e.g. using
the helper functions shown in the PreDist notebooks).
The model will then reconstruct only the **main** features (all columns
that are not listed as conditional).
