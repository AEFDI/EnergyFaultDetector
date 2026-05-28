Sequence models
===============

Sequence models operate on **windows** of time-series data and require:

- a `pandas.DataFrame` with a :class:`pandas.DatetimeIndex`,
- sorted in ascending time order,
- numeric columns only (after the :class:`~energy_fault_detector.data_preprocessing.DataPreprocessor`),
- a ``sequence_builder`` configuration in the YAML config.

The package provides two families:

- **Seq2One** (sequence-to-one): reconstruct the **last timestep** of each window.
- **Seq2Seq** (sequence-to-sequence): reconstruct the **entire window**.

For purely static data (one row = one independent sample), the dense
:class:`~energy_fault_detector.autoencoders.MultilayerAutoencoder` is
usually sufficient. In many time-series applications, the dense
:class:`~energy_fault_detector.autoencoders.ConditionalAE` can also work well
if time features such as hour-of-day or day-of-week are provided as conditionals.

When to use sequence models
---------------------------

Sequence models are useful when:

- your data is **time-correlated** and recent history matters for the current timestep,
- you want to take advantage of temporal context instead of treating each row independently.

They assume **causal** windows by construction (only past timesteps in each window).

Data requirements
-----------------

For all sequence models, your ``sensor_data`` must be:

- a :class:`pandas.DataFrame` with a :class:`pandas.DatetimeIndex`,
- sorted in ascending time order,
- with numeric columns only.

Any gaps larger than ``ts_freq`` are treated as **data gaps**. Windows that cross such gaps
are dropped, so sequences always represent contiguous data.

Sequence models do not (yet) support MultiIndex data (e.g. ``(asset_id, timestamp)``).

Common configuration: ``sequence_builder``
------------------------------------------

All sequence models (Seq2One and Seq2Seq) use a ``sequence_builder`` block under
``train.autoencoder.params``:

.. code-block:: yaml

   train:
     autoencoder:
       name: LSTMSeq2OneAutoencoder    # or LSTMSeqAutoencoder, CNNAutoencoder, ...
       verbose: 0
       params:
         sequence_builder:
           sequence_length: 36       # window length (number of time steps)
           stride: 1                 # step between window starts
           ts_freq: "30s"            # expected time delta between rows
           pad_incomplete: false
           pad_value: 0.0

         # model-specific params go here (layers, dropout_rate, etc.)

Important points:

- ``sequence_length``: size of each input window, in number of rows.
- ``stride``:
  - ``stride=1`` → heavily overlapping windows,
  - ``stride=sequence_length`` → disjoint windows.
- ``ts_freq``: expected sampling interval. The config parser converts
  strings like ``"30s"``, ``"10m"`` into ``numpy.timedelta64``.
- ``pad_incomplete``: if ``true``, the data is resampled to a regular
  grid at ``ts_freq`` and missing timestamps are filled with ``pad_value``.

Conditional features (all sequence models)
------------------------------------------

All sequence models can use additional **conditional features** that are
fed to the model at each timestep but are **not reconstructed**.

Configure them as:

.. code-block:: yaml

   train:
     autoencoder:
       name: LSTMSeq2OneAutoencoder    # or any sequence model
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

Then add these columns to your DataFrame before training. The model will reconstruct only
the **main** features (all columns that are not listed as conditional).

Example usage (all sequence models)
-----------------------------------

Training and prediction are identical to the dense autoencoder case:

.. code-block:: python

   import pandas as pd
   from energy_fault_detector import FaultDetector, Config

   df = pd.read_csv("my_timeseries.csv",
                    parse_dates=["timestamp"],
                    index_col="timestamp").sort_index()

   sensor_data = df[["power", "wind_speed", "pitch"]]
   normal_index = df["status"] == "normal"

   cfg = Config("lstm_seq2one_config.yaml")    # or a seq2seq config
   fd = FaultDetector(config=cfg)

   model_meta = fd.fit(sensor_data=sensor_data, normal_index=normal_index)

   results = fd.predict(sensor_data=sensor_data)

   print(results.reconstruction.index[:5])
   print(results.reconstruction.shape)


Seq2One models (sequence-to-one)
================================

Seq2One models take a window of length ``sequence_length`` and reconstruct the
**last timestep** of that window:

- :py:class:`LSTMSeq2OneAutoencoder <energy_fault_detector.autoencoders.lstm_seq2one_autoencoder.LSTMSeq2OneAutoencoder>`
- :py:class:`BidirectionalLSTMSeq2OneAutoencoder <energy_fault_detector.autoencoders.bidirectional_lstm_seq2one_autoencoder.BidirectionalLSTMSeq2OneAutoencoder>`
- :py:class:`CNNSeq2OneAutoencoder <energy_fault_detector.autoencoders.cnn_seq2one_autoencoder.CNNSeq2OneAutoencoder>`

Input:
    ``(batch_size, sequence_length, n_main_features)`` [+ conditional features]

Output:
    ``(batch_size, n_main_features)`` (reconstruction of the last timestep)

When to use Seq2One models
--------------------------

Seq2One autoencoders are useful when:

- you care primarily about the **current** (last) timestep,
- you want **causal** models that only look at past data in the window,
- you want a single reconstruction per timestep (aligned to the last timestamp of each window).

Example configuration
---------------------

.. code-block:: yaml

   train:
     autoencoder:
       name: LSTMSeq2OneAutoencoder
       verbose: 0
       params:
         sequence_builder:
           sequence_length: 36
           stride: 1
           ts_freq: "30s"
           pad_incomplete: false
           pad_value: 0.0

         layers: [64, 32]
         code_size: 16
         decoder_layers: [32, 64]
         dropout_rate: 0.0
         regularization: 0.01
         learning_rate: 0.001
         batch_size: 32
         epochs: 15
         loss_name: "mean_squared_error"
         metrics: ["mean_absolute_error"]
         early_stopping: false
         noise: 0.0

Bidirectional encoder variant
-----------------------------

The bidirectional variant uses the same configuration, plus ``merge_mode``:

.. code-block:: yaml

   train:
     autoencoder:
       name: BidirectionalLSTMSeq2OneAutoencoder
       params:
         sequence_builder:
           sequence_length: 36
           stride: 1
           ts_freq: "30s"
         layers: [64, 32]
         merge_mode: "sum"   # "concat", "sum", "ave", or "mul"

The ``merge_mode`` controls how forward and backward encoder outputs are merged. The
recommended setting is ``"sum"``.


Seq2Seq models (sequence-to-sequence)
=====================================

Seq2Seq autoencoders reconstruct the **entire input window**:

- :py:class:`LSTMSeqAutoencoder <energy_fault_detector.autoencoders.lstm_seq_autoencoder.LSTMSeqAutoencoder>`
- :py:class:`CNNAutoencoder <energy_fault_detector.autoencoders.cnn_autoencoder.CNNAutoencoder>`

Input:
    ``(batch_size, sequence_length, n_main_features)`` [+ conditional features]

Output:
    ``(batch_size, sequence_length, n_main_features)`` (reconstruction of the full window)

When to use Seq2Seq models
--------------------------

Seq2Seq autoencoders are useful when:

- you want to detect anomalies **anywhere within a window**, not just at the last timestep,
- you care about temporal *patterns* (e.g. “rise then fall”) rather than point-wise deviations,
- you want a richer reconstruction error signal (one value per feature and timestep) to localise
  *when* within a window the anomaly occurs.

Seq2One vs Seq2Seq
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Seq2One
     - Seq2Seq
   * - Output shape
     - ``(batch, n_features)``
     - ``(batch, sequence_length, n_features)``
   * - Anomaly localisation
     - Single point (last timestep)
     - Per-timestep within window
   * - Overlap handling
     - Each timestep gets one prediction
     - Overlapping predictions are averaged
   * - Training cost
     - Lower (smaller decoder)
     - Higher (decoder mirrors encoder)

Configuration example (LSTMSeqAutoencoder)
------------------------------------------

Seq2Seq models use the same ``sequence_builder`` block:

.. code-block:: yaml

   train:
     autoencoder:
       name: LSTMSeqAutoencoder
       verbose: 0
       params:
         sequence_builder:
           sequence_length: 36
           stride: 1
           ts_freq: "30s"
           pad_incomplete: false
           pad_value: 0.0

         layers: [128, 64, 32]
         dropout_rate: 0.1
         regularization: 0.01
         learning_rate: 0.001
         batch_size: 32
         epochs: 20
         loss_name: "mean_squared_error"
         early_stopping: false
         noise: 0.0

CNN seq2seq variant
-------------------

The CNN autoencoder uses symmetric ``Conv1D`` / ``Conv1DTranspose`` stacks with batch
normalisation:

.. code-block:: yaml

   train:
     autoencoder:
       name: CNNAutoencoder
       params:
         sequence_builder:
           sequence_length: 36
           stride: 6
           ts_freq: "5m"

         layers: [128, 64, 32]
         kernel_size: 3
         strides: 1
         dropout_rate: 0.0
         learning_rate: 0.001
         batch_size: 128
         epochs: 15
         loss_name: "mean_squared_error"

Key parameters:

- ``layers``: Number of filters per Conv1D layer in the encoder. The decoder mirrors these
  layers in reverse. The last number of filters effectively determines the latent dimension
  (per timestep).
- ``kernel_size``: Width of the 1D convolution window.
- ``strides``: Stride of the convolution (usually ``1`` with ``padding="same"`` to preserve
  sequence length).

Overlap averaging
-----------------

When ``stride < sequence_length``, consecutive windows overlap. During prediction, each
timestep may appear in multiple output windows. The seq2seq base class **averages** overlapping
predictions to produce a single value per timestep, resulting in a smoother reconstruction signal.

Example usage (Seq2Seq)
-----------------------

.. code-block:: python

   import pandas as pd
   from energy_fault_detector import FaultDetector, Config

   df = pd.read_csv("my_timeseries.csv",
                    parse_dates=["timestamp"],
                    index_col="timestamp").sort_index()

   sensor_data = df[["power", "wind_speed", "pitch"]]
   normal_index = df["status"] == "normal"

   cfg = Config("lstm_seq2seq_config.yaml")
   fd = FaultDetector(config=cfg)

   model_meta = fd.fit(sensor_data=sensor_data, normal_index=normal_index)

   results = fd.predict(sensor_data=sensor_data)

   # Reconstruction has one row per timestep (overlaps averaged):
   print(results.reconstruction.shape)
   print(results.reconstruction.index[:5])