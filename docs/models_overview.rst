.. _models_overview:

Model overview
==============

This page summarises the main model types provided by :mod:`energy_fault_detector`.

For configuration details see :doc:`configuration` and :doc:`sequence_models`.


Autoencoders
------------
Autoencoders learn a model of *normal* behaviour and are configured under ``train.autoencoder`` in the YAML config.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Class
     - Typical config names (``train.autoencoder.name``)
     - Description
   * - :class:`MultilayerAutoencoder <energy_fault_detector.autoencoders.multilayer_autoencoder.MultilayerAutoencoder>`
     - ``"default"``, ``"MultilayerAutoencoder"``
     - Dense symmetric autoencoder for tabular or time-series data treated as independent rows. Good default choice.
   * - :class:`ConditionalAE <energy_fault_detector.autoencoders.conditional_autoencoder.ConditionalAE>`
     - ``"ConditionalAE"``, ``"ConditionalAutoencoder"``
     - Dense autoencoder where selected features are used as *conditions* and not reconstructed (e.g. time-of-day).
   * - :class:`LSTMSeq2OneAutoencoder <energy_fault_detector.autoencoders.lstm_seq2one_autoencoder.LSTMSeq2OneAutoencoder>`
     - ``"LSTMSeq2OneAutoencoder"``, ``"lstm_seq2one"``
     - Sequence-to-one LSTM autoencoder. Uses windows of historical data to reconstruct the last timestep.
   * - :class:`BidirectionalLSTMSeq2OneAutoencoder <energy_fault_detector.autoencoders.bidirectional_lstm_seq2one_autoencoder.BidirectionalLSTMSeq2OneAutoencoder>`
     - ``"BidirectionalLSTMSeq2OneAutoencoder"``, ``"bilstm_seq2one"``
     - Bidirectional LSTM variant of the seq2one model for richer temporal context.
   * - :class:`CNNSeq2OneAutoencoder <energy_fault_detector.autoencoders.cnn_seq2one_autoencoder.CNNSeq2OneAutoencoder>`
     - ``"CNNSeq2OneAutoencoder"``, ``"cnn_seq2one"``
     - CNN-based seq2one autoencoder. Efficient, focuses on local temporal patterns.
   * - :class:`LSTMSeqAutoencoder <energy_fault_detector.autoencoders.lstm_seq2seq_autoencoder.LSTMSeqAutoencoder>`
     - ``"LSTMSeqAutoencoder"``, ``"lstm_seq2seq"``
     - Sequence-to-sequence LSTM autoencoder. Reconstructs the full window, useful when localisation within the window matters.
   * - :class:`CNNAutoencoder <energy_fault_detector.autoencoders.cnn_seq_autoencoder.CNNAutoencoder>`
     - ``"CNNAutoencoder"``, ``"cnn_seq2seq"``
     - Sequence-to-sequence CNN autoencoder using Conv1D/Conv1DTranspose layers.

All sequence models require a ``sequence_builder`` block in
``train.autoencoder.params``. See :doc:`sequence_models` for details and examples.


Anomaly scores
--------------

Anomaly score classes map reconstruction errors to a scalar anomaly score per sample.
They are configured under ``train.anomaly_score``.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Class
     - Typical config names (``train.anomaly_score.name``)
     - Description
   * - :class:`RMSEScore <energy_fault_detector.anomaly_scores.rmse_score.RMSEScore>`
     - ``"rmse"``, ``"RMSEScore"``
     - Root mean squared error over reconstruction errors. Default choice.
   * - :class:`MahalanobisScore <energy_fault_detector.anomaly_scores.mahalanobis_score.MahalanobisScore>`
     - ``"mahalanobis"``, ``"MahalanobisScore"``
     - Mahalanobis distance on (optionally PCA-transformed) reconstruction errors. Captures feature correlations.


Threshold selectors
-------------------

Threshold selectors map anomaly scores to boolean anomaly decisions and are configured under ``train.threshold_selector``.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Class
     - Typical config names (``train.threshold_selector.name``)
     - Description
   * - :class:`QuantileThresholdSelector <energy_fault_detector.threshold_selectors.quantile_threshold.QuantileThresholdSelector>`
     - ``"quantile"``
     - Sets the threshold to a fixed quantile of the (normal) anomaly scores. Works without labels.
   * - :class:`FbetaSelector <energy_fault_detector.threshold_selectors.fbeta_threshold.FbetaSelector>`
     - ``"fbeta"``, ``"FbetaSelector"``
     - Chooses the threshold that maximises the F-beta score (requires labels / ``normal_index``).
   * - :class:`FDRSelector <energy_fault_detector.threshold_selectors.fdr_threshold.FDRSelector>`
     - ``"fdr"``, ``"FDRSelector"``
     - Chooses the threshold to match a target false discovery rate (requires labels / ``normal_index``).
   * - :class:`AdaptiveThresholdSelector <energy_fault_detector.threshold_selectors.adaptive_threshold.AdaptiveThresholdSelector>`
     - ``"adaptive_threshold"``, ``"AdaptiveThresholdSelector"``
     - Learns an input-dependent threshold using a small regression NN on autoencoder inputs and scores.

For a full reference of model parameters, see the API docs:

- :mod:`energy_fault_detector.autoencoders`
- :mod:`energy_fault_detector.anomaly_scores`
- :mod:`energy_fault_detector.threshold_selectors`
