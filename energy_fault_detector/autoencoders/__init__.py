"""Autoencoder normal-behaviour models trained to reconstruct normal operation.

Large reconstruction errors at prediction time indicate anomalies. Models range from dense
(point-wise) autoencoders to sequence models (seq2one and seq2seq) for time-series input.

Every autoencoder model here subclasses :class:`~energy_fault_detector.core.autoencoder.Autoencoder`
and is referenced from the configuration under the ``train.autoencoder`` key.
"""

# Dense
from .multilayer_autoencoder import MultilayerAutoencoder
from .conditional_autoencoder import ConditionalAE
# Seq2One (half-autoencoder)
from .lstm_seq2one_autoencoder import LSTMSeq2OneAutoencoder
from .cnn_seq2one_autoencoder import CNNSeq2OneAutoencoder
from .bidirectional_lstm_seq2one_autoencoder import BidirectionalLSTMSeq2OneAutoencoder
# Seq2seq
from .cnn_seq_autoencoder import CNNAutoencoder
from .lstm_seq2seq_autoencoder import LSTMSeqAutoencoder

__all__ = [
    "MultilayerAutoencoder",
    "ConditionalAE",
    "LSTMSeq2OneAutoencoder",
    "CNNSeq2OneAutoencoder",
    "BidirectionalLSTMSeq2OneAutoencoder",
    "CNNAutoencoder",
    "LSTMSeqAutoencoder",
]
