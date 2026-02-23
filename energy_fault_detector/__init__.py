"""The anomaly-detection-iee package"""

from .__about__ import __version__
from energy_fault_detector.core._logs import setup_logging
from pathlib import Path

# Setup logging
setup_logging(Path(__file__).parent / 'logging.yaml')

from energy_fault_detector.registration import registry, register
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.config import Config
from energy_fault_detector.quick_fault_detection import quick_fault_detector

# Register models and other classes
# class types: autoencoder, anomaly_score, threshold_selector

# autoencoders
register(module_path='energy_fault_detector.autoencoders.multilayer_autoencoder.MultilayerAutoencoder',
         class_type='autoencoder',
         class_names=['MultilayerAutoencoder', 'multilayer_ae', 'multilayer_autoencoder', 'default', 'dense'])
register(module_path='energy_fault_detector.autoencoders.conditional_autoencoder.ConditionalAE',
         class_type='autoencoder',
         class_names=['ConditionalAE', 'ConditionalAutoencoder', 'conditional_ae', 'conditional',
                      'conditional_autoencoder'])
register(module_path='energy_fault_detector.autoencoders.lstm_seq2one_autoencoder.LSTMSeq2OneAutoencoder',
         class_type='autoencoder',
         class_names=['LSTMSeq2OneAutoencoder', 'LSTMSeq2One', 'lstm_seq2one', 'lstm_seq2one_autoencoder',
                      'lstm'])
register(module_path='energy_fault_detector.autoencoders.cnn_seq2one_autoencoder.CNNSeq2OneAutoencoder',
         class_type='autoencoder',
         class_names=['CNNSeq2OneAutoencoder', 'CNNSeq2One', 'cnn_seq2one', 'cnn_seq2one_autoencoder',
                      'cnn'])

# scores
register(module_path='energy_fault_detector.anomaly_scores.mahalanobis_score.MahalanobisScore',
         class_type='anomaly_score',
         class_names=['MahalanobisScore', 'Mahalanobis', 'mahalanobis'])
register(module_path='energy_fault_detector.anomaly_scores.rmse_score.RMSEScore',
         class_type='anomaly_score',
         class_names=['RMSEScore', 'RMSE', 'rmse'])

# threshold selectors
register(module_path='energy_fault_detector.threshold_selectors.fdr_threshold.FDRSelector',
         class_type='threshold_selector',
         class_names=['FDRSelector', 'FDR_selector', 'fdr_selector', 'FDR', 'fdr'])
register(module_path='energy_fault_detector.threshold_selectors.fbeta_threshold.FbetaSelector',
         class_type='threshold_selector',
         class_names=['FbetaSelector', 'Fbeta_selector', 'fbeta_selector', 'fbeta'])
register(module_path='energy_fault_detector.threshold_selectors.quantile_threshold.QuantileThresholdSelector',
         class_type='threshold_selector',
         class_names=['QuantileThresholdSelector', 'quantile_selector', 'quantile'])
register(module_path='energy_fault_detector.threshold_selectors.adaptive_threshold.AdaptiveThresholdSelector',
         class_type='threshold_selector',
         class_names=['AdaptiveThresholdSelector', 'adaptive_threshold', 'SVR', 'svr', 'adaptive'])
