"""The anomaly-detection-iee package"""

import os

from energy_fault_detector.registration import registry, register

# Register models and other classes
# class types: autoencoder, anomaly_score, threshold_selector

# autoencoders
register(module_path='energy_fault_detector.autoencoders.MultilayerAutoencoder',
         class_type='autoencoder',
         class_names=['MultilayerAutoencoder', 'multilayer_ae', 'multilayer_autoencoder', 'default', 'dense'])
register(module_path='energy_fault_detector.autoencoders.ConditionalAE',
         class_type='autoencoder',
         class_names=['ConditionalAE', 'ConditionalAutoencoder', 'conditional_ae', 'conditional',
                      'conditional_autoencoder'])

# scores
register(module_path='energy_fault_detector.anomaly_scores.MahalanobisScore',
         class_type='anomaly_score',
         class_names=['MahalanobisScore', 'Mahalanobis', 'mahalanobis'])
register(module_path='energy_fault_detector.anomaly_scores.RMSEScore',
         class_type='anomaly_score',
         class_names=['RMSEScore', 'RMSE', 'rmse'])

# threshold selectors
register(module_path='energy_fault_detector.threshold_selectors.FDRSelector',
         class_type='threshold_selector',
         class_names=['FDRSelector', 'FDR_selector', 'fdr_selector', 'FDR', 'fdr'])
register(module_path='energy_fault_detector.threshold_selectors.FbetaSelector',
         class_type='threshold_selector',
         class_names=['FbetaSelector', 'Fbeta_selector', 'fbeta_selector', 'fbeta'])
register(module_path='energy_fault_detector.threshold_selectors.QuantileThresholdSelector',
         class_type='threshold_selector',
         class_names=['QuantileThresholdSelector', 'quantile_selector', 'quantile'])
register(module_path='energy_fault_detector.threshold_selectors.AdaptiveThresholdSelector',
         class_type='threshold_selector',
         class_names=['AdaptiveThresholdSelector', 'adaptive_threshold', 'SVR', 'svr', 'adaptive'])

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, 'VERSION'), 'r', encoding='utf-8') as f:
    version = f.readlines()[0].strip()

__version__ = version
