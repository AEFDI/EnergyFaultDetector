
from typing import Optional, Dict, List

import logging
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore

logger = logging.getLogger('energy_fault_detector')


class WeightedRMSEScore(RMSEScore):
    """Calculate the RMSE of given reconstruction errors manipulated with specified weights.
    
        Attributes:
            feature_weights (Dict[str, float]): Weight definition for input features. 
            Weights can be chosen from [0, infinity). Defaults to None.
    
        Configuration example:
    
        .. code-block:: yaml
    
            train:
              anomaly_score:
                name: weighted_rmse
                params:
                  feature_weights:
                    feature_1: 2.0
                    feature_2: 0.5  
                    feature_3: 1.0  # If a feature should not be weighted it can also be omitted.
    
        """

    def __init__(self, feature_weights: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(**kwargs)
        self.feature_weights = {} if feature_weights is None else feature_weights
        if self.feature_weights:
            if np.min(list(self.feature_weights.values())) < 0:
                raise ValueError('WeightedRMSEScore does not accept negative feature weights. ' \
                                    'If you want to decrease the importance '
                                    'of a feature in the AnomalyScore, choose a weight x with 0<= x < 1.')

    def scale_with_std(self, x: pd.DataFrame) -> pd.DataFrame:
        if np.all(self.std_x_ > 0):
                x_ = x / self.std_x_
        x_[np.isinf(x_)] = 0
        return x_

    def apply_weights_to_squared_residuals(self, x: pd.DataFrame) -> pd.DataFrame:
        """ Applies specified weights to squared standardized x, if x is a pandas DataFrame and the features acutally occur in x's columns.

        x (pd.DataFrame): DataFrame of reconstruction errors.
        Returns:
                pd.DataFrame weighted version of x.
        """
        # Standardize reconstruction errors to remove potential model bias towards specific features
        # x_squared = pd.DataFrame(data=super().standardize(x), columns=x.columns, index=x.index) ** 2
        x_squared = pd.DataFrame(data=self.scale_with_std(x), columns=x.columns, index=x.index) ** 2

        # Weight standardized reconstruction errors to introduce useful application context bias
        for feature in self.feature_weights:
            if feature in x.columns:
                x_squared[feature] = x_squared[feature] * self.feature_weights[feature]
            else:
                logger.warning(f'Specified feature {feature} is not part of the list of input '
                                f'features. Thus it can not be weighted. Input features: {x.columns}.')
        return x_squared

    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> 'WeightedRMSEScore':
        """Calculate standard deviation and mean on weighted training data
        
        Args:
            x (pd.DataFrame): DataFrame with differences between prediction and actual sensor values
            y (Optional[pd.Series]): not used, labels indicating whether sample is normal (True) or anomalous (False).
            variable_name_features (Dict[str, List[str]]): Mapping of input feature names that change during preprocessing. 
                        Example: Angle features might be transformed into [feature_name_sin, feature_name_cos. Defaults to an empty dictionary.
        """
        if not isinstance(x, pd.DataFrame):
                            raise ValueError('WeightedRMSEScore requires a DataFrame as input to correctly ' \
                            'apply specified weights.')

        # Standardize reconstruction errors to remove potential model bias towards specific features
        super().fit(x, y)
        self.fitted_ = True
        return self


    def transform(self, x: pd.DataFrame):
        """Calculate the RMSE based on the weighted deviation matrix.
        
        Args:
            x (pd.DataFrame): Dataframe with differences between prediction and actual sensor values

        Returns:
            weighted RMSE for each sample.
        """
        if not isinstance(x, pd.DataFrame):
                                    raise ValueError('WeightedRMSEScore requires a DataFrame as input to correctly ' \
                                    'apply specified weights.')

        x_squared_weighted = self.apply_weights_to_squared_residuals(x)

        total_feature_weights = np.sum(list(self.feature_weights.values()))
        total_feature_weights += x.shape[1] - len(self.feature_weights)  # add omitted features with weight 1.0
        scores = np.sqrt(np.sum(x_squared_weighted, axis=1) / total_feature_weights)
        if isinstance(x, (pd.DataFrame, pd.Series)):
            scores = pd.Series(scores, index=x.index)
        return scores
