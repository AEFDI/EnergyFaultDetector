
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
        if feature_weights is not None:
            if np.min(list(feature_weights.values())) < 0:
                raise ValueError('WeightedRMSEScore does not accept negative feature weights. ' \
                                 'If you want to decrease the importance '
                                 'of a feature in the AnomalyScore, choose a weight x with 0<= x < 1.')
        # `feature_weights` is restored from the pickled state on load, so it is allowed to be
        # None only temporarily (e.g. when the scorer is constructed with no arguments by the
        # load mechanism). The actual weights must be provided before `fit` or `transform` are called.
        self.feature_weights = feature_weights

    def standardize_weighted(self, x: pd.DataFrame):
        """Standardization of the weighted reconstruction error in x"""

        check_is_fitted(self)
        x_ = x.copy()
        if np.all(self.std_x_ > 0):
            x_ = (x - self.mean_x_weighted_) / self.std_x_weighted_
        else:
            x_ = x - self.mean_x_weighted_
        # replace possible inf values with 0
        x_[np.isinf(x_)] = 0
        return x_

    def apply_weights(self, x: pd.DataFrame) -> pd.DataFrame:
        """ Applies specified weights to x, if x is a pandas DataFrame and the features acutally occur in x's columns.

        x (pd.DataFrame): DataFrame of reconstruction errors.
        Returns:
                pd.DataFrame weighted verison of x.
        """
        # Standardize reconstruction errors to remove potential model bias towards specific features
        x_ = pd.DataFrame(data=super().standardize(x), columns=x.columns, index=x.index)

        # Weight standardized reconstruction errors to introduce useful application context bias
        for feature in self.feature_weights:
            if feature in x.columns:
                x_[feature] = x[feature] * self.feature_weights[feature]
            else:
                logger.warning(f'Specified feature {feature} is not part of the list of input '
                                'features. Thus it can not be weighted. Input features: {x.columns}.')
        return x_

    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> 'WeightedRMSEScore':
        """Calculate standard deviation and mean on weighted training data
        
        Args:
            x (pd.DataFrame): DataFrame with differences between prediction and actual sensor values
            y (Optional[pd.Series]): not used, labels indicating whether sample is normal (True) or anomalous (False).
            variable_name_features (Dict[str, List[str]]): Mapping of input feature names that change during preprocessing. 
                        Example: Angle features might be transformed into [feature_name_sin, feature_name_cos. Defaults to an empty dictionary.
        """
        if not isinstance(x, pd.DataFrame):
                            raise ValueError('WeightedRMSEScore requires a DataFrame as input to correctly apply ' \
                            'specified weights.')

        # Standardize reconstruction errors to remove potential model bias towards specific features
        super().fit(x, y)
        x_weighted = self.apply_weights(x)
        # Compute mean and std of weighted reconstruction errors for standardization in 
        self.mean_x_weighted_ = x_weighted.mean()
        self.std_x_weighted_ = x_weighted.std()
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
                                    raise ValueError('WeightedRMSEScore requires a DataFrame as input to correctly apply ' \
                                    'specified weights.')

        x_weighted = self.apply_weights(x)
        x_standardized = self.standardize_weighted(x_weighted)
        
        scores = np.sqrt(np.mean(x_standardized ** 2, axis=1))
        if isinstance(x, (pd.DataFrame, pd.Series)):
            scores = pd.Series(scores, index=x.index)
        return scores
