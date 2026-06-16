from typing import Optional, List, Union, Callable
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from energy_fault_detector.core.data_transformer import DataTransformer

class Imputer(DataTransformer):
    " Class containing the imputation step of categorical features to be used in the pre-pocessing pipeline"
    def __init__(self, strategy: str = 'most_frequent'):
        super().__init__()
        self.strategy = strategy

    def fit(self, x: pd.DataFrame, y=None):
        self.feature_names_in_ = x.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.input_index_ = x.index
        # Impute the most frequent value for each column
        self.fill_values_ = x.mode().iloc[0]
        return self
    
    def transform(self, x: pd.DataFrame):
        check_is_fitted(self)
        x_ = x.fillna(self.fill_values_)
        self.feature_names_out_ = self.get_feature_names_out()
        return pd.DataFrame(x_, columns=self.feature_names_out_, index=self.input_index_)
    
    def inverse_transform(self, x: pd.DataFrame):
        check_is_fitted(self)
        x_ = x
        return pd.DataFrame(x_, columns=self.feature_names_in_, index=self.input_index_)
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_in_