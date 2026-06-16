from typing import Optional, List, Union, Callable
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from energy_fault_detector.core.data_transformer import DataTransformer


class CategoricalEncoder(DataTransformer):
    " Class containing the one-hot-code step of categorical features to be used in the pre-pocessing pipeline"
    def __init__(self, categorical_features):
        super().__init__()
        self.categorical_features = categorical_features
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

    def fit(self, x: pd.DataFrame, y=None):
        self.feature_names_in_ = x.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.input_index_ = x.index
        # Do the one-hot-encode
        self.one_hot_encoder.fit(x)
        return self
    
    def transform(self, x: pd.DataFrame):
        check_is_fitted(self)
        x_ = self.one_hot_encoder.transform(x)
        self.feature_names_out_ = self.get_feature_names_out()
        return pd.DataFrame(x_, columns=self.feature_names_out_, index=self.input_index_)
    
    def inverse_transform(self, x: pd.DataFrame):
        check_is_fitted(self)
        x_ = self.one_hot_encoder.inverse_transform(x)
        return pd.DataFrame(x_, columns=self.feature_names_in_, index=self.input_index_)
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.one_hot_encoder.get_feature_names_out(self.feature_names_in_)
