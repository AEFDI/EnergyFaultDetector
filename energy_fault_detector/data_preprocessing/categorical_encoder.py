from typing import Optional, List, Union, Callable
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from energy_fault_detector.core.data_transformer import DataTransformer


class CategoricalEncoder(DataTransformer):
    " Class containing the one-hot-code step of categorical features to be used in the pre-pocessing pipeline"
    def __init__(self, categorical_features: list = None):
        super().__init__()
        self.categorical_features = categorical_features if categorical_features else []
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.categorical_columns = None

    def fit(self, x: pd.DataFrame, y=None):
        # Only categorical features are fitted and therefore features_names_in_ contain only categorical features
        self.categorical_columns = [col for col in x.columns if any(feature in col for feature in self.categorical_features)]
        self.feature_names_in_ = self.categorical_columns
        self.n_features_in_ = len(self.feature_names_in_)
        categorical_data = x[self.categorical_columns]

        # Do the one-hot-encode on categorical data
        self.one_hot_encoder.fit(categorical_data)
        return self
    
    def transform(self, x: pd.DataFrame):
        check_is_fitted(self)

        # Separate data into numerical and categorical features. Only categorical features are transformed
        # TODO: what happens if x doesnt have all columns as during fit?
        numerical_data = x[[col for col in x.columns if col not in self.categorical_columns]]
        try:
            categorical_data = x[self.categorical_columns]
        except KeyError:
            raise KeyError(f"The categorical features {self.categorical_columns} are not present in the input data.")

        if not categorical_data.empty:
            x_categorical_ = self.one_hot_encoder.transform(categorical_data)
            self.feature_names_out_ = self.get_feature_names_out(self.feature_names_in_)
            categorical_transformed = pd.DataFrame(x_categorical_, columns=self.feature_names_out_, index=categorical_data.index)
            transformed_data = pd.concat([numerical_data, categorical_transformed], axis=1)
            return transformed_data
        else:
            return numerical_data  # Returns input df if no categorical features are specified in config file

    def inverse_transform(self, x: pd.DataFrame):
        check_is_fitted(self)

        # Separate numerical columns from one-hot-encoded categorical columns
        categorical_encoded_columns = [col for col in x.columns if col in self.feature_names_out_]
        numerical_columns = [col for col in x.columns if col not in self.feature_names_out_]

        numerical_data = x[numerical_columns]

        if categorical_encoded_columns:
            categorical_data = x[categorical_encoded_columns]
            x_categorical_ = self.one_hot_encoder.inverse_transform(categorical_data)
            categorical_original = pd.DataFrame(x_categorical_, columns=self.categorical_columns, index=x.index)
            return pd.concat([numerical_data, categorical_original], axis=1)
        else:
            return numerical_data # Returns input df if no categorical features are specified in config file
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.one_hot_encoder.get_feature_names_out(self.feature_names_in_)
