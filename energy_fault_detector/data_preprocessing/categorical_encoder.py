from typing import Optional, List, Union, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from energy_fault_detector.core.data_transformer import DataTransformer


class CategoricalEncoder(DataTransformer):
    """
    CategoricalEncoder is a specialized transformer for encoding categorical features in a dataset.

    This class is designed to handle preprocessing of datasets by encoding specified categorical
    features using one-hot encoding while maintaining numerical features unchanged. It provides
    methods to fit to the dataset, transform it into an encoded format, inversely transform
    encoded data back to the original format, and retrieve the transformed feature names.
    It assumes the input data will be in the form of a DataFrame.

    It can be used in preprocessing pipelines for machine learning models.

    Attributes:
        categorical_features (list): A list of strings representing the categorical feature names to be one-hot encoded.
                                     If not provided, the encoder will not encode any categorical features.
        feature_names_out_ (list): The list of output feature names after transformation.
        numerical_columns (list): The list of numerical feature names identified from the input data.
        n_features_in_ (int): The total number of input features in the dataset.
        feature_names_in_ (list): The list of input feature names identified during the fit process.
        one_hot_encoder (OneHotEncoder): An instance of `OneHotEncoder` used for transforming categorical features.
        categorical_columns (list): The list of categorical columns identified from the input data.

    Methods:
        fit(x: pd.DataFrame, y=None):
            Fits the OneHotEncoder to the categorical features in the provided DataFrame.

            Args:
                x (pd.DataFrame): The input data containing all features.
                                  Only the specified categorical features will be fitted.
                y: Ignored. Retained for compatibility with scikit-learn API.

            Returns:
                self: The fitted CategoricalEncoder instance.

        transform(x: pd.DataFrame) -> pd.DataFrame:
            Transforms the input DataFrame by applying one-hot encoding to categorical features
            and combining them with numerical features.

            Args:
                x (pd.DataFrame): The input data to be transformed.

            Returns:
                pd.DataFrame: The transformed DataFrame with one-hot encoded categorical features
                              and numerical features.

            Raises:
                KeyError: If the input DataFrame is missing any of the features fitted during the `fit` step.

        inverse_transform(x: pd.DataFrame) -> pd.DataFrame:
            Reverts the transformed data back to its original form by mapping one-hot encoded
            categorical features back to the original categorical values.

            Args:
                x (pd.DataFrame): The transformed input data.

            Returns:
                pd.DataFrame: The original data with categorical features restored to their original values.

        get_feature_names_out(input_features=None) -> list:
            Returns the names of features after the transformation.

            Args:
                input_features (list, optional): Unused. Retained for compatibility with scikit-learn API.

            Returns:
                list: List of output feature names including both numerical and one-hot encoded features.

    Example:
        ```python
        import pandas as pd
        from categorical_encoder import CategoricalEncoder

        # Example dataset
        data = pd.DataFrame({
            'Category1': ['A', 'B', 'A'],
            'Category2': ['X', 'Y', 'Z'],
            'Numerical': [1, 2, 3]
        })

        # Initialize and fit the encoder
        encoder = CategoricalEncoder(categorical_features=['Category1', 'Category2'])
        encoder.fit(data)

        # Transform the data
        transformed_data = encoder.transform(data)
        print(transformed_data)

        # Inverse transform the data
        original_data = encoder.inverse_transform(transformed_data)
        print(original_data)

    """
    def __init__(self, categorical_features: list = None):
        super().__init__()
        self.feature_names_out_ = None
        self.numerical_columns = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.categorical_features = categorical_features if categorical_features else []
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.categorical_columns = None

    def fit(self, x: pd.DataFrame, y=None) -> "CategoricalEncoder":
        # TODO: non-numerical features that are not specified in categorical_features should bbe dropped and warning should be raised
        self.feature_names_in_ = x.columns.tolist()
        self.categorical_columns = [col for col in self.feature_names_in_ if any(feature in col for feature in self.categorical_features)]
        self.numerical_columns = [col for col in self.feature_names_in_ if col not in self.categorical_columns]
        self.n_features_in_ = len(self.feature_names_in_)
        categorical_data = x[self.categorical_columns]

        # Do the one-hot-encode on categorical data
        self.one_hot_encoder.fit(categorical_data)
        return self
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)

        # Separate data into numerical and categorical features. Only categorical features are transformed
        # TODO: what happens if x doesnt have all columns as during fit?
        try:
            numerical_data = x[self.numerical_columns]
            categorical_data = x[self.categorical_columns]
        except KeyError:
            raise KeyError(f"The features {self.feature_names_in_} are not present in the input data.")

        if not categorical_data.empty:
            x_categorical_ = self.one_hot_encoder.transform(categorical_data)
            x_numerical_ = numerical_data.values
            x_ = np.concatenate((x_numerical_, x_categorical_), axis=1)
            self.feature_names_out_ = self.get_feature_names_out(self.feature_names_in_)
            transformed_data = pd.DataFrame(x_, columns=self.feature_names_out_, index=x.index)
            return transformed_data
        else:
            return numerical_data  # Returns input df if no categorical features are specified in config file

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)

        # Get the one-hot encoded column names from the encoder
        encoded_categorical_columns = list(self.one_hot_encoder.get_feature_names_out(self.categorical_features))

        # Separate numerical columns from one-hot-encoded categorical columns
        categorical_encoded_columns = [col for col in x.columns if col in encoded_categorical_columns]
        numerical_columns = [col for col in x.columns if col not in encoded_categorical_columns]

        numerical_data = x[numerical_columns]

        if categorical_encoded_columns:
            categorical_data = x[categorical_encoded_columns]
            x_categorical_ = self.one_hot_encoder.inverse_transform(categorical_data)
            categorical_original = pd.DataFrame(x_categorical_, columns=self.categorical_columns, index=x.index)
            return pd.concat([numerical_data, categorical_original], axis=1)
        else:
            return numerical_data # Returns input df if no categorical features are specified in config file
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        check_is_fitted(self)
        self.feature_names_out_ = self.numerical_columns + list(self.one_hot_encoder.get_feature_names_out(self.categorical_columns))
        return self.feature_names_out_