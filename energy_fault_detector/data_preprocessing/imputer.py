from typing import Optional, List, Union, Callable
import pandas as pd
from dask.dataframe.dispatch import categorical_dtype
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer
from statsmodels.tools import categorical

from energy_fault_detector.core.data_transformer import DataTransformer
import logging

logger = logging.getLogger('energy_fault_detector')


class Imputer(DataTransformer):
    """
    Class containing the imputation step. It is a wrap around methods for
    imputation of numerical and categorical features.
    """

    def __init__(self, strategy: str = 'mean', categorical_features: list = None, **params):
        super().__init__()
        self.strategy = strategy
        self.categorical_features = categorical_features if categorical_features else []
        # Initialize nested estimators
        self.params = params
        # Separate imputers for numerical and categorical data
        self.numerical_imputer = SimpleImputer(strategy=self.strategy, **self.params)
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')  # Categorical default to 'most_frequent'

        # Attributes to be defined during fitting
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.input_index_ = None
        self.numerical_columns = None
        self.categorical_columns = None

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fits the imputer on the provided DataFrame by separately handling numerical
        and categorical columns.
        """
        self.feature_names_in_ = x.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.input_index_ = x.index

        # Split data into numerical (and boolean) and categorical features
        self.numerical_columns = [col for col in self.feature_names_in_ if not any(feature in col for feature in self.categorical_features)]
        self.categorical_columns = [col for col in self.feature_names_in_ if any(feature in col for feature in self.categorical_features)]
        numerical_data = x[self.numerical_columns]
        categorical_data = x[self.categorical_columns]

        # Fit the imputers
        self.numerical_imputer.fit(numerical_data)
        if not categorical_data.empty:
            self.categorical_imputer.fit(categorical_data)

        return self
    
    def transform(self, x: pd.DataFrame):
        """
        Transforms the DataFrame by imputing missing values for numerical and
        categorical columns separately. Rejoins the transformed dataframes afterward.
        """
        check_is_fitted(self)

        # Separate data into numerical and categorical features
        # TODO: what happens if x doesnt have all columns as during fit?
        numerical_data = x[self.numerical_columns]
        categorical_data = x[self.categorical_columns]

        # Transform the data
        numerical_transformed = pd.DataFrame(
            self.numerical_imputer.transform(numerical_data),
            columns=self.numerical_columns,
            index=x.index
        )
        if not categorical_data.empty:
            categorical_transformed = pd.DataFrame(
                self.categorical_imputer.transform(categorical_data),
                columns=self.categorical_columns,
                index=x.index
            )
        else:
            categorical_transformed = pd.DataFrame(index=x.index)  # Empty DataFrame for consistency

        # Concatenate numerical and categorical data back together
        self.feature_names_out_ = self.get_feature_names_out()
        transformed_data = pd.concat([numerical_transformed, categorical_transformed], axis=1)
        transformed_data = transformed_data[self.feature_names_out_]  # Ensure original column order

        return transformed_data
    
    def inverse_transform(self, x: pd.DataFrame):
        """
        For compatibility, this method returns the DataFrame with the original column
        order but assumes no special inversions are necessary after imputation.
        """
        check_is_fitted(self)
        return pd.DataFrame(x, columns=self.feature_names_in_, index=self.input_index_)
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_in_