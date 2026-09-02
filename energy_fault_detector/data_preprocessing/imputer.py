from typing import Optional, List, Union, Callable
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer

from energy_fault_detector.core.data_transformer import DataTransformer
import logging

logger = logging.getLogger('energy_fault_detector')

class Imputer(DataTransformer):
    """Wrapper around scikit-learn's SimpleImputer to handle numerical and categorical features separately.

    Supports imputation via 'mean' (for numerical) and 'most_frequent' (for categorical) strategies.
    Automatically detects and excludes non-declared categorical features (object dtype in numerical slots).
    Preserves column order and index through fit/transform/inverse_transform.

    Attributes:
        strategy (str): Imputation strategy for numerical features ('mean' or 'median').
        categorical_features (List[str]): List of column names to treat as categorical (matched by exact name).
        params (dict): Additional keyword arguments passed to `SimpleImputer` for numerical imputation.
        numerical_imputer (SimpleImputer): Imputer instance for numerical features.
        categorical_imputer (SimpleImputer): Imputer instance for categorical features (always 'most_frequent').
        n_features_in_ (int): Number of features seen during `fit`.
        feature_names_in_ (List[str]): Names of features seen during `fit`.
        feature_names_out_ (List[str]): Names of features output after `transform`.
        input_index_ (pd.Index): Index of input DataFrame during `fit`.
        numerical_columns (List[str]): Column names identified as numerical.
        categorical_columns (List[str]): Column names identified as categorical.
        non_declared_categorical_features (List[str]): Object-type columns detected in numerical slots and excluded.
    """

    def __init__(self, strategy: str = 'mean', categorical_features: list = None, **params):
        """Initializes the Imputer with specified strategy and categorical feature handling.

        Args:
            strategy (str, optional): Imputation strategy for numerical features. Supports 'mean' (default) and 'median'.
                Categorical features are always imputed using 'most_frequent'.
            categorical_features (List[str], optional): List of column names to treat as categorical.
                Columns matching any of these (by exact name) are treated as categorical; others as numerical.
                Non-declared object-type columns in numerical slots are automatically detected and excluded.
                Defaults to None (interpreted as empty list).
            **params: Additional keyword arguments passed to `SimpleImputer` for numerical imputation.
                Example: `{'add_indicator': True}`.

        Raises:
            ValueError: If `strategy` is not supported ('mean' or 'median').
        """
        super().__init__()
        self.strategy = strategy
        self.categorical_features = categorical_features if categorical_features else []
        self.params = params
        # Initialize nested estimators
        # Separate imputers for numerical and categorical data
        if self.strategy == 'mean':
            self.numerical_imputer = SimpleImputer(strategy=self.strategy, **self.params)
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')  # Categorical default to 'most_frequent'
        elif self.strategy == 'median':
            self.numerical_imputer = SimpleImputer(strategy=self.strategy, **self.params)
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')  # Categorical default to 'most_frequent'
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}. Supported strategies are 'mean' and 'median'.")

        # Attributes to be defined during fitting
        
        self.feature_names_in_: List[str] = []
        self.feature_names_out_ = None
        self.input_index_ = None
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.non_declared_categorical_features: List[str] = []

    def fit(self, x: pd.DataFrame, y=None) -> "Imputer":
        """Fits the imputer separately for numerical and categorical features.

        Identifies and separates features by type, logs warnings for non-declared categorical columns,
        and fits `SimpleImputer` instances on their respective subsets.

        Args:
            x (pd.DataFrame): Input feature DataFrame.
            y (optional): Target variable. Ignored; included for scikit-learn API compatibility.

        Returns:
            Imputer: The fitted imputer instance (self), for method chaining.

        Raises:
            TypeError: If `x` is not a pandas DataFrame.
            ValueError: If numerical columns cannot be safely converted (via downstream usage).
        """

        self.feature_names_in_ = x.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.input_index_ = x.index

        # Split data into numerical (and boolean) and categorical features
        self.numerical_columns = [col for col in self.feature_names_in_
                                  if col not in self.categorical_features]
        self.categorical_columns = [col for col in self.feature_names_in_
                                    if col in self.categorical_features]
        numerical_data = x.loc[:, self.numerical_columns]
        categorical_data = x.loc[:, self.categorical_columns]

        # Clean numerical columns from non_declared categorical features
        self.non_declared_categorical_features = numerical_data.select_dtypes(include='object').columns.tolist()
        self.numerical_columns = [col for col in self.numerical_columns if col not in self.non_declared_categorical_features]
        numerical_data = numerical_data.loc[:, self.numerical_columns]
        if self.non_declared_categorical_features:
            logger.warning(f"Non-declared categorical features found in data: {self.non_declared_categorical_features}. "
                        f"They will be dropped. Consider adding them to the categorical_features list if they should be"
                        f" treated as categorical.")

        logger.debug(f"Numerical columns: {self.numerical_columns}")
        logger.debug(f"Categorical columns: {self.categorical_columns}")

        # Fit the imputers
        self.numerical_imputer.fit(numerical_data)
        if not categorical_data.empty:
            self.categorical_imputer.fit(categorical_data)

        return self
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Applies imputation to input DataFrame, handling numerical and categorical columns separately.

        Steps:
        1. Validates presence of all training features.
        2. Harmonizes numerical column dtypes to float.
        3. Applies `numerical_imputer` and `categorical_imputer` independently.
        4. Reconcatenates and reorders columns to match original feature order.

        Args:
            x (pd.DataFrame): Input feature DataFrame, with same columns as used in `fit()`.

        Returns:
            pd.DataFrame: Imputed DataFrame with same column names and original index.

        Raises:
            ValueError: If input is missing columns seen during `fit`.
            ValueError: If a numerical column cannot be converted to float.
            ValueError: If imputer has not been fitted (via `check_is_fitted`).
        """
        check_is_fitted(self, "n_features_in_")
        missing_columns = [col for col in self.feature_names_in_ if col not in x.columns]
        if missing_columns:
            raise ValueError(f"Input is missing columns seen during fit: {missing_columns}")

        # Harmonize data types in numerical columns to avoid issues during concatenation after one-hot encoding
        x = x.copy()  # Avoid modifying the original DataFrame
        for col in self.numerical_columns:
            if col in x.columns:
                try:
                    x[col] = x[col].astype(float, errors='raise')
                except ValueError as e:
                    raise ValueError(f"Column '{col}' cannot be converted to float.") from e

        # Separate data into numerical and categorical features
        numerical_data = x.loc[:, self.numerical_columns]
        categorical_data = x.loc[:, self.categorical_columns]

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
    
    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Returns input DataFrame with columns reordered to match original feature order.

        This method preserves the index of the input `x` and does **not** reverse imputation (since original
        NaN positions are not stored). Included for scikit-learn compatibility.

        Args:
            x (pd.DataFrame): Transformed DataFrame (output of `transform()`).

        Returns:
            pd.DataFrame: DataFrame with columns reordered to match `feature_names_in_`. The index of `x` is preserved.

        Raises:
            ValueError: If imputer has not been fitted (via `check_is_fitted`).
        """
        check_is_fitted(self, "n_features_in_")
        
        return pd.DataFrame(x, columns=self.feature_names_in_)
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns ordered list of output feature names.

        The output feature names are the concatenation of `numerical_columns` and `categorical_columns`,
        in the order they were identified during `fit`.

        Args:
            input_features: Ignored; included for scikit-learn API compatibility.

        Returns:
            List[str]: Ordered list of output feature names.

        Raises:
            ValueError: If imputer has not been fitted (via `check_is_fitted`).
        """
        check_is_fitted(self, "n_features_in_")
        return self.numerical_columns + self.categorical_columns