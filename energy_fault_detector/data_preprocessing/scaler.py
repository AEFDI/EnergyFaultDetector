import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer
import logging

logger = logging.getLogger('energy_fault_detector')


class Scaler(DataTransformer):
    """Preprocessing step for scaling numerical (and optionally categorical) features.

    Supports 'standard' (StandardScaler) and 'minmax' (MinMaxScaler) scalers.
    Automatically enforces float conversion for all columns and optionally excludes
    encoded categorical features from scaling depending on `scale_categorical_features`.

    Attributes:
        scaler_type (str): Type of scaler selected ('standard' or 'minmax').
        scale_categorical_features (bool): Whether to scale features matched in `categorical_features`.
        categorical_features (List[str]): List of column names or prefixes treated as categorical.
        scaler (SklearnScaler): Instantiated sklearn scaler object (e.g., StandardScaler).
        feature_names_in_ (List[str]): Names of features seen during `fit()`.
        feature_names_out_ (List[str]): Names of features output after `transform()` (same as input).
        columns_dropped_ (List[str]): Placeholder (unused in current implementation; preserved for API).
        params (dict): Keyword arguments passed to the underlying sklearn scaler constructor.
    """

    SCALER_REGISTRY = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler
    }

    def __init__(self, scaler_type: str = 'standard', scale_categorical_features: bool = True,
                 categorical_features: list = None, **params):
        """Initializes the Scaler with specified strategy and feature selection logic.

        Args:
            scaler_type (str, optional): Type of scaler to use. Supported: `'standard'` (default), `'minmax'`.
            scale_categorical_features (bool, optional): Whether to include categorical (e.g., one-hot encoded)
                features in scaling. If `False`, only non-encoded features are scaled.
                Defaults to `True`.
            categorical_features (List[str], optional): List of column names or prefixes to treat as categorical.
                Used only when `scale_categorical_features=False` to determine which features to exclude.
                Defaults to `None` (interpreted as empty list).
            **params: Additional keyword arguments passed to the underlying sklearn scaler constructor.
                Example: `{'with_mean': False, 'with_std': True}`.

        Raises:
            ValueError: If `scaler_type` is not supported (must be `'standard'` or `'minmax'`).
        """
        super().__init__()

        # Validate scaler_type
        if scaler_type not in self.SCALER_REGISTRY:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Valid types are: {list(self.SCALER_REGISTRY.keys())}")

        self.scaler_type = scaler_type
        self.scale_categorical_features = scale_categorical_features
        self.categorical_features = categorical_features if categorical_features else []
        self.scaler = self.SCALER_REGISTRY.get(scaler_type)

        # Attributes to be defined during fitting
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.columns_dropped_ = []

        # Parameters of nested estimators
        self.params = params
        # Initialize nested estimators
        self.scaler = self.scaler(**self.params)

    def fit(self, x: pd.DataFrame, y: pd.Series = None) -> 'Scaler':
        """Fits the scaler on the input data, optionally excluding categorical features.

        Enforces float dtype for all columns. Scales either all columns or only
        non-encoded numerical features, depending on `scale_categorical_features`.

        Args:
            x (pd.DataFrame): Input feature DataFrame.
            y (pd.Series, optional): Target variable. Ignored; included for scikit-learn API compatibility.

        Returns:
            Scaler: The fitted transformer instance (self), for method chaining.

        Raises:
            ValueError: If any column cannot be converted to float.
            ValueError: If column dtypes are inconsistent across features.
        """
        logger.debug("Fitting Scaler transformer...")
        x = x.copy()  # Avoid modifying the original DataFrame

        # Check that all columns in input are numerical types.
        for col in x.columns:
            try:
                x[col] = x[col].astype(float, errors='raise')
            except ValueError as e:
                raise ValueError(f"Column '{col}' cannot be converted to float.") from e

        self.feature_names_in_ = list(x.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        # Determine features to scale
        if self.scale_categorical_features:
            subset_to_fit = x
        else:
            non_encoded_features = [col for col in x.columns if not any(feature in col for feature in self.categorical_features)]
            subset_to_fit = x[non_encoded_features]

        # Fit the scaler
        self.scaler.fit(subset_to_fit)

        # Define output feature names
        self.feature_names_out_ = list(x.columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Applies the learned scaling transformation to input data.

        Enforces float dtypes and validates feature alignment with `fit()`.
        Scales either all columns or only numerical (non-encoded) columns,
        depending on `scale_categorical_features`.

        Args:
            x (pd.DataFrame): Input feature DataFrame, with column order and names matching those seen in `fit()`.

        Returns:
            pd.DataFrame: Scaled DataFrame, same shape and index as input.

        Raises:
            ValueError: If input feature names/order do not match those seen during `fit()`.
            ValueError: If any column cannot be converted to float.
            ValueError: If scaler has not been fitted (via `check_is_fitted`).
        """
        logger.debug("Transforming data with Scaler transformer...")
        check_is_fitted(self, "n_features_in_")

        x_transformed = x.copy()  # Avoid modifying the original DataFrame

        # Check that all columns in input are numerical types.
        for col in x_transformed.columns:
            try:
                x_transformed[col] = x_transformed[col].astype(float, errors='raise')
            except ValueError as e:
                raise ValueError(f"Column '{col}' cannot be converted to float.") from e
                
        # Check if input features match the features seen during fit
        if list(x.columns) != self.feature_names_in_:
            raise ValueError(f"Input features {list(x.columns)} do not match the features seen during fit {self.feature_names_in_}.")
        # Transform the appropriate features
        if self.scale_categorical_features:
            x_transformed.iloc[:, :] = self.scaler.transform(x_transformed)
        else:
            non_encoded_features = [col for col in x.columns if not any(feature in col for feature in self.categorical_features)]
            x_transformed[non_encoded_features] = self.scaler.transform(x[non_encoded_features])

        return x_transformed

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Applies the inverse scaling transformation to revert scaled data.

        Requires that the scaler was fitted and that input column structure matches `fit()`.

        Args:
            x (pd.DataFrame): Scaled input DataFrame, with same columns and order as `fit()` output.

        Returns:
            pd.DataFrame: Inversely transformed (descaled) DataFrame, with same shape/index as input.

        Raises:
            ValueError: If scaler has not been fitted (via `check_is_fitted`).
            ValueError: If input feature names/order do not match those seen during `fit()`.
        """
        logger.debug("Applying inverse transformation with Scaler transformer...")
        check_is_fitted(self, "n_features_in_")

        x_inverse_transformed = x.copy()

        # Apply inverse transform to the appropriate features
        if self.scale_categorical_features:
            x_inverse_transformed.iloc[:, :] = self.scaler.inverse_transform(x_inverse_transformed)
        else:
            non_encoded_features = [col for col in x.columns if not any(feature in col for feature in self.categorical_features)]
            x_inverse_transformed[non_encoded_features] = self.scaler.inverse_transform(x[non_encoded_features])

        return x_inverse_transformed

    def get_feature_names_out(self, input_features=None) -> list:
        """Returns the list of output feature names (same as input feature names).

        Preserves original column order and names seen during `fit()`.

        Args:
            input_features: Ignored; included for scikit-learn API compatibility.

        Returns:
            List[str]: Ordered list of output feature names.

        Raises:
            ValueError: If scaler has not been fitted (via `check_is_fitted`).
        """
        check_is_fitted(self, "n_features_in_")
        return self.feature_names_out_
