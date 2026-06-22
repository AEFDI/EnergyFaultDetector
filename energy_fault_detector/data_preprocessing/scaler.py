import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer
import logging

logger = logging.getLogger('energy_fault_detector')


class Scaler(DataTransformer):
    """Scaler pre-processor step for scaling datasets."""

    SCALER_REGISTRY = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler
    }

    def __init__(self, scaler_type: str = 'standard', scale_categorical_features: bool = True,
                 categorical_features: list = None, **params):
        """
        Initialize the Scaler object.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax'). Defaults to 'standard'.
            scale_categorical_features: Flag to scale categorical features after encoding. Defaults to True.
            categorical_features: List of names of categorical features. Defaults to None.
        """
        super().__init__()
        self.scaler_type = scaler_type
        self.scale_categorical_features = scale_categorical_features
        self.categorical_features = categorical_features if categorical_features else []
        self.scaler = self.SCALER_REGISTRY.get(scaler_type)

        # Attributes to be defined during fitting
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.columns_dropped_ = []

        # Parameters of nested estimators
        self.params = params
        # Initialize nested estimators
        self.scaler = self.scaler(**self.params)

    def fit(self, x: pd.DataFrame, y: pd.Series = None) -> 'Scaler':
        """
        Fit the scaler to the dataset.

        Args:
            x: pandas DataFrame with input data.
            y: (optional) labels. Defaults to None.

        Returns:
            Self.
        """
        logger.debug("Fitting Scaler transformer...")
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
        """
        Apply the scaling transformation to the data.

        Args:
            x: pandas DataFrame of input data.

        Returns:
            Transformed DataFrame.
        """
        logger.debug("Transforming data with Scaler transformer...")
        check_is_fitted(self)
        x_transformed = x.copy()
        # TODO: not sure if this work when x has different columns as the ones during fit...
        # Transform the appropriate features
        if self.scale_categorical_features:
            x_transformed.iloc[:, :] = self.scaler.transform(x_transformed)
        else:
            non_encoded_features = [col for col in x.columns if not any(feature in col for feature in self.categorical_features)]
            x_transformed[non_encoded_features] = self.scaler.transform(x[non_encoded_features])

        return x_transformed

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the inverse scaling transformation to the data.

        Args:
            x: pandas DataFrame of scaled input data.

        Returns:
            Inversely transformed DataFrame.
        """
        logger.debug("Applying inverse transformation with Scaler transformer...")
        x_inverse_transformed = x.copy()

        # Apply inverse transform to the appropriate features
        if self.scale_categorical_features:
            x_inverse_transformed.iloc[:, :] = self.scaler.inverse_transform(x_inverse_transformed)
        else:
            non_encoded_features = [col for col in x.columns if not any(feature in col for feature in self.categorical_features)]
            x_inverse_transformed[non_encoded_features] = self.scaler.inverse_transform(x[non_encoded_features])

        return x_inverse_transformed

    def get_feature_names_out(self, input_features=None) -> list:
        """
        Get output feature names for the transformed data.

        Args:
            input_features: Optional list of input features.

        Returns:
            List of output feature names.
        """
        return self.feature_names_out_
