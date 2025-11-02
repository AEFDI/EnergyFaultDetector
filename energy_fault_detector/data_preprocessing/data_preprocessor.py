"""Generic class for building a preprocessing pipeline."""

from typing import List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from energy_fault_detector.core.save_load_mixin import SaveLoadMixin
from energy_fault_detector.data_preprocessing.column_selector import ColumnSelector
from energy_fault_detector.data_preprocessing.low_unique_value_filter import LowUniqueValueFilter
from energy_fault_detector.data_preprocessing.angle_transformer import AngleTransformer
from energy_fault_detector.data_preprocessing.duplicate_value_to_nan import DuplicateValuesToNan
from energy_fault_detector.data_preprocessing.counter_diff_transformer import CounterDiffTransformer


class DataPreprocessor(Pipeline, SaveLoadMixin):
    """A data preprocessing pipeline that allows for configurable steps based on the extended pipeline.

        0. (optional) Replace any consecutive duplicate zero-values (or another value) with NaN. This step should be
            used if 0 can also represent missing values in the data.
        1. (optional) Column selection: A ColumnSelector object filters out columns/features with too many NaN values.
        2. (optional) Features containing angles are transformed to sine/cosine values.
        3. (optional) Low unique value filter: Remove columns/features with a low number of unique values or
            high fraction of zeroes. The high fraction of zeros setting should be used if 0 can also represent missing
            values in the data.
        4. Imputation with sklearn's SimpleImputer
        5. Scaling: Apply either sklearn's StandardScaler or MinMaxScaler.

    Args:
        angles: List of angle features for transformation. Defaults to None.
            If none provided (or empty list), this step is skipped.
        imputer_strategy: Strategy for imputation ('mean', 'median', 'most_frequent', 'constant'). Defaults to 'mean'.
        imputer_fill_value: Value to fill for imputation (if imputer_strategy=='constant').
        scale: Type of scaling ('standardize' or 'normalize'). Defaults to 'standardize'.
        include_column_selector: Whether to include the column selector step. Defaults to True.
        features_to_exclude: ColumnSelector option, list of features to exclude from processing.
        max_nan_frac_per_col: ColumnSelector option, max fraction of NaN values allowed per column. Defaults to 0.05.
        include_low_unique_value_filter: Whether to include the low unique value filter step. Defaults to True.
        min_unique_value_count: Minimum number of unique values for low unique value filter. Defaults to 2.
        max_col_zero_frac: Maximum fraction of zeroes for low unique value filter. Defaults to 1.0.
        include_duplicate_value_to_nan: Whether to include the duplicate value replacement step. Defaults to False.
        value_to_replace: Value to replace with NaN (if using duplicate value replacement). Defaults to None.
        n_max_duplicates: Max number of consecutive duplicates to replace with NaN. Defaults to 144.

    Configuration example:

    .. code-block:: text

        train:
          data_preprocessor:
            params:
              scale: normalize
              imputer_strategy: mean
              max_nan_frac_per_col: 0.05
              include_low_unique_value_filter: true
              min_unique_value_count: 2
              max_col_zero_frac: 0.99
              angles:
              - angle1
              - angle2
              features_to_exclude:
              - feature1
              - feature2
    """

    def __init__(self,
                 angles: Optional[List[str]] = None,
                 imputer_strategy: str = 'mean',
                 imputer_fill_value: Optional[int] = None,
                 scale: str = 'standardize',
                 include_column_selector: bool = True,
                 features_to_exclude: Optional[List[str]] = None,
                 max_nan_frac_per_col: float = 0.05,
                 include_low_unique_value_filter: bool = True,
                 min_unique_value_count: int = 2,
                 max_col_zero_frac: float = 1.,
                 include_duplicate_value_to_nan: bool = False,
                 value_to_replace: float = 0,
                 n_max_duplicates: int = 144,
                 duplicate_features_to_exclude: Optional[List[str]] = None,
                 counter_columns_to_transform: Optional[List[str]] = None,
                 ):

        self.include_column_selector = include_column_selector
        self.features_to_exclude = features_to_exclude
        self.max_nan_frac_per_col = max_nan_frac_per_col

        self.angles = angles

        self.include_low_unique_value_filter = include_low_unique_value_filter
        self.min_unique_value_count = min_unique_value_count
        self.max_col_zero_frac = max_col_zero_frac

        self.imputer_strategy = imputer_strategy
        self.imputer_fill_value = imputer_fill_value

        self.scale = scale

        self.include_duplicate_value_to_nan = include_duplicate_value_to_nan
        self.value_to_replace = value_to_replace
        self.n_max_duplicates = n_max_duplicates
        self.duplicate_features_to_exclude = duplicate_features_to_exclude

        # Define the scaler based on the chosen scale type
        scaler = (StandardScaler(with_mean=True, with_std=True)
                  if scale in ['standardize', 'standard', 'standardscaler']
                  else MinMaxScaler(feature_range=(0, 1)))

        # Configure the pipeline steps
        steps = []

        if include_duplicate_value_to_nan:
            steps.append(
                ('value_to_nan',
                 # Do not open source, very specific to our data problems
                 DuplicateValuesToNan(value_to_replace=value_to_replace, n_max_duplicates=n_max_duplicates,
                                      features_to_exclude=duplicate_features_to_exclude))
            )
        if counter_columns_to_transform is not None and len(counter_columns_to_transform) > 0:
            steps.append((
                'counter_diff',
                CounterDiffTransformer(
                    counters=counter_columns_to_transform,
                    compute_rate=False,  # per-sample diffs by default
                    reset_strategy='zero',  # assume reset-to-zero
                    rollover_values=None,  # or dict per counter if known
                    small_negative_tolerance=0.0,
                    fill_first='nan',
                    keep_original=False,
                    gap_policy='mask',  # mask after long gaps
                    max_gap_seconds=None,  # if None, uses max_gap_factor * median(dt)
                    max_gap_factor=3.0  # e.g., mask when gap > 3x typical cadence
                )
            ))
        if include_column_selector:
            steps.append(
                ('column_selector',
                 ColumnSelector(max_nan_frac_per_col=max_nan_frac_per_col, features_to_exclude=features_to_exclude))
            )
        if include_low_unique_value_filter:
            steps.append(
                ('low_unique_value_filter',
                 LowUniqueValueFilter(min_unique_value_count=min_unique_value_count, max_col_zero_frac=max_col_zero_frac))
            )
        if angles is not None and len(angles) > 0:
            steps.append(('angle_transform', AngleTransformer(angles=angles)))

        # default steps:
        steps.append(('imputer', SimpleImputer(strategy=imputer_strategy,
                                               fill_value=imputer_fill_value).set_output(transform='pandas')))
        steps.append(('scaler', scaler))

        super().__init__(steps=steps)
        self.set_output(transform="pandas")  # set output of all transformers to pandas

    def inverse_transform(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Reverses the scaler and angle transforms applied to the data.
        Other transformations are not reversed.

        Args:
            x: The transformed data.

        Returns:
            A DataFrame with the inverse transformed data.
        """

        x_ = self.named_steps['scaler'].inverse_transform(x.copy())
        x_ = pd.DataFrame(data=x_, columns=self.named_steps['scaler'].get_feature_names_out())
        if 'angle_transform' in self.named_steps:
            x_ = self.named_steps['angle_transform'].inverse_transform(x_)

        if isinstance(x, pd.DataFrame):
            # ensure the index is kept
            x_.index = x.index

        return x_

    # pylint: disable=arguments-renamed
    def transform(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transforms the input DataFrame using the pipeline.

        Args:
            x: Input DataFrame.

        Returns:
             a dataframe with the same index as the input dataframe.
        """

        x_ = super().transform(X=x.copy())
        return pd.DataFrame(data=x_,
                            columns=self.get_feature_names_out(),
                            index=x.index)

    # pylint: disable=arguments-renamed
    def fit_transform(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit the model and transform with the final estimator.

        Args:
            x: Input DataFrame.

        Returns:
            Transformed DataFrame with the same index as the input dataframe.
        """

        super().fit(X=x)
        return self.transform(x)
