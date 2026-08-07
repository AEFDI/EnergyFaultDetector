"""Generic class for building a preprocessing pipeline."""

from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ..core.save_load_mixin import SaveLoadMixin
from .column_selector import ColumnSelector
from .low_unique_value_filter import LowUniqueValueFilter
from .angle_transformer import AngleTransformer
from .duplicate_value_to_nan import DuplicateValuesToNan
from .counter_diff_transformer import CounterDiffTransformer
from .timestamp_transformer import TimestampTransformer
from .categorical_encoder import CategoricalEncoder
from .scaler import Scaler
from .imputer import Imputer
from .ffill_imputer import ForwardFillImputer


class DataPreprocessor(Pipeline, SaveLoadMixin):
    """
    A configurable data preprocessing pipeline for tabular data.
    """

    STEP_REGISTRY = {
        'duplicate_to_nan': DuplicateValuesToNan,
        'column_selector': ColumnSelector,
        'low_unique_value_filter': LowUniqueValueFilter,
        'angle_transformer': AngleTransformer,
        'counter_diff_transformer': CounterDiffTransformer,
        'simple_imputer': Imputer,
        'scaler': Scaler,
        'timestamp_transformer': TimestampTransformer,
        'categorical_encoder': CategoricalEncoder,
        'ffill_imputer': ForwardFillImputer,
    }

    NAME_ALIASES: Dict[str, str] = {
        "angle_transform": "angle_transformer",
        "counter_diff": "counter_diff_transformer",
        "counter_diff_transform": "counter_diff_transformer",
        "scaler": "scaler",
        "imputer": "simple_imputer",
        "duplicate_value_to_nan": "duplicate_to_nan",
        "duplicate_values_to_nan": "duplicate_to_nan",
        "timestamp_features": "timestamp_transformer",
        "timestamp_transform": "timestamp_transformer",
        "categorical_encoder": "categorical_encoder",
        "ffill_imputer": "ffill_imputer",
    }

    def __init__(self, steps: Optional[List[Dict[str, Any]]] = None) -> None:
        """A data preprocessing pipeline that allows for configurable steps based on the extended pipeline.

        When no steps are provided, a default pipeline is created which removes features that are constant or binary
        and contain more 5% missing values. Afterward, remaining missing values are imputed with the mean and the
        features are scaled with the StandardScaler.

        Args:
            steps: Optional list of step specifications. Each item is a dict with:

                - name: registered step name (see STEP_REGISTRY).
                - enabled: optional bool (default True).
                - params: dict of constructor arguments for the step.
                - step_name: optional explicit pipeline name (defaults to name).

        Notes:
            Enforced ordering in steps mode:

              1) NaN introducing steps first (DuplicateValuesToNan, CounterDiffTransformer),
              2) ColumnSelector (if present),
              3) Other steps
              4) Imputer placed before scaler (always present; mean strategy by default),
              5) Scaler always last (StandardScaler by default).
              6) TimestampTransformer (if present).

        Configuration example:

            .. code-block:: text

                train:
                  data_preprocessor:
                    steps:
                    - name: column_selector
                      params:
                        max_nan_frac_per_col: 0.05
                        features_to_exclude: ['exclude_this_feature']
                    - name: counter_diff_transformer
                      step_name: counter_flow
                      params:
                        counters: ['flow_total_m3']
                        compute_rate: True
                        fill_first: 'zero'
                    - name: counter_diff_transformer
                      step_name: counter_energy
                      params:
                        counters: ['energy_total_kwh']
                        compute_rate: False
                        fill_first: 'zero'
                        reset_strategy: 'rollover',
                        rollover_values:
                          'energy_total_kwh': 100000.0
        """

        self.steps_spec_: Optional[List[Dict[str, Any]]] = steps

        if steps is not None and len(steps) > 0:
            built_steps = self._build_from_steps_spec()
        else:
            # Build the default pipeline
            built_steps = self._build_default_pipeline()

        super().__init__(steps=built_steps)
        # Ensure pandas output for supported transformers.
        self.set_output(transform="pandas")

    def inverse_transform(self, x: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Inverse-transform scaler and angles (other transforms are not reversed).

        Args:
            x: The transformed data.

        Returns:
            DataFrame with inverse scaling and angle back-transformation.
        """
        check_is_fitted(self)
        x_ = x.copy()  # avoid modifying the original DataFrame

        # Drop time features
        timestamp_key, _ = self._find_step_by_type((TimestampTransformer,))
        if timestamp_key is not None:
            x_ = self.named_steps[timestamp_key].inverse_transform(x_.copy())

        # Find scaler by type and reverse scaling
        scaler_key, _ = self._find_step_by_type((Scaler,))
        x_ = self.named_steps[scaler_key].inverse_transform(x_)
        x_ = pd.DataFrame(data=x_, columns=self.named_steps[scaler_key].get_feature_names_out())

        # Try to reverse angle transformation
        angle_key, _ = self._find_step_by_type((AngleTransformer,))
        if angle_key is not None:
            x_ = self.named_steps[angle_key].inverse_transform(x_)

        # Keep original index
        if isinstance(x, pd.DataFrame):
            x_.index = x.index

        return x_

    # pylint: disable=arguments-renamed
    def transform(self, x: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Apply pipeline steps to the input DataFrame.

        Args:
            x: Input DataFrame.

        Returns:
            DataFrame with the same index as input.
        """
        check_is_fitted(self)
        x_ = super().transform(X=x.copy())
        return x_ # x_ is already a dataframe. Converting to a dataframe here and adding the input index is not working in case rows are dropped during preprocessing (e.g. in ForwardFillImputer). The index of the returned dataframe should be the index of the transformed dataframe, not the input dataframe.

    # pylint: disable=arguments-renamed
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """Fit all transformers in the pipeline.

        Args:
            X: Input DataFrame.
            y: Target variable (optional).
            **fit_params: Parameters to pass to the fit method of each step.
                Use step_name__parameter format (e.g., column_selector__protected_features=['feature1']).
                Special parameter 'protected_features' can be used to validate that these features
                are present in the output after fitting.

        Returns:
            self

        Raises:
            ValueError: If protected_features are specified but not all are present in output.
        """
        # Extract protected features for validation (if provided)
        protected_features = fit_params.pop('protected_features', None)

        # Call parent fit
        result = super().fit(X=X, y=y, **fit_params)

        # Validate that protected features are in the output
        if protected_features:
            output_features = self.get_feature_names_out()
            missing_features = [f for f in protected_features if f not in output_features]
            if missing_features:
                raise ValueError(
                    f"Protected features were dropped by the preprocessing pipeline: {missing_features}. "
                    f"This may be caused by AngleTransformer or CounterDiffTransformer modifying these features. "
                    f"Please ensure protected features (e.g., conditional features for autoencoders) are not "
                    f"transformed by these steps or consider the transformed version of these features (e.g. "
                    f"<feature_name>_sin, <feature_name>_cos, <feature_name>_rate, <feature_name>_diff) as "
                    f"conditional features."
                )

        return result

    # pylint: disable=arguments-renamed
    def fit_transform(self, x: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            x: Input DataFrame.
            **kwargs: Parameters to pass to the fit method of each step.
                Use step_name__parameter format (e.g., column_selector__protected_features=['feature1']).

        Returns:
            Transformed DataFrame with the same index as input.
        """
        super().fit(X=x, **kwargs)
        return self.transform(x)

    def _find_step_by_type(self, types: Tuple[type, ...]) -> Tuple[Optional[str], Optional[object]]:
        """Return the (step name, estimator) of the first step matching any of the given types."""
        for name, est in self.named_steps.items():
            if isinstance(est, types):
                return name, est
        return None, None

    def _normalize_name(self, name: str) -> str:
        """Normalize a user-provided step name to a canonical registry key."""
        return self.NAME_ALIASES.get(name, name)

    @staticmethod
    def _validate_singletons(steps_spec: List[Dict[str, Any]]) -> None:
        """Ensure only one instance of selected steps is present (enabled ones)."""
        singleton_names = {
            "angle_transformer",
            "column_selector",
            "low_unique_value_filter",
            "simple_imputer",
            "ffill_imputer",
            "timestamp_transformer",
            # scaler handled separately (standard_scaler/minmax_scaler) in your code
        }
        counts: List[Tuple[str, int]] = []
        for name in singleton_names:
            n = sum(1 for s in steps_spec if s.get("enabled", True) and s.get("name") == name)
            if n > 1:
                counts.append((name, n))
        if counts:
            raise ValueError(
                "Each of these steps may appear at most once: "
                f"{[n for n, _ in counts]}. Found duplicates: {counts}"
            )

    def _build_default_pipeline(self) -> List:
        """Build default pipeline if no steps are provided.

        Steps:
            - Column selection: A ColumnSelector object filters out columns/features with too many NaN values.
            - Low unique value filter: Remove columns/features with <= 2 unique values.
            - Simple imputation
            - Scaling: Apply either sklearn's StandardScaler or MinMaxScaler.

        Returns:
            List of (name, estimator) tuples for the pipeline.
        """

        steps = [
            ("column_selector", ColumnSelector(max_nan_frac_per_col=0.05)),
            ("low_unique_value_filter", LowUniqueValueFilter(min_unique_value_count=2, max_col_zero_frac=1.0)),
            ("simple_imputer", Imputer(strategy="mean").set_output(transform="pandas")),
            ("standard_scaler", Scaler(with_mean=True, with_std=True)),
        ]

        return steps

    def _build_from_steps_spec(self) -> List:
        """Build pipeline from steps specification (preferred mode) with enforced ordering.

        Each step has the following keys:
          - name: registered step name (see STEP_REGISTRY).
          - enabled: optional, defaults to True.
          - params: dict of constructor parameters for the step.
          - step_name: optional explicit pipeline key (defaults to name).

        Returns:
            List of (name, estimator) tuples for the pipeline.

        Raises:
            ValueError: If a step lacks 'name' or references an unknown step.
        """

        self._validate_step_spec_keys(self.steps_spec_)

        # Filter disabled steps first to simplify ordering.
        enabled_spec = [s for s in self.steps_spec_ if s.get("enabled", True)]
        # Order the steps
        ordered_spec = self._order_steps_spec(enabled_spec)
        self._validate_singletons(enabled_spec)
        # Assign unique step names for duplicates or missing step_name
        ordered_spec = self._assign_unique_step_names(ordered_spec)

        # Create the pipeline from the specified steps
        steps: List = []
        for step_idx, spec in enumerate(ordered_spec):
            name = spec.get("name")
            if name is None:
                raise ValueError("Each step spec requires a 'name'.")
            params = spec.get("params", {})
            cls = self.STEP_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown step name '{name}'. Register it in STEP_REGISTRY.")
            estimator = cls(**params)
            step_name = spec.get("step_name", name)
            steps.append((step_name, estimator))

        return steps

    def _order_steps_spec(self, steps_spec: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize ordering rules for a steps specification.

        Rules:
          - NaN introducing steps first (DuplicateValuesToNan and CounterDiffTransformer)
          - ColumnSelector (if present).
          - Other steps
          - Any imputer placed at the end, before scaler. If no imputer was defined, the a simple imputer with imputation
            strategy 'mean' is added.
          - Scaler last (if present). If no scaler is added, the StandardScaler with default values is added.
          - TimestampTransformer (if present).

        Args:
            steps_spec: List of step dictionaries.

        Returns:
            Reordered list of step dictionaries.
        """
        # Normalize names to canonical keys for grouping
        for s in steps_spec:
            s["name"] = self._normalize_name(s.get("name"))

        # Separate groups by type
        column_selector = [s for s in steps_spec if s.get("name") == "column_selector"]
        low_unique_value_filter = [s for s in steps_spec if s.get("name") == "low_unique_value_filter"]
        duplicates = [s for s in steps_spec if s.get("name") == "duplicate_to_nan"]
        counter = [s for s in steps_spec if s.get("name") == "counter_diff_transformer"]
        imputer = [s for s in steps_spec if s.get("name") in {"simple_imputer", "ffill_imputer"}]
        scalers = [s for s in steps_spec if s.get("name") == "scaler"]
        encoders = [s for s in steps_spec if s.get("name") == "categorical_encoder"]
        if len(scalers) > 1:
            raise ValueError(f"Only one scaler can be used, two found in the steps specification: {scalers}")
        if len(imputer) > 1:
            raise ValueError(f"Only one imputer can be used, two found in the steps specification: {imputer}")
        timestamp_step = [s for s in steps_spec if s.get("name") == "timestamp_transformer"]

        others = [
            s for s in steps_spec
            if s.get("name") not in {
                "column_selector", "duplicate_to_nan", "counter_diff_transformer", "simple_imputer",
                "categorical_encoder", "low_unique_value_filter", "timestamp_transformer", "scaler",
                "ffill_imputer"
            }
        ]

        # Add default scaler if empty
        if not scalers:
            scalers = [{'name': 'scaler',
                        'step_name': 'scaler',
                        'params': {
                            'scaler_type': 'standard',
                            'scale_categorical_features': True,
                            'with_mean': True,
                            'with_std': True}}]
        # Add default imputer if empty
        if not imputer:
            imputer = [{'name': 'simple_imputer',
                        'step_name': 'simple_imputer',
                        'params': {'strategy': 'mean'}}]

        # Order the preprocessing steps
        ordered = []
        # can add NaN values or add new features that may be constant
        ordered.extend(duplicates)
        ordered.extend(counter)
        # drop columns based on the values (NaNs, no variance)
        ordered.extend(column_selector)
        ordered.extend(low_unique_value_filter)
        # other transformations
        ordered.extend(others)
        # Imputation
        ordered.extend(imputer)
        # Encoding categorical features before scaling and after imputation (encoder would crash if NaN values are present)
        ordered.extend(encoders)
        # Scaling
        ordered.extend(scalers)
        # No scaling needed for the time features
        ordered.extend(timestamp_step)
        return ordered

    @staticmethod
    def _assign_unique_step_names(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign unique pipeline step names. If step_name is provided, use it; if it collides, append _2, _3, ...
        If step_name is not provided, use the 'name' key. If this key occurs  multiple times, assign name_1, name_2, ...

        This method mutates specs in place and also returns it.

        Returns:
            Altered step specifications.
        """
        total_counts = Counter(s["name"] for s in specs)
        used: set[str] = set()
        per_base_index = defaultdict(int)

        for s in specs:
            explicit = s.get("step_name")
            if explicit:
                base = explicit
                candidate = base
                i = 1
                while candidate in used:
                    i += 1
                    candidate = f"{base}_{i}"
                s["step_name"] = candidate
                used.add(candidate)
                continue

            base = s["name"]
            if total_counts[base] == 1 and base not in used:
                candidate = base
            else:
                per_base_index[base] += 1
                candidate = f"{base}_{per_base_index[base]}"
                while candidate in used:
                    per_base_index[base] += 1
                    candidate = f"{base}_{per_base_index[base]}"

            s["step_name"] = candidate
            used.add(candidate)

        return specs

    @staticmethod
    def _validate_step_spec_keys(steps_spec: List[Dict[str, Any]]) -> None:
        """Validate that each step spec uses only allowed keys and includes 'name'.

        Allowed keys: {'name', 'enabled', 'params', 'step_name'}.

        Args:
            steps_spec: Raw steps specification provided by the user.

        Raises:
            ValueError: If a step is missing 'name' or contains unknown keys.
        """
        allowed = {"name", "enabled", "params", "step_name"}

        for i, spec in enumerate(steps_spec):
            if "name" not in spec:
                raise ValueError(f"Step #{i} is missing required key 'name'.")
            unknown = set(spec.keys()) - allowed
            if unknown:
                raise ValueError(
                    f"Step #{i} has unknown keys: {sorted(unknown)}. "
                    f"Allowed keys are: {sorted(allowed)}."
                )
