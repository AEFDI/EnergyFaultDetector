"""Generic class for building a preprocessing pipeline."""

from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any, Tuple
import warnings

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
    STEP_REGISTRY = {
        'duplicate_to_nan': DuplicateValuesToNan,
        'column_selector': ColumnSelector,
        'low_unique_value_filter': LowUniqueValueFilter,
        'angle_transformer': AngleTransformer,
        'counter_diff_transformer': CounterDiffTransformer,
        'imputer': SimpleImputer,
        'standard_scaler': StandardScaler,
        'minmax_scaler': MinMaxScaler,
    }

    NAME_ALIASES: Dict[str, str] = {
        "angle_transform": "angle_transformer",
        "counter_diff": "counter_diff_transformer",
        "counter_diff_transform": "counter_diff_transformer",
        "standardize": "standard_scaler",
        "standard": "standard_scaler",
        "standardscaler": "standard_scaler",
        "minmax": "minmax_scaler",
        "simple_imputer": "imputer",
        "duplicate_value_to_nan": "duplicate_to_nan",
        "duplicate_values_to_nan": "duplicate_to_nan",
    }

    def __init__(self, steps: Optional[List[Dict[str, Any]]] = None, **params: Any) -> None:
        """A data preprocessing pipeline that allows for configurable steps based on the extended pipeline.

        If both steps and legacy params are provided, steps take precedence and a warning is emitted.
        When neither steps nor legacy params are provided, a default "old-style" pipeline is created:
          - ColumnSelector with max_nan_frac_per_col = 0.05.
            This drops all columns that have more 5% NaN values
          - LowUniqueValueFilter with min_unique_value_count = 2 and max_col_zero_frac = 1.
            This drops columns with less than 2 unique values, i.e. unchanging values.
          - SimpleImputer with 'mean' as strategy.
          - StandardScaler.

        Args:
            steps: Optional list of step specifications. Each item is a dict with:
                - name: registered step name (see STEP_REGISTRY).
                - enabled: optional bool (default True).
                    - params: dict of constructor arguments for the step.
                - step_name: optional explicit pipeline name (defaults to name).

            **params: Legacy parameters used when steps is None (see _legacy_keys()).

        Notes:
            - Enforced ordering in steps mode:
              1) NaN introducing steps first (DuplicateValuesToNan, CounterDiffTransformer),
              2) ColumnSelector (if present),
              3) Other steps
              4) Imputer placed before scaler (always present; mean strategy by default),
              5) Scaler always last (StandardScaler by default).
            - Supports two configuration modes: steps and legacy. For steps set up pass steps=[...] with per-step
              parameters. For legacy use pass old flags via **params (angles, scale, include_* etc.).
            - Legacy mode keeps the historical order (Duplicate/Counter -> ColumnSelector -> others ->
              Imputer -> Scaler).
            - In steps mode, only one AngleTransformer, ColumnSelector, LowUniqueValueFilter, and Imputer are allowed;
              a ValueError is raised if duplicates are provided. Multiple CounterDiffTransformer and
              DuplicateValuesToNan steps are allowed.

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
        self.params_: Dict[str, Any] = dict(params)

        if steps is not None and len(steps) > 0:
            # Warn if legacy params are present alongside steps.
            legacy_keys = set(self._legacy_keys())
            legacy_used = [k for k in self.params_.keys() if k in legacy_keys]
            if legacy_used:
                warnings.warn(
                    f"DataPreprocessor: 'steps' provided; legacy params are ignored: {legacy_used}",
                    UserWarning
                )
            built_steps = self._build_from_steps_spec()
        else:
            # Build the default or legacy pipeline. If params is empty, defaults are applied.
            built_steps = self._build_from_legacy()

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
        # Find scaler by type
        scaler_key, _ = self._find_step_by_type((StandardScaler, MinMaxScaler))
        x_ = self.named_steps[scaler_key].inverse_transform(x.copy())
        x_ = pd.DataFrame(data=x_, columns=self.named_steps[scaler_key].get_feature_names_out())

        # AngleTransformer supports inverse_transform; apply if present.
        angle_key, _ = self._find_step_by_type((AngleTransformer,))
        if angle_key is not None:
            x_ = self.named_steps[angle_key].inverse_transform(x_)

        # Keep original index (important for time series).
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
        x_ = super().transform(X=x.copy())
        return pd.DataFrame(data=x_, columns=self.get_feature_names_out(), index=x.index)

    # pylint: disable=arguments-renamed
    def fit_transform(self, x: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            x: Input DataFrame.

        Returns:
            Transformed DataFrame with the same index as input.
        """
        super().fit(X=x)
        return self.transform(x)

    def _find_step_by_type(self, types: Tuple[type, ...]) -> Tuple[Optional[str], Optional[object]]:
        """Return the (step name, estimator) of the first step matching any of the given types."""
        for name, est in self.named_steps.items():
            if isinstance(est, types):
                return name, est
        return None, None

    @staticmethod
    def _legacy_keys() -> List[str]:
        """Return the list of supported legacy parameter keys."""
        return [
            "angles",
            "imputer_strategy",
            "imputer_fill_value",
            "scale",
            "include_column_selector",
            "features_to_exclude",
            "max_nan_frac_per_col",
            "include_low_unique_value_filter",
            "min_unique_value_count",
            "max_col_zero_frac",
            "include_duplicate_value_to_nan",
            "value_to_replace",
            "n_max_duplicates",
            "duplicate_features_to_exclude",
            "counter_columns_to_transform",
        ]

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
            "imputer",
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

    def _build_from_legacy(self) -> List:
        """Build pipeline from legacy parameters (old behavior + enforced ordering).

        Steps:
            0. (optional) Replace any consecutive duplicate zero-values (or another value) with NaN. This step should be
                used if 0 can also represent missing values in the data.
            1. (optional) Normalize counters to differences.
            2. (optional) Column selection: A ColumnSelector object filters out columns/features with too many NaN values.
            3. (optional) Low unique value filter: Remove columns/features with a low number of unique values or
                high fraction of zeroes. The high fraction of zeros setting should be used if 0 can also represent missing
                values in the data.
            4. (optional) Features containing angles are transformed to sine/cosine values.
            5. Imputation with sklearn's SimpleImputer
            6. Scaling: Apply either sklearn's StandardScaler or MinMaxScaler.

        Use legacy parameters passed via **params. If empty, defaults are used.
            - angles: List of angle features for transformation. Default: None (skipped).
            - imputer_strategy: Strategy for imputation ('mean', 'median', 'most_frequent', 'constant'). Default: 'mean'.
            - imputer_fill_value: Value to fill for imputation (if imputer_strategy=='constant').
            - scale: Type of scaling ('standardize' or 'normalize'). Default: 'standardize'.
            - include_column_selector: Whether to include the column selector step. Default: True.
            - features_to_exclude: ColumnSelector option, list of features to exclude from processing.
            - max_nan_frac_per_col: ColumnSelector option, max fraction of NaN values allowed per column. Default: 0.05.
            - include_low_unique_value_filter: Whether to include the low unique value filter step. Default: True.
            - min_unique_value_count: Minimum number of unique values for low unique value filter. Default: 2.
            - max_col_zero_frac: Maximum fraction of zeroes for low unique value filter. Default: 1.0.
            - include_duplicate_value_to_nan: Whether to include the duplicate value replacement step. Default: False.
            - value_to_replace: Value to replace with NaN (if using duplicate value replacement). Default: None.
            - n_max_duplicates: Max number of consecutive duplicates to replace with NaN. Default: 144.
            - counter_columns_to_transform: List of counters to normalize to differences. Default: None (skipped).

        Returns:
            List of (name, estimator) tuples for the pipeline.
        """
        steps: List = []
        params = self.params_

        # 0. Replace any consecutive duplicate zero-values (or another value) with NaN.
        if params.get("include_duplicate_value_to_nan", False):
            steps.append(
                (
                    "value_to_nan",
                    DuplicateValuesToNan(value_to_replace=params.get("value_to_replace", 0),
                                         n_max_duplicates=params.get("n_max_duplicates", 144),
                                         features_to_exclude=params.get("duplicate_features_to_exclude")),
                )
            )
        # 1. (optional) Normalize counters to differences.
        counter_cols = params.get("counter_columns_to_transform", [])
        if len(counter_cols) > 0:
            steps.append(
                (
                    "counter_diff",
                    CounterDiffTransformer(
                        counters=counter_cols,
                        compute_rate=False,
                        reset_strategy="zero",
                        rollover_values=None,
                        small_negative_tolerance=0.0,
                        fill_first="nan",
                        keep_original=False,
                        gap_policy="mask",
                        max_gap_seconds=None,
                        max_gap_factor=3.0,
                    ),
                )
            )
        # 2. ColumnSelector (default enabled)
        if params.get("include_column_selector", True):
            steps.append(
                (
                    "column_selector",
                    ColumnSelector(max_nan_frac_per_col=params.get("max_nan_frac_per_col", 0.05),
                                   features_to_exclude=params.get("features_to_exclude")),
                )
            )
        # 3. Optional value filters and angle transforms (before imputer)
        if params.get("include_low_unique_value_filter", True):
            steps.append(
                (
                    "low_unique_value_filter",
                    LowUniqueValueFilter(
                        min_unique_value_count=params.get("min_unique_value_count", 2),
                        max_col_zero_frac=params.get("max_col_zero_frac", 1.0),
                    ),
                )
            )
        # 4. Apply optional angle transformations
        angles = params.get("angles", [])
        if len(angles) > 0:
            steps.append(("angle_transform", AngleTransformer(angles=angles)))
        # 5. Impute missing values with SimpleImputer
        steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy=params.get("imputer_strategy", "mean"),
                    fill_value=params.get("imputer_fill_value", None),
                ).set_output(transform="pandas"),
            )
        )
        # 6. Scale data
        scale = params.get("scale", "standardize")
        scaler = (
            StandardScaler(with_mean=True, with_std=True)
            if scale in ["standardize", "standard", "standardscaler"]
            else MinMaxScaler(feature_range=(0, 1))
        )
        steps.append(("scaler", scaler))
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
        self._validate_singletons(enabled_spec)
        ordered_spec = self._order_steps_spec(enabled_spec)
        # Assign unique step names for duplicates or missing step_name
        ordered_spec = self._assign_unique_step_names(ordered_spec)

        steps: List = []
        scaler_defined = False
        scaler_names = {"standard_scaler", "minmax_scaler"}
        scaler_idx = None
        for step_idx, spec in enumerate(ordered_spec):
            name = spec.get("name")
            if name is None:
                raise ValueError("Each step spec requires a 'name'.")
            if name in scaler_names:
                scaler_defined = True
                scaler_idx = step_idx
            params = spec.get("params", {})
            cls = self.STEP_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown step name '{name}'. Register it in STEP_REGISTRY.")
            estimator = cls(**params)
            step_name = spec.get("step_name", name)
            steps.append((step_name, estimator))

        # Ensure an Imputer exists and is placed before the scaler.
        if not any(n == "imputer" for n, _ in steps):
            default_imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")
            # Insert before scaler if scaler already present; else append.
            if scaler_idx is not None:
                steps.insert(scaler_idx, ("imputer", default_imputer))
            else:
                steps.append(("imputer", default_imputer))

        # Ensure a scaler exists and is last. If missing, add StandardScaler by default.
        if not scaler_defined:
            steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

        return steps

    def _order_steps_spec(self, steps_spec: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize ordering rules for a steps specification.

        Rules:
          - NaN introducing steps first (DuplicateValuesToNan and CounterDiffTransformer)
          - ColumnSelector (if present).
          - Other steps
          - Any imputer placed at the end, before scaler. If no imputer was defined, the SimpleImputer with imputation
            strategy 'mean' is added.
          - Scaler last (if present). If no scaler is added, the StandardScaler with default values is added.

        Args:
            steps_spec: List of step dictionaries.

        Returns:
            Reordered list of step dictionaries.
        """
        # Normalize names to canonical keys for grouping
        for s in steps_spec:
            s["name"] = self._normalize_name(s.get("name"))

        # Separate groups by type for easy reassembly.
        column_selector = [s for s in steps_spec if s.get("name") == "column_selector"]
        low_unique_value_filter = [s for s in steps_spec if s.get("name") == "low_unique_value_filter"]
        duplicates = [s for s in steps_spec if s.get("name") == "duplicate_to_nan"]
        counter = [s for s in steps_spec if s.get("name") == "counter_diff_transformer"]
        imputer = [s for s in steps_spec if s.get("name") == "imputer"]
        scaler_names = {"standard_scaler", "minmax_scaler"}
        scalers = [s for s in steps_spec if s.get("name") in scaler_names]
        if len(scalers) > 1:
            raise ValueError("Only one scaler can be used, two found in the steps specification: ."
                             f"{scalers}")
        others = [
            s for s in steps_spec
            if s.get("name") not in {
                "column_selector", "duplicate_to_nan", "counter_diff_transformer", "imputer", "low_unique_value_filter",
            } | scaler_names
        ]

        # Keep 'others' in their original relative order.
        ordered = []
        # can add NaN avalues or add new features that may be constant
        ordered.extend(duplicates)
        ordered.extend(counter)
        # drop columns based on the values (NaNs, no variance)
        ordered.extend(column_selector)
        ordered.extend(low_unique_value_filter)
        # other transformations
        ordered.extend(others)
        # end with imputation and scaling
        ordered.extend(imputer)  # may be empty; scaler gets default added later if missing
        ordered.extend(scalers)  # may be empty; scaler gets default added later if missing
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
