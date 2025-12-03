from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def _build_preprocessor_steps(
    *,
    max_nan_frac: float,
    min_unique_value_count: int,
    max_col_zero_frac: float,
    angle_columns: Optional[List[str]],
    counter_columns: Optional[List[str]],
    imputer_strategy: str,
    scaler: str,
) -> List[Dict[str, Any]]:
    """
    Build the steps specification for the DataPreprocessor pipeline.

    This helper focuses solely on the steps list for the preprocessing pipeline
    and keeps the public function small, readable, and testable.

    Args:
        max_nan_frac (float): Maximum fraction of missing values allowed per column.
        min_unique_value_count (int): Minimal number of unique values required for a column to remain.
        max_col_zero_frac (float): Maximum allowed fraction of zeros in a column (used in the unique-value filter).
        angle_columns (Optional[List[str]]): Optional list of column names to be angle-transformed.
        counter_columns (Optional[List[str]]): Optional list of counter columns to be transformed to differences.
        imputer_strategy (str): SimpleImputer strategy, e.g., "mean", "median", "most_frequent", or "constant".
        scaler (str): Scaler type; supports "standard" (and aliases) or "minmax" (and aliases).

    Returns:
        List[Dict[str, Any]]: A steps list suitable for DataPreprocessor(steps=[...]).

    Notes:

        - The order is kept minimal here; DataPreprocessor enforces proper ordering internally.

    """
    steps: List[Dict[str, Any]] = []

    # Optional counter-diff transformation (DataPreprocessor will place it early).
    if counter_columns:
        steps.append(
            {
                "name": "counter_diff_transformer",
                "params": {
                    "counters": counter_columns,
                    "compute_rate": False,
                    "fill_first": "nan",
                },
            }
        )

    # Column selection: drop columns with too many NaNs.
    steps.append(
        {
            "name": "column_selector",
            "params": {"max_nan_frac_per_col": max_nan_frac},
        }
    )

    # Filter for columns with very few unique values or many zeros.
    steps.append(
        {
            "name": "low_unique_value_filter",
            "params": {
                "min_unique_value_count": min_unique_value_count,
                "max_col_zero_frac": max_col_zero_frac,
            },
        }
    )

    # Optional angle transformer (e.g., degrees => sin/cos).
    if angle_columns:
        steps.append(
            {
                "name": "angle_transformer",
                "params": {"angles": angle_columns},
            }
        )

    # Explicit imputer; adding it avoids relying on DataPreprocessor defaults.
    steps.append(
        {
            "name": "simple_imputer",
            "params": {"strategy": imputer_strategy},
        }
    )

    # Final scaler with aliases supported for convenience.
    scaler_key = scaler.lower()
    if scaler_key in ("standard", "standardize", "standard_scaler"):
        steps.append({"name": "standard_scaler"})
    elif scaler_key in ("minmax", "minmax_scaler", "normalize"):
        steps.append({"name": "minmax_scaler"})
    else:
        raise ValueError(
            f"Unknown scaler '{scaler}'. Use 'standard' (aka 'standardize') or 'minmax'."
        )

    return steps


def _dump_yaml_if_requested(
    config: Dict[str, Any],
    output_path: Optional[Union[str, Path]],
) -> None:
    """
    Write the configuration dictionary to a YAML file if a path is provided.

    Args:
        config (Dict[str, Any]): The configuration dictionary to serialize.
        output_path (Optional[Union[str, Path]]): Destination path. If None, nothing is written.

    Raises:
        RuntimeError: If PyYAML is not installed but output_path is not None.
    """
    if output_path is None:
        return

    if yaml is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyYAML is not installed; install 'pyyaml' or set output_path=None."
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def generate_quickstart_config(
    output_path: Optional[Union[str, Path]] = "base_config.yaml",
    *,
    # Preprocessor configuration
    max_nan_frac: float = 0.05,
    min_unique_value_count: int = 2,
    max_col_zero_frac: float = 1.0,
    angle_columns: Optional[List[str]] = None,
    counter_columns: Optional[List[str]] = None,
    imputer_strategy: str = "mean",
    scaler: str = "standard",
    # Early stopping
    early_stopping: bool = False,
    validation_split: float = 0.2,
    # Thresholding
    threshold_quantile: float = 0.99,
    # Autoencoder defaults
    batch_size: int = 128,
    code_size: int = 20,
    epochs: int = 10,
    layers: Optional[List[int]] = None,
    learning_rate: float = 1e-3,
) -> Dict[str, Any]:
    """
    Generate a minimal, valid configuration for EnergyFaultDetector.

    This function returns a configuration dictionary that uses the steps-based
    DataPreprocessor and sensible defaults for training. It can also write the
    configuration to YAML if an output path is supplied.

    Example:
        from energy_fault_detector import FaultDetector, Config
        cfg = generate_quickstart_config(output_path=None)
        fault_detector = FaultDetector(config=Config(config_dict=cfg))

    Args:
        output_path (Optional[Union[str, Path]]): YAML output path; set None to return only the dict.
        max_nan_frac (float): Max fraction of missing values per column for selection. Default: 0.05
        min_unique_value_count (int): Minimal unique values required to keep a column. Default: 2
        max_col_zero_frac (float): Max fraction of zeros allowed in a column. Default: 1.0
        angle_columns (Optional[List[str]]): Optional columns to transform as angles (sin/cos). Default: None
        counter_columns (Optional[List[str]]): Optional counter columns to convert to differences. Default: None
        imputer_strategy (str): Strategy for SimpleImputer ("mean", "median", etc.). Default: mean
        scaler (str): Scaler selection ("standard" or "minmax"; common aliases allowed). Default: standard
        early_stopping (bool): Enable early stopping in the autoencoder training. Default: False
        validation_split (float): Fraction for validation in sklearn splitter (0 < val < 1).
        threshold_quantile (float): Quantile for the "quantile" threshold selector. Default: 0.99
        batch_size (int): Autoencoder batch size. Default: 128
        code_size (int): Bottleneck code size. Default: 20
        epochs (int): Number of training epochs. Default: 10
        layers (Optional[List[int]]): Autoencoder layer sizes; defaults to [200, 100, 50] if None.
        learning_rate (float): Optimizer learning rate.

    Returns:
        Dict[str, Any]: Configuration dictionary ready for Config(config_dict=...).

    Raises:
        ValueError: If early_stopping is True but validation_split is not in (0, 1).
    """
    if not (0 < validation_split < 1.0):
        raise ValueError("validation_split must be in (0, 1).")

    # Fallback layers if none provided by user
    if layers is None:
        layers = [200, 100, 50]

    # Build the preprocessor steps list
    steps = _build_preprocessor_steps(
        max_nan_frac=max_nan_frac,
        min_unique_value_count=min_unique_value_count,
        max_col_zero_frac=max_col_zero_frac,
        angle_columns=angle_columns,
        counter_columns=counter_columns,
        imputer_strategy=imputer_strategy,
        scaler=scaler,
    )

    # Assemble training configuration
    train_config: Dict[str, Any] = {
        "data_preprocessor": {"steps": steps},
        "data_splitter": {
            "type": "sklearn",
            "validation_split": validation_split,
            "shuffle": True,
        },
        "autoencoder": {
            "name": "default",
            "params": {
                "batch_size": batch_size,
                "code_size": code_size,
                "early_stopping": early_stopping,
                "epochs": epochs,
                "layers": layers,
                "learning_rate": learning_rate,
            },
            "verbose": 1,
        },
        "anomaly_score": {"name": "rmse"},
        "threshold_selector": {
            "fit_on_val": False,
            "name": "quantile",
            "params": {"quantile": threshold_quantile},
        },
        # Optional clipping (disabled by default; uncomment to enable):
        # "data_clipping": {"lower_percentile": 0.001, "upper_percentile": 0.999},
    }

    config: Dict[str, Any] = {"train": train_config}

    # Optionally write YAML
    _dump_yaml_if_requested(config=config, output_path=output_path)

    return config
