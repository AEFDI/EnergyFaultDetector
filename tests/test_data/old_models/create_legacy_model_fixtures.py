"""Create small saved-model fixtures for compatibility testing.

1. Create a python environment with a previous version of energy-fault-detector.
2. Run this script:
    python tests/test_data/old_models/create_legacy_model_fixtures.py \
        --output tests/test_data/old_models/py<version>_efd<version> \
        --purpose "Test compatibility for EFD <version> models."

It creates:

    old_models/
    ├── dense/
    ├── sequence/
    └── manifest.json

"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from energy_fault_detector import FaultDetector
from energy_fault_detector.config import Config


def make_dense_data(n_rows: int = 80) -> pd.DataFrame:
    """Create deterministic tabular/time-series data for the dense model.

    Args:
        n_rows: Number of samples to generate.

    Returns:
        Numeric DataFrame with a 10-minute DatetimeIndex.
    """
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=n_rows, freq="10min")
    phase = np.linspace(0, 4 * np.pi, n_rows)

    return pd.DataFrame(
        {
            "temperature": 50 + 3 * np.sin(phase) + rng.normal(0, 0.15, n_rows),
            "pressure": 10 + 0.5 * np.cos(phase) + rng.normal(0, 0.05, n_rows),
            "power": 100 + 8 * np.sin(phase / 2) + rng.normal(0, 0.3, n_rows),
        },
        index=index,
    )


def make_sequence_data(n_rows: int = 120) -> pd.DataFrame:
    """Create deterministic contiguous time-series data for a sequence model.

    Args:
        n_rows: Number of samples to generate.

    Returns:
        Numeric DataFrame with a regular 10-minute DatetimeIndex.
    """
    rng = np.random.default_rng(123)
    index = pd.date_range("2025-02-01", periods=n_rows, freq="10min")
    phase = np.linspace(0, 6 * np.pi, n_rows)

    return pd.DataFrame(
        {
            "sensor_a": np.sin(phase) + rng.normal(0, 0.03, n_rows),
            "sensor_b": np.cos(phase) + rng.normal(0, 0.03, n_rows),
            "sensor_c": 0.3 * np.sin(phase / 2) + rng.normal(0, 0.02, n_rows),
        },
        index=index,
    )


def dense_config() -> Config:
    """Build a small dense-autoencoder configuration."""
    return Config(
        config_dict={
            "train": {
                "data_preprocessor": None,
                "data_splitter": {
                    "type": "sklearn",
                    "validation_split": 0.2,
                    "shuffle": False,
                },
                "autoencoder": {
                    "name": "default",
                    "verbose": 0,
                    "params": {
                        "layers": [8],
                        "code_size": 3,
                        "batch_size": 16,
                        "epochs": 1,
                        "learning_rate": 0.001,
                        "loss_name": "mean_squared_error",
                        "early_stopping": False,
                    },
                },
                "anomaly_score": {
                    "name": "rmse",
                    "params": {"scale": False},
                },
                "threshold_selector": {
                    "name": "quantile",
                    "fit_on_val": False,
                    "params": {"quantile": 0.95},
                },
            }
        }
    )


def sequence_config() -> Config:
    """Build a small LSTM seq2one configuration."""
    return Config(
        config_dict={
            "train": {
                "data_preprocessor": {
                    "steps": [
                        {"name": "standard_scaler"},
                    ]
                },
                "data_splitter": {
                    "type": "sklearn",
                    "validation_split": 0.2,
                    "shuffle": False,
                },
                "autoencoder": {
                    "name": "lstm_seq2one",
                    "verbose": 0,
                    "params": {
                        "sequence_builder": {
                            "sequence_length": 8,
                            "ts_freq": "10m",
                            "stride": 1,
                            "pad_incomplete": False,
                            "pad_value": 0.0,
                        },
                        "layers": [8],
                        "decoder_layers": [8],
                        "code_size": 4,
                        "batch_size": 16,
                        "epochs": 1,
                        "learning_rate": 0.001,
                        "loss_name": "mean_squared_error",
                        "early_stopping": False,
                    },
                },
                "anomaly_score": {
                    "name": "rmse",
                    "params": {"scale": False},
                },
                "threshold_selector": {
                    "name": "quantile",
                    "fit_on_val": False,
                    "params": {"quantile": 0.95},
                },
            }
        }
    )


def train_and_save(
    name: str,
    config: Config,
    data: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Train, smoke-test, and save one model.

    Args:
        name: Fixture name.
        config: Model configuration.
        data: Training and prediction data.
        output_dir: Directory where the fixture should be saved.

    Returns:
        Directory containing the saved model.
    """
    fixture_dir = output_dir / name
    detector = FaultDetector(config=config, model_directory=fixture_dir)

    normal_index = pd.Series(True, index=data.index)
    metadata = detector.fit(
        sensor_data=data,
        normal_index=normal_index,
        save_models=True,
        overwrite_models=True,
    )

    result = detector.predict(data)

    if result.anomaly_score.empty:
        raise RuntimeError(f"{name}: prediction unexpectedly produced no anomaly scores.")

    if not result.predicted_anomalies.index.equals(result.anomaly_score.index):
        raise RuntimeError(f"{name}: anomaly predictions and scores have different indices.")

    saved_path = Path(metadata.model_path)
    print(f"Created {name} fixture: {saved_path}")
    print(
        f"  scores={len(result.anomaly_score)}, "
        f"predicted_anomalies={int(result.predicted_anomalies.sum())}"
    )

    return saved_path


def installed_version(distribution: str) -> str | None:
    """Return an installed package version if available."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def main() -> None:
    """Create dense and sequence saved-model compatibility fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--purpose",
        type=str,
        default="test",
        help="Purpose of the saved-model fixtures.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/test_data/old_models"),
        help="Output directory for saved fixtures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output.resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)

    dense_path = train_and_save(
        name="dense",
        config=dense_config(),
        data=make_dense_data(),
        output_dir=output_dir,
    )
    sequence_path = train_and_save(
        name="sequence",
        config=sequence_config(),
        data=make_sequence_data(),
        output_dir=output_dir,
    )

    manifest = {
        "purpose": args.purpose,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "energy-fault-detector": installed_version("energy-fault-detector"),
            "tensorflow": installed_version("tensorflow"),
            "keras": installed_version("keras"),
            "pydantic": installed_version("pydantic"),
            "scikit-learn": installed_version("scikit-learn"),
            "pandas": installed_version("pandas"),
            "numpy": installed_version("numpy"),
        },
        "fixtures": {
            "dense": str(dense_path.relative_to(output_dir)),
            "sequence": str(sequence_path.relative_to(output_dir)),
        },
        "data_generation": {
            "dense": "make_dense_data(n_rows=80)",
            "sequence": "make_sequence_data(n_rows=120)",
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nManifest written to: {manifest_path}")
    print("\nLater compatibility smoke test:")
    print(f'  FaultDetector.load("{dense_path}")')
    print(f'  FaultDetector.load("{sequence_path}")')


if __name__ == "__main__":
    main()
