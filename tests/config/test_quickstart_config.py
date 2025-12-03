from unittest import TestCase
from typing import Any, Dict

from energy_fault_detector.config.config import Config
from energy_fault_detector.config.quickstart_config import generate_quickstart_config  # adjust


class TestQuickstartConfig(TestCase):
    def test_generate_quickstart_config_valid_dict(self) -> None:
        """Should return a valid config dict that Config accepts and includes required sections."""
        cfg: Dict[str, Any] = generate_quickstart_config(
            output_path=None,
            angle_columns=["theta_deg"],
            counter_columns=["energy_total_kwh"],
            scaler="standard",
            imputer_strategy="mean",
            early_stopping=False,
        )

        # Basic structure checks
        self.assertIn("train", cfg)
        train = cfg["train"]
        self.assertIn("data_preprocessor", train)
        self.assertIn("steps", train["data_preprocessor"])
        self.assertIn("autoencoder", train)
        self.assertIn("params", train["autoencoder"])
        self.assertIn("threshold_selector", train)
        self.assertIn("params", train["threshold_selector"])

        # Ensure certain steps exist
        step_names = [s["name"] for s in train["data_preprocessor"]["steps"]]
        self.assertIn("column_selector", step_names)
        self.assertIn("simple_imputer", step_names)
        self.assertTrue(
            any(n in ("standard_scaler", "minmax_scaler") for n in step_names),
            "Expected a scaler step in the pipeline."
        )

        # Should not raise: validate via Config
        Config(config_dict=cfg)

    def test_generate_quickstart_config_validation_split_guard(self) -> None:
        """If validation split not in (0, 1) it should raise ValueError."""
        with self.assertRaises(ValueError):
            _ = generate_quickstart_config(
                validation_split=0.0,  # invalid by design
            )
