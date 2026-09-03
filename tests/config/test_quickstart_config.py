from unittest import TestCase
from typing import Any, Dict
from pathlib import Path
import tempfile
import shutil

from energy_fault_detector.config.config import Config
from energy_fault_detector.config.quickstart_config import generate_quickstart_config


class TestQuickstartConfig(TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_generate_quickstart_config_valid_dict(self) -> None:
        """Should return a valid config dict that Config accepts and includes required sections."""
        cfg: Config = generate_quickstart_config(
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
        self.assertIn("scaler", step_names)

    def test_generate_quickstart_config_validation_split_guard(self) -> None:
        """If validation split not in (0, 1) it should raise ValueError."""
        with self.assertRaises(ValueError):
            _ = generate_quickstart_config(
                validation_split=0.0,  # invalid by design
            )

    def test_save_load_quickstart_config(self) -> None:
        """Test that generated config can be saved and loaded back correctly."""
        config_path = Path(self.test_dir) / "config.yaml"
        cfg: Config = generate_quickstart_config(
            output_path=config_path,
            angle_columns=["theta_deg"],
            early_stopping=True,
            validation_split=0.25
        )

        self.assertTrue(config_path.exists())

        # Load the config back
        loaded_cfg = Config(config_path)

        # Compare dictionaries
        self.assertEqual(cfg.config_dict, loaded_cfg.config_dict)
        self.assertEqual(loaded_cfg["train"]["data_splitter"]["validation_split"], 0.25)
