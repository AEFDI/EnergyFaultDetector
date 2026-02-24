import unittest

from energy_fault_detector.config import Config
from energy_fault_detector.core.model_factory import ModelFactory


class TestInvalidPreprocessorConfig(unittest.TestCase):
    def test_unknown_step_name_raises(self) -> None:
        """Mistyped step names in data_preprocessor.steps should raise a clear ValueError."""

        cfg_dict = {
            "train": {
                "anomaly_score": {
                    "name": "rmse",
                },
                "data_preprocessor": {
                    "steps": [
                        {"name": "colum_selector", "params": {"max_nan_frac_per_col": 0.2}},  # typo
                    ]
                },
                "autoencoder": {
                    "name": "MultilayerAutoencoder",
                    "params": {
                        "layers": [10],
                        "code_size": 2,
                        "batch_size": 8,
                        "epochs": 1,
                        "learning_rate": 1e-3,
                        "loss_name": "mean_squared_error",
                    },
                },
                "threshold_selector": {
                    "name": "quantile",
                    "params": {"quantile": 0.95},
                },
            }
        }

        conf = Config(config_dict=cfg_dict)

        with self.assertRaises(ValueError) as ctx:
            _ = ModelFactory(conf)

        msg = str(ctx.exception)
        self.assertIn("Unknown step name", msg)
        self.assertIn("colum_selector", msg)
