import unittest

from energy_fault_detector.config import Config
from energy_fault_detector.core.model_factory import ModelFactory


class TestInvalidSequenceBuilderConfig(unittest.TestCase):
    def test_missing_ts_freq_raises_keyerror(self) -> None:
        """sequence_builder without ts_freq should fail when initializing the autoencoder."""

        cfg_dict = {
            "train": {
                "anomaly_score": {"name": "rmse"},
                "data_preprocessor": {},  # default pipeline
                "autoencoder": {
                    "name": "LSTMSeq2OneAutoencoder",
                    "params": {
                        "sequence_builder": {
                            "sequence_length": 10,
                            "stride": 1,
                            # 'ts_freq' missing
                            "pad_incomplete": False,
                            "pad_value": 0.0,
                        },
                        "layers": [8, 4],
                        "dropout_rate": 0.0,
                        "regularization": 0.01,
                        "stateful": False,
                        "learning_rate": 1e-3,
                        "batch_size": 8,
                        "epochs": 1,
                        "loss_name": "mean_squared_error",
                    },
                },
                "threshold_selector": {"name": "quantile", "params": {"quantile": 0.95}},
            }
        }

        conf = Config(config_dict=cfg_dict)

        with self.assertRaises(KeyError) as ctx:
            _ = ModelFactory(conf)

        # Keep current behavior stable
        self.assertIn("ts_freq", str(ctx.exception))

    def test_negative_sequence_length_raises(self) -> None:
        """sequence_length <= 0 should raise ValueError from SequenceDatasetBuilder."""

        cfg_dict = {
            "train": {
                "anomaly_score": {"name": "rmse"},
                "data_preprocessor": {},
                "autoencoder": {
                    "name": "LSTMSeq2OneAutoencoder",
                    "params": {
                        "sequence_builder": {
                            "sequence_length": 0,  # invalid
                            "stride": 1,
                            "ts_freq": "30s",
                            "pad_incomplete": False,
                            "pad_value": 0.0,
                        },
                        "layers": [8, 4],
                        "dropout_rate": 0.0,
                        "regularization": 0.01,
                        "stateful": False,
                        "learning_rate": 1e-3,
                        "batch_size": 8,
                        "epochs": 1,
                        "loss_name": "mean_squared_error",
                    },
                },
                "threshold_selector": {"name": "quantile", "params": {"quantile": 0.95}},
            }
        }

        conf = Config(config_dict=cfg_dict)

        with self.assertRaises(ValueError) as ctx:
            _ = ModelFactory(conf)

        self.assertIn("sequence_length must be > 0", str(ctx.exception))
