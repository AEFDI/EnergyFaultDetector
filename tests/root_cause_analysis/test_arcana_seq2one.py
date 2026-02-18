import unittest

import numpy as np
import pandas as pd

from energy_fault_detector.root_cause_analysis.arcana import Arcana
from energy_fault_detector.autoencoders import LSTMSeq2OneAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class TestArcanaSeq2One(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        n_samples = 100
        n_features = 3
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        timestamps = pd.date_range("2023-01-01", periods=n_samples, freq="10Min")
        self.df = pd.DataFrame(data, index=timestamps, columns=[f"f{i}" for i in range(n_features)])
        
        # Build model
        self.seq_len = 5
        sb = SequenceDatasetBuilder(sequence_length=self.seq_len, ts_freq=np.timedelta64(10, 'm'), overlap=self.seq_len - 1)
        self.model = LSTMSeq2OneAutoencoder(
            sequence_builder=sb,
            layers=[16, 8],
            epochs=1,
            batch_size=32
        )
        # Fit on a few samples just to initialize shapes
        self.model.fit(self.df)

    def test_arcana_seq2one_basic(self):
        arcana = Arcana(model=self.model, num_iter=10, alpha=0.5)
        # We test find_arcana_bias
        # We use a subset of data to be fast
        subset = self.df.iloc[:20]
        x_bias, losses, tracked_bias = arcana.find_arcana_bias(subset, track_losses=True, track_bias=True)
        
        # Check shapes
        # For Seq2One, we expect one row per valid window.
        # With seq_len=5, subset size 20, we expect 20 - 5 + 1 = 16 windows.
        expected_rows = 16
        self.assertEqual(len(x_bias), expected_rows)
        self.assertEqual(x_bias.shape[1], 3)
        self.assertIsInstance(x_bias.index, pd.DatetimeIndex)
        
        # Check tracked losses
        self.assertFalse(losses.empty)
        self.assertIn('Combined Loss', losses.columns)
        
        # Check tracked bias
        self.assertGreater(len(tracked_bias), 0)
        self.assertIsInstance(tracked_bias[0], pd.DataFrame)
        self.assertEqual(tracked_bias[0].shape, (expected_rows, 3))

    def test_draw_samples_sequential(self):
        # Test the draw_samples bug fix
        arcana = Arcana(model=self.model, max_sample_threshold=5)
        # Manually create 3D input
        x_3d = np.random.randn(10, self.seq_len, 3).astype(np.float32)
        selection = arcana.draw_samples(x_3d)
        
        self.assertEqual(len(selection), 10)
        self.assertEqual(np.sum(selection), 5)
        self.assertEqual(selection.dtype, bool)


if __name__ == "__main__":
    unittest.main()
