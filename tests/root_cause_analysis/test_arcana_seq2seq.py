import unittest

import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Model as KerasModel

from energy_fault_detector.root_cause_analysis.arcana import Arcana
from energy_fault_detector.autoencoders.seq2seq_autoencoder import Seq2SeqAutoencoder
from energy_fault_detector.data_splitting.sequence_dataset import SequenceDatasetBuilder


class MockSeq2SeqAutoencoder(Seq2SeqAutoencoder):
    """A minimal concrete Seq2SeqAutoencoder for testing ARCANA."""
    
    def __init__(self, sequence_builder=None, conditional_features=None, **kwargs):
        super().__init__(sequence_builder=sequence_builder, conditional_features=conditional_features, **kwargs)
        
    def create_model(self, input_dimension, condition_dimension=None, **kwargs):
        seq_len, n_features = input_dimension
        
        main_input = layers.Input(shape=(seq_len, n_features), name="main_input")
        x = main_input
        
        if condition_dimension:
            cond_input = layers.Input(shape=(seq_len, condition_dimension), name="cond_input")
            x = layers.Concatenate(axis=-1)([main_input, cond_input])
            inputs = [main_input, cond_input]
        else:
            inputs = main_input
            
        # Encoder
        encoded = layers.LSTM(8, return_sequences=True, name="encoded")(x)
        
        # Decoder
        decoded = layers.LSTM(n_features, return_sequences=True, name="reconstruction")(encoded)
        
        self.model = KerasModel(inputs=inputs, outputs=decoded)
        return self.model


class TestArcanaSeq2Seq(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        n_samples = 100
        n_features = 3
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        timestamps = pd.date_range("2023-01-01", periods=n_samples, freq="10Min")
        self.df = pd.DataFrame(data, index=timestamps, columns=[f"f{i}" for i in range(n_features)])
        
        # Build model
        self.seq_len = 5
        self.sb = SequenceDatasetBuilder(
            sequence_length=self.seq_len, 
            ts_freq=np.timedelta64(10, 'm'), 
            overlap=self.seq_len - 1
        )
        self.model = MockSeq2SeqAutoencoder(
            sequence_builder=self.sb,
            epochs=1,
            batch_size=32
        )
        # Fit on a few samples just to initialize shapes and compile
        self.model.fit(self.df)

    def test_arcana_seq2seq_basic(self):
        arcana = Arcana(model=self.model, num_iter=10, alpha=0.5)
        # We use a subset of data to be fast
        subset = self.df.iloc[:20]
        x_bias, losses, tracked_bias = arcana.find_arcana_bias(subset, track_losses=True, track_bias=True)
        
        # Check shapes
        # For Seq2Seq, we expect one row per original timestamp (if overlapping enough)
        # SequenceDatasetBuilder.build_sliding_dataset uses sequences_to_dataframe with mode="first_full_rest_last"
        # For subset size 20 and seq_len 5, we expect 20 rows.
        self.assertEqual(len(x_bias), 20)
        self.assertEqual(x_bias.shape[1], 3)
        self.assertIsInstance(x_bias.index, pd.DatetimeIndex)
        
        # Check tracked losses
        self.assertFalse(losses.empty)
        self.assertIn('Combined Loss', losses.columns)
        
        # Check tracked bias
        self.assertGreater(len(tracked_bias), 0)
        self.assertIsInstance(tracked_bias[0], pd.DataFrame)
        self.assertEqual(tracked_bias[0].shape, (20, 3))

    def test_arcana_seq2seq_conditional(self):
        # Add a conditional feature
        cond_df = self.df.copy()
        cond_df['cond'] = np.random.randn(len(cond_df)).astype(np.float32)
        
        model_cond = MockSeq2SeqAutoencoder(
            sequence_builder=self.sb,
            conditional_features=['cond'],
            epochs=1,
            batch_size=32
        )
        model_cond.fit(cond_df)
        
        arcana = Arcana(model=model_cond, num_iter=5, alpha=0.5)
        subset = cond_df.iloc[:15]
        x_bias, _, _ = arcana.find_arcana_bias(subset)
        
        # Should have 15 rows and 3 main features
        self.assertEqual(x_bias.shape, (15, 3))
        self.assertNotIn('cond', x_bias.columns)

    def test_draw_samples_seq2seq(self):
        # Test drawing samples from 3D inputs
        arcana = Arcana(model=self.model, max_sample_threshold=5)
        # Manually create 3D input (n_windows, seq_len, n_features)
        x_3d = np.random.randn(10, self.seq_len, 3).astype(np.float32)
        selection = arcana.draw_samples(x_3d)
        
        self.assertEqual(len(selection), 10)
        self.assertEqual(np.sum(selection), 5)


if __name__ == "__main__":
    unittest.main()
