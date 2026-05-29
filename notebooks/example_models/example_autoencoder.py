from typing import List, Union, Tuple

from keras import Model as KerasModel
from keras import layers

from energy_fault_detector.autoencoders import MultilayerAutoencoder


class NewAE(MultilayerAutoencoder):
    """An autoencoder model that is not symmetric."""

    def __init__(self, encoder_layers: List[int] = None, decoder_layers: List[str] = None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.encoder_layers = [32, 32] if encoder_layers is None else encoder_layers
        self.decoder_layers = [8, 16] if decoder_layers is None else decoder_layers

    def create_model(self, input_dimension: Union[int, Tuple], **kwargs) -> KerasModel:

        inputs = layers.Input(shape=(input_dimension,))
        x = layers.Dense(units=self.encoder_layers[0], activation='relu')(inputs)
        for n_units in self.encoder_layers[1:]:
            x = layers.Dense(units=n_units, activation='relu')(x)

        encoded = layers.Dense(units=self.code_size, activation='relu', name='encoded')(x)

        x = layers.Dense(units=self.decoder_layers[0], activation='relu')(encoded)
        for n_units in self.decoder_layers[1:]:
            x = layers.Dense(units=n_units, activation='relu')(x)

        decoded = layers.Dense(units=input_dimension, activation='sigmoid')(x)
        self.model = KerasModel(inputs=inputs, outputs=decoded)
        return self.model
