import numpy as np
import os
import tensorflow as tf
from typing import Union
from fv3fit._shared import get_dir, put_dir


class Autoencoder(tf.keras.Model):
    _ENCODER_NAME = "encoder.tf"
    _DECODER_NAME = "decoder.tf"

    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model):
        super(Autoencoder, self).__init__()
        self.n_latent_dims = encoder.layers[-1].output.shape[-1]
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        return self.encoder.predict(x)

    def decode(self, latent_x: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        return self.decoder.predict(latent_x)

    @classmethod
    def load(cls, path: str) -> "Autoencoder":
        with get_dir(path) as model_path:
            encoder = tf.keras.models.load_model(
                os.path.join(model_path, cls._ENCODER_NAME)
            )
            decoder = tf.keras.models.load_model(
                os.path.join(model_path, cls._DECODER_NAME)
            )
        return cls(encoder=encoder, decoder=decoder)

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            encoder_filename = os.path.join(path, self._ENCODER_NAME)
            self.encoder.save(encoder_filename)
            decoder_filename = os.path.join(path, self._DECODER_NAME)
            self.decoder.save(decoder_filename)
