import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from typing import Sequence
import yaml

from fv3fit._shared import (
    get_dir,
    put_dir,
)
from fv3fit._shared import io

io.register("sk-transformer")


class SkTransformer:
    """ Used to encode higher-dimension inputs into a
    lower dimension latent space and decode latent vectors
    back to the original feature space.

    """

    _TRANSFORMER_NAME = "sk_transformer.pkl"
    _SCALER_NAME = "sk_scaler.pkl"
    _METADATA_NAME = "metadata.yaml"

    def __init__(
        self,
        transformer: TransformerMixin,
        scaler: StandardScaler,
        enforce_positive_outputs: bool = False,
    ):
        self.transformer = transformer
        self.scaler = scaler
        self.enforce_positive_outputs = enforce_positive_outputs

    def predict(self, x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        original_feature_sizes = [feature.shape[-1] for feature in x]
        encoded = self.encode(np.concatenate(x, axis=-1))
        decoded = self.decode(encoded)

        if self.enforce_positive_outputs is True:
            decoded = np.where(decoded >= 0, decoded, 0.0)

        decoded_split_features = np.split(
            decoded, np.cumsum(original_feature_sizes[:-1])
        )
        return decoded_split_features

    def encode(self, x):
        return self.transformer.transform(self.scaler.transform(x))

    def decode(self, c):
        return self.scaler.inverse_transform(self.transformer.inverse_transform(c))

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            transformer_path = os.path.join(path, self._TRANSFORMER_NAME)
            joblib.dump(self.transformer, transformer_path)
            scaler_path = os.path.join(path, self._SCALER_NAME)
            joblib.dump(self.scaler, scaler_path)

            with open(os.path.join(path, self._METADATA_NAME), "w") as f:
                f.write(
                    yaml.dump(
                        {"enforce_positive_outputs": self.enforce_positive_outputs}
                    )
                )

    @classmethod
    def load(cls, path: str) -> "SkTransformer":
        with get_dir(path) as model_path:
            transformer = joblib.load(os.path.join(model_path, cls._TRANSFORMER_NAME))
            scaler = joblib.load(os.path.join(model_path, cls._SCALER_NAME))
            with open(os.path.join(path, cls._METADATA_NAME), "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
        return cls(
            transformer=transformer,
            scaler=scaler,
            enforce_positive_outputs=config["enforce_positive_outputs"],
        )
