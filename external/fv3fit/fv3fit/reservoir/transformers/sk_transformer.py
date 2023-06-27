import fsspec
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

from typing import Sequence
import yaml

from fv3fit._shared.predictor import Reloadable
from fv3fit._shared import get_dir, put_dir, io
from fv3fit._shared.xr_prediction import ArrayPredictor
from fv3fit.reservoir.transformers.transformer import Transformer


@io.register("sk-transformer")
class SkTransformer(Transformer, ArrayPredictor, Reloadable):
    """ Used to encode higher-dimension inputs into a
    lower dimension latent space and decode latent vectors
    back to the original feature space.

    This class has a predict method, which returns the round trip
    of the encoded/decoded input. It is useful for checking whether
    the transformer introduces error that affects its usage online.
    In actual production models, this class is mean to be used
    in a similar manner to the DenseAutoencoder, where its encode
    and decode methods are used to transform in and out of a reduced
    dimensional space.
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

    @property
    def n_latent_dims(self):
        return self.transformer.n_components

    def _ensure_sample_dim(self, x):
        # Sklearn scalers and transforms expect the first dimension
        # to be sample. If not present, i.e. a single sample is provided,
        # reshape to add this dimension.

        if x.shape[-1] == self.scaler.n_features_in_:
            if x.ndim == 1:
                return x.reshape(1, -1)
            elif x.ndim == 2:
                return x
        else:
            raise ValueError(
                f"Input has shape {x.shape}, must have either "
                f"({self.scaler.n_features_in_},) or "
                f"(samples, {self.scaler.n_features_in_})."
            )

    def predict(self, x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        original_feature_sizes = [feature.shape[-1] for feature in x]
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        decoded_split_features = np.split(
            decoded, np.cumsum(original_feature_sizes[:-1]), axis=-1
        )

        return decoded_split_features

    def encode(self, x):
        # input is a sequence of time series for each variables: [var, time, feature]
        x_concat = np.concatenate(x, axis=-1)
        x = self._ensure_sample_dim(x_concat)
        return self.transformer.transform(self.scaler.transform(x))

    def decode(self, c):
        decoded = self.transformer.inverse_transform(c)
        decoded = self.scaler.inverse_transform(self._ensure_sample_dim(decoded))

        if self.enforce_positive_outputs is True:
            decoded = np.where(decoded >= 0, decoded, 0.0)

        return decoded

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
            with fsspec.open(os.path.join(path, cls._METADATA_NAME), "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
        return cls(
            transformer=transformer,
            scaler=scaler,
            enforce_positive_outputs=config["enforce_positive_outputs"],
        )
