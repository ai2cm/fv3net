import fsspec
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

from typing import Sequence, Optional
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
        original_feature_sizes: Optional[Sequence[int]] = None,
    ):
        self.transformer = transformer
        self.scaler = scaler
        self.enforce_positive_outputs = enforce_positive_outputs
        self.original_feature_sizes = original_feature_sizes

    @property
    def n_latent_dims(self):
        return self.transformer.n_components

    def _ensure_sample_dim(self, x: np.ndarray):
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
        encoded = self.encode(x)
        return self.decode(encoded)

    def encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        # input is a sequence of variable data with dims [..., feature_dim]

        self._set_feature_sizes_if_not_present(x)
        x_concat = np.concatenate(x, axis=-1)
        x_with_sample_dim = self._ensure_sample_dim(x_concat)

        return self.transformer.transform(self.scaler.transform(x_with_sample_dim))

    def decode(self, c: np.ndarray) -> Sequence[np.ndarray]:
        feature_sizes = self._get_original_feature_sizes()
        decoded = self.transformer.inverse_transform(c)
        decoded = self.scaler.inverse_transform(self._ensure_sample_dim(decoded))

        if self.enforce_positive_outputs is True:
            decoded = np.where(decoded >= 0, decoded, 0.0)
        decoded_split_features = np.split(
            decoded, np.cumsum(feature_sizes[:-1]), axis=-1
        )

        return decoded_split_features

    def _set_feature_sizes_if_not_present(self, x: Sequence[np.ndarray]):
        if self.original_feature_sizes is None:
            self.original_feature_sizes = [var.shape[-1] for var in x]

    def _get_original_feature_sizes(self):
        if self.original_feature_sizes is None:
            raise ValueError("Feature sizes must be set before calling decode")
        else:
            return self.original_feature_sizes

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            transformer_path = os.path.join(path, self._TRANSFORMER_NAME)
            joblib.dump(self.transformer, transformer_path)
            scaler_path = os.path.join(path, self._SCALER_NAME)
            joblib.dump(self.scaler, scaler_path)

            with open(os.path.join(path, self._METADATA_NAME), "w") as f:
                f.write(
                    yaml.dump(
                        {
                            "enforce_positive_outputs": self.enforce_positive_outputs,
                            "original_feature_sizes": self.original_feature_sizes,
                        }
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
            original_feature_sizes=config["original_feature_sizes"],
        )
