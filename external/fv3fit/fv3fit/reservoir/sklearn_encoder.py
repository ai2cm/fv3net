import numpy as np
import os
from typing import Sequence, Iterable, Hashable
from fv3fit._shared import (
    get_dir,
    put_dir,
    io,
)

from fv3fit.keras import PureKerasModel
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import joblib


class SkTransformAutoencoder:
    _TRANSFORMER_NAME = "sk_transformer.pkl"
    _SCALER_NAME = "sk_scaler.pkl"

    def __init__(
        self,
        transformer: TransformerMixin,
        scaler: StandardScaler,
        enforce_positive_outputs: bool = False,
    ):
        self.transformer = transformer
        self.scaler = scaler
        self.enforce_positive_outputs = enforce_positive_outputs

    def predict(self, x):
        original_feature_sizes = [feature.shape[-1] for feature in x]
        encoded = self.encode(np.concatenate(x, axis=-1))
        decoded = self.decode(encoded)

        if self.enforce_positive_outputs is True:
            decoded = np.where(decoded >= 0, decoded, 0.0)

        decoded_split_features = []
        start = 0
        for feature_size in original_feature_sizes:
            decoded_split_features.append(decoded[:, start : start + feature_size])
            start += feature_size
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

    @classmethod
    def load(cls, path: str) -> "SkTransformAutoencoder":
        with get_dir(path) as model_path:
            transformer = joblib.load(os.path.join(model_path, cls._TRANSFORMER_NAME))
            scaler = joblib.load(os.path.join(model_path, cls._SCALER_NAME))

        return cls(transformer=transformer, scaler=scaler)


@io.register("sktransform-autoencoder")
class SkTransformAutoencoderModel(PureKerasModel):
    """ Mostly identical to the PureKerasModel but takes an
    SklearnAutoencoder as its model and calls SklearnAutoencoder.dump/load
    instead of keras.model.save and keras.load. This is because input
    shape information is somehow not saved and loaded correctly if the
    encoder/decoder are saved as one model.
    """

    _MODEL_NAME = "sk_transformer"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        autoencoder: SkTransformAutoencoder,
        unstacked_dims: Sequence[str],
        n_halo: int = 0,
    ):
        if set(input_variables) != set(output_variables):
            raise ValueError(
                "input_variables and output_variables must be the same set."
            )
        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            model=autoencoder,
            unstacked_dims=unstacked_dims,
            n_halo=n_halo,
        )

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self.model is not None:
                self.model.dump(os.path.join(path, self._MODEL_NAME))
            with open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
                f.write(
                    yaml.dump(
                        {
                            "input_variables": self.input_variables,
                            "output_variables": self.output_variables,
                            "unstacked_dims": self._unstacked_dims,
                            "n_halo": self._n_halo,
                        }
                    )
                )

    @classmethod
    def load(cls, path: str) -> "SkTransformAutoencoderModel":
        """Load a serialized model from a directory."""
        with get_dir(path) as path:
            model_filename = os.path.join(path, cls._MODEL_NAME)
            autoencoder = SkTransformAutoencoder.load(model_filename)
            with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            obj = cls(
                input_variables=config["input_variables"],
                output_variables=config["output_variables"],
                autoencoder=autoencoder,
                unstacked_dims=config.get("unstacked_dims", None),
                n_halo=config.get("n_halo", 0),
            )

            return obj

    def input_sensitivity(self, stacked_sample):
        """Calculate sensitivity to input features."""
        raise NotImplementedError(
            "Input_sensitivity is not implemented for PureKerasAutoencoderModel."
        )
