from fv3fit._shared.packer import ArrayPacker
from fv3fit._shared import (
    Predictor,
    io,
    match_prediction_to_input_coords,
    SAMPLE_DIM_NAME,
    stack_non_vertical,
)
from fv3fit._shared.config import RegularizerConfig
import tensorflow as tf
from typing import Any, Dict, Hashable, Iterable, Optional, Sequence
import xarray as xr
import os
from ._filesystem import get_dir, put_dir
import yaml
import dataclasses
import tensorflow_addons as tfa


def get_input_vector(
    packer: ArrayPacker, n_window: Optional[int] = None, series: bool = True,
):
    """
    Given a packer, return a list of input layers with one layer
    for each input used by the packer, and a list of output tensors which are
    the result of packing those input layers.

    Args:
        packer
        n_window: required if series is True, number of timesteps in a sample
        series: if True, returned inputs have shape [n_window, n_features], otherwise
            they are 1D [n_features] arrays
    """
    features = [packer.feature_counts[name] for name in packer.pack_names]
    if series:
        if n_window is None:
            raise TypeError("n_window is required if series is True")
        input_layers = [
            tf.keras.layers.Input(shape=[n_window, n_features])
            for n_features in features
        ]
    else:
        input_layers = [
            tf.keras.layers.Input(shape=[n_features]) for n_features in features
        ]
    packed = tf.keras.layers.Concatenate()(input_layers)
    return input_layers, packed


@dataclasses.dataclass
class DenseNetwork:
    """
    Attributes:
        output: final output of dense network
        hidden_outputs: consecutive outputs of hidden layers in dense network
    """

    output: tf.Tensor
    hidden_outputs: Sequence[tf.Tensor]


@dataclasses.dataclass
class DenseNetworkConfig:
    width: int = 8
    depth: int = 3
    kernel_regularizer: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )
    gaussian_noise: float = 0.0
    spectral_normalization: bool = False

    def build(
        self, x_in: tf.Tensor, n_features_out: int, label: str = ""
    ) -> DenseNetwork:
        """
        Take an input tensor to a dense network and return the result of a dense
        network's prediction, as a tensor.

        Can be used within code that builds a larger neural network. This should
        take in and return normalized values.

        Args:
            x_in: input tensor whose last dimension is the feature dimension
            n_features_out: dimensionality of last (feature) dimension of output
            config: configuration for dense network
            label: inserted into layer names, if this function is used multiple times
                to build one network you must provide a different label each time
        
        Returns:
            tensor resulting from the requested dense network
        """
        hidden_outputs = []
        x = x_in
        for i in range(self.depth - 1):
            if self.gaussian_noise > 0.0:
                x = tf.keras.layers.GaussianNoise(
                    self.gaussian_noise, name=f"gaussian_noise_{label}_{i}"
                )(x)
            hidden_layer = tf.keras.layers.Dense(
                self.width,
                activation=tf.keras.activations.relu,
                kernel_regularizer=self.kernel_regularizer.instance,
                name=f"hidden_{label}_{i}",
            )
            if self.spectral_normalization:
                hidden_layer = tfa.layers.SpectralNormalization(
                    hidden_layer, name=f"spectral_norm_{label}_{i}"
                )
            x = hidden_layer(x)
            hidden_outputs.append(x)
        output = tf.keras.layers.Dense(
            n_features_out, activation="linear", name=f"dense_network_{label}_output",
        )(x)
        return DenseNetwork(hidden_outputs=hidden_outputs, output=output)


@io.register("all-keras")
class PureKerasModel(Predictor):
    """Model which uses Keras for packing and normalization"""

    _MODEL_FILENAME = "model.tf"
    _CONFIG_FILENAME = "config.yaml"
    custom_objects: Dict[str, Any] = {}

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        output_metadata: Iterable[Dict[str, Any]],
        model: tf.keras.Model,
    ):
        """Initialize the predictor
        
        Args:
            sample_dim_name: name of sample dimension
            input_variables: names of input variables
            output_variables: names of output variables
            output_metadata: attributes and stacked dimension order for each variable
                in output_variables
        """
        super().__init__(sample_dim_name, input_variables, output_variables)
        self.sample_dim_name = sample_dim_name
        self.input_variables = input_variables
        self.output_variables = output_variables
        self._output_metadata = output_metadata
        self.model = model

    @classmethod
    def load(cls, path: str) -> "PureKerasModel":
        """Load a serialized model from a directory."""
        with get_dir(path) as path:
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            model = tf.keras.models.load_model(
                model_filename, custom_objects=cls.custom_objects
            )
            with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            obj = cls(
                config["sample_dim_name"],
                config["input_variables"],
                config["output_variables"],
                config.get("output_metadata", None),
                model,
            )
            return obj

    def _array_prediction_to_dataset(
        self, names, outputs, stacked_output_metadata, stacked_coords
    ) -> xr.Dataset:
        ds = xr.Dataset()
        for name, output, metadata in zip(names, outputs, stacked_output_metadata):
            scalar_output_as_singleton_dim = (
                len(metadata["dims"]) == 1
                and len(output.shape) == 2
                and output.shape[1] == 1
            )
            if scalar_output_as_singleton_dim:
                output = output[:, 0]  # remove singleton dimension
            da = xr.DataArray(
                data=output,
                dims=[SAMPLE_DIM_NAME] + list(metadata["dims"][1:]),
                coords={SAMPLE_DIM_NAME: stacked_coords},
            ).unstack(SAMPLE_DIM_NAME)
            dim_order = [dim for dim in metadata["dims"] if dim in da.dims]
            ds[name] = da.transpose(*dim_order, ...)
        return ds

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        X_stacked = stack_non_vertical(X)
        inputs = [X_stacked[name].values for name in self.input_variables]
        outputs = self.model.predict(inputs)
        if self._output_metadata is not None:
            ds = self._array_prediction_to_dataset(
                self.output_variables,
                outputs,
                self._output_metadata,
                X_stacked.coords[SAMPLE_DIM_NAME],
            )
            return ds

        else:
            # workaround for saved datasets which do not have output metadata
            # from an initial version of the BPTT code. Can be removed
            # eventually
            dQ1, dQ2 = outputs
            ds = xr.Dataset(
                data_vars={
                    "dQ1": xr.DataArray(
                        dQ1,
                        dims=X_stacked["air_temperature"].dims,
                        coords=X_stacked["air_temperature"].coords,
                        attrs={"units": X_stacked["air_temperature"].units + " / s"},
                    ),
                    "dQ2": xr.DataArray(
                        dQ2,
                        dims=X_stacked["specific_humidity"].dims,
                        coords=X_stacked["specific_humidity"].coords,
                        attrs={"units": X_stacked["specific_humidity"].units + " / s"},
                    ),
                }
            )
            return match_prediction_to_input_coords(X, ds.unstack(SAMPLE_DIM_NAME))

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self.model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            with open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
                f.write(
                    yaml.dump(
                        {
                            "sample_dim_name": self.sample_dim_name,
                            "input_variables": self.input_variables,
                            "output_variables": self.output_variables,
                            "output_metadata": self._output_metadata,
                        }
                    )
                )
