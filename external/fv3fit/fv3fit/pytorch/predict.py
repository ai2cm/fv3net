from fv3fit._shared.predictor import Reloadable, Predictor
from .._shared.scaler import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from typing import (
    Any,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Type,
    IO,
    Protocol,
)
import zipfile
from fv3fit.pytorch.system import DEVICE
import os
import yaml
import vcm
from fv3fit._shared import io


L = TypeVar("L", bound="BinaryLoadable")


class BinaryLoadable(Protocol):
    """
    Abstract base class for objects that can be dumped.
    """

    @classmethod
    def load(cls: Type[L], f: IO[bytes]) -> L:
        ...


def dump_mapping(mapping: Mapping[str, StandardScaler], f: IO[bytes]) -> None:
    """
    Serialize a mapping to a zip file.
    """
    with zipfile.ZipFile(f, "w") as archive:
        for key, value in mapping.items():
            with archive.open(str(key), "w") as f_dump:
                value.dump(f_dump)


def load_mapping(cls: Type[L], f: IO[bytes]) -> Mapping[str, L]:
    """
    Load a mapping from a zip file.
    """
    with zipfile.ZipFile(f, "r") as archive:
        return {name: cls.load(archive.open(name, "r")) for name in archive.namelist()}


@io.register("pytorch_predictor")
class PytorchPredictor(Predictor):

    _MODEL_FILENAME = "weight.pt"
    _CONFIG_FILENAME = "config.yaml"
    _SCALERS_FILENAME = "scalers.zip"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        model: nn.Module,
        scalers: Mapping[str, StandardScaler],
    ):
        """Initialize the predictor
        Args:
            state_variables: names of state variables
            model: pytorch model to wrap
            scalers: normalization data for each of the state variables
        """
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.model = model
        self.scalers = scalers

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """
        Predict an output xarray dataset from an input xarray dataset.
        Note that returned datasets include the initial state of the prediction,
        where by definition the model will have perfect skill.

        Args:
            X: input dataset
            timesteps: number of timesteps to predict

        Returns:
            predicted: predicted timeseries data
            reference: true timeseries data from the input dataset
        """
        tensor = self.pack_to_tensor(X)
        with torch.no_grad():
            outputs = self.model(tensor)
        predicted = self.unpack_tensor(outputs)
        return predicted

    def pack_to_tensor(self, X: xr.Dataset) -> torch.Tensor:
        packed = _pack_to_tensor(
            ds=X,
            timesteps=0,
            state_variables=tuple(str(item) for item in self.input_variables),
            scalers=self.scalers,
        )
        # dimensions are [time, tile, x, y, z],
        # we must combine [time, tile] into one sample dimension
        return torch.reshape(
            packed, (packed.shape[0] * packed.shape[1],) + tuple(packed.shape[2:]),
        )

    def unpack_tensor(self, data: torch.Tensor) -> xr.Dataset:
        data = torch.reshape(data, (-1, 6) + tuple(data.shape[1:]))
        return _unpack_tensor(
            data,
            varnames=tuple(str(item) for item in self.output_variables),
            scalers=self.scalers,
            dims=["time", "tile", "x", "y", "z"],
        )

    @classmethod
    def load(cls, path: str) -> "PytorchPredictor":
        """Load a serialized model from a directory."""
        return _load_pytorch(cls, path)

    def dump(self, path: str) -> None:
        _dump_pytorch(self, path)

    def get_config(self):
        return {
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
        }


@io.register("pytorch_autoregressor")
class PytorchAutoregressor(Reloadable):

    _MODEL_FILENAME = "weight.pt"
    _CONFIG_FILENAME = "config.yaml"
    _SCALERS_FILENAME = "scalers.zip"

    def __init__(
        self,
        state_variables: Iterable[str],
        model: nn.Module,
        scalers: Mapping[str, StandardScaler],
    ):
        """Initialize the predictor
        Args:
            state_variables: names of state variables
            model: pytorch model to wrap
            scalers: normalization data for each of the state variables
        """
        self.state_variables = state_variables
        self.model = model
        self.scalers = scalers

    def pack_to_tensor(self, ds: xr.Dataset, timesteps: int) -> torch.Tensor:
        """
        Packs the dataset into a tensor to be used by the pytorch model.

        Subdivides the dataset evenly into windows
        of size (timesteps + 1) with overlapping start and end points.
        Overlapping the window start and ends is necessary so that every
        timestep (evolution from one time to the next) is included within
        one of the windows.

        Args:
            ds: dataset containing values to pack
            timesteps: number timesteps to include in each window after initial time

        Returns:
            tensor of shape [window, time, tile, x, y, feature]
        """
        return _pack_to_tensor(
            ds=ds,
            timesteps=timesteps,
            state_variables=self.state_variables,
            scalers=self.scalers,
        )

    def unpack_tensor(self, data: torch.Tensor) -> xr.Dataset:
        """
        Unpacks the tensor into a dataset.

        Args:
            data: tensor of shape [window, time, tile, x, y, feature]

        Returns:
            xarray dataset with values of shape [window, time, tile, x, y, feature]
        """
        return _unpack_tensor(
            data,
            varnames=self.state_variables,
            scalers=self.scalers,
            dims=["window", "time", "tile", "x", "y", "z"],
        )

    def step_model(self, state: torch.Tensor, timesteps: int):
        """
        Step the model forward.
        Args:
            state: tensor of shape [sample, tile, x, y, feature]
            timesteps: number of timesteps to predict
        Returns:
            tensor of shape [sample, time, tile, x, y, feature], with time dimension
                having length timesteps + 1 and including the initial state
        """
        outputs = torch.zeros(
            [state.shape[0]] + [timesteps + 1] + list(state.shape[1:])
        )
        outputs[:, 0, :] = state
        for i in range(timesteps):
            with torch.no_grad():
                outputs[:, i + 1, :] = self.model(outputs[:, i, :])
        return outputs

    def predict(self, X: xr.Dataset, timesteps: int) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Predict an output xarray dataset from an input xarray dataset.

        Note that returned datasets include the initial state of the prediction,
        where by definition the model will have perfect skill.

        Args:
            X: input dataset
            timesteps: number of timesteps to predict

        Returns:
            predicted: predicted timeseries data
            reference: true timeseries data from the input dataset
        """
        tensor = self.pack_to_tensor(X, timesteps=timesteps)
        outputs = self.step_model(tensor[:, 0, :], timesteps=timesteps)
        predicted = self.unpack_tensor(outputs)
        reference = self.unpack_tensor(tensor)
        return predicted, reference

    @classmethod
    def load(cls, path: str) -> "PytorchAutoregressor":
        """Load a serialized model from a directory."""
        return _load_pytorch(cls, path)

    def dump(self, path: str) -> None:
        _dump_pytorch(self, path)

    def get_config(self) -> Mapping[str, Any]:
        return {"state_variables": self.state_variables}


class _PytorchDumpable(Protocol):
    _MODEL_FILENAME: str
    _SCALERS_FILENAME: str
    _CONFIG_FILENAME: str
    scalers: Mapping[str, StandardScaler]
    model: torch.nn.Module

    def __init__(
        self, model: torch.nn.Module, scalers: Mapping[str, StandardScaler], **kwargs,
    ):
        ...

    def dump(self, path: str) -> None:
        ...

    def get_config(self) -> Mapping[str, Any]:
        """
        Returns additional keyword arguments needed to initialize this object.
        """
        ...


def _load_pytorch(cls: Type[_PytorchDumpable], path: str):
    """Load a serialized model from a directory."""
    fs = vcm.get_fs(path)
    model_filename = os.path.join(path, cls._MODEL_FILENAME)
    with fs.open(model_filename, "rb") as f:
        model = torch.load(f)
    with fs.open(os.path.join(path, cls._SCALERS_FILENAME), "rb") as f:
        scalers = load_mapping(StandardScaler, f)
    with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    obj = cls(model=model, scalers=scalers, **config)
    return obj


def _dump_pytorch(obj: _PytorchDumpable, path: str) -> None:
    fs = vcm.get_fs(path)
    model_filename = os.path.join(path, obj._MODEL_FILENAME)
    with fs.open(model_filename, "wb") as f:
        torch.save(obj.model, model_filename)
    with fs.open(os.path.join(path, obj._SCALERS_FILENAME), "wb") as f:
        dump_mapping(obj.scalers, f)
    with fs.open(os.path.join(path, obj._CONFIG_FILENAME), "w") as f:
        f.write(yaml.dump(obj.get_config()))


def _pack_to_tensor(
    ds: xr.Dataset,
    timesteps: int,
    state_variables: Iterable[str],
    scalers: Mapping[str, StandardScaler],
) -> torch.Tensor:
    """
    Packs the dataset into a tensor to be used by the pytorch model.

    Subdivides the dataset evenly into windows
    of size (timesteps + 1) with overlapping start and end points.
    Overlapping the window start and ends is necessary so that every
    timestep (evolution from one time to the next) is included within
    one of the windows.

    Args:
        ds: dataset containing values to pack
        timesteps: number timesteps to include in each window after initial time

    Returns:
        tensor of shape [window, time, tile, x, y, feature]
    """

    expected_dims = ("time", "tile", "x", "y", "z")
    ds = ds.transpose(*expected_dims)
    if timesteps > 0:
        n_times = ds.time.size
        n_windows = int((n_times - 1) // timesteps)
        # times need to be evenly divisible into windows
        ds = ds.isel(time=slice(None, n_windows * timesteps + 1))
    all_data = []
    for varname in state_variables:
        var_dims = ds[varname].dims
        if tuple(var_dims[:4]) != expected_dims[:4]:
            raise ValueError(
                f"received variable {varname} with " f"unexpected dimensions {var_dims}"
            )
        data = ds[varname].values
        normalized_data = scalers[varname].normalize(data)
        if timesteps > 0:
            # segment time axis into windows, excluding last time of each window
            data = normalized_data[:-1, :].reshape(
                n_windows, timesteps, *data.shape[1:]
            )
            # append first time of next window to end of each window
            end_data = np.concatenate(
                [data[1:, :1, :], normalized_data[None, -1:, :]], axis=0
            )
            data = np.concatenate([data, end_data], axis=1)
        else:
            data = normalized_data
        if "z" not in var_dims:
            # need a z-axis for concatenation into feature axis
            data = data[..., np.newaxis]
        all_data.append(data)
    concatenated_data = np.concatenate(all_data, axis=-1)
    return torch.as_tensor(concatenated_data).float().to(DEVICE)


def _unpack_tensor(
    data: torch.Tensor,
    varnames: Iterable[str],
    scalers: Mapping[str, StandardScaler],
    dims: Sequence[Hashable],
) -> xr.Dataset:
    i_feature = 0
    data_vars = {}
    for varname in varnames:
        mean_value = scalers[varname].mean
        if mean_value is None:
            raise RuntimeError(f"scaler for {varname} has not been fit")
        else:
            if len(mean_value.shape) > 0 and mean_value.shape[0] > 1:
                n_features = mean_value.shape[0]
                var_data = data[..., i_feature : i_feature + n_features]
            else:
                n_features = 1
                var_data = data[..., i_feature]
            var_data = scalers[varname].denormalize(var_data.to("cpu").numpy())
            data_vars[varname] = xr.DataArray(
                data=var_data, dims=dims[: len(var_data.shape)]
            )
            i_feature += n_features
    return xr.Dataset(data_vars=data_vars)
