from fv3fit._shared.scaler import StandardScaler
from fv3fit.pytorch.predict import (
    _dump_pytorch,
    _load_pytorch,
    _pack_to_tensor,
    _unpack_tensor,
)
from fv3fit.pytorch.cyclegan.modules import FoldFirstDimension
from torch import nn
import xarray as xr
from typing import Iterable, Mapping, Tuple
from fv3fit._shared.predictor import Reloadable
import torch
from fv3fit._shared import io


class FMRModule(nn.Module):
    """Module to pack generator and discriminator into a single module for saving."""

    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator


@io.register("fmr")
class FullModelReplacement(Reloadable):

    _MODEL_FILENAME = "weight.pt"
    _CONFIG_FILENAME = "config.yaml"
    _SCALERS_FILENAME = "scalers.zip"

    def __init__(
        self,
        model: FMRModule,
        scalers: Mapping[str, StandardScaler],
        state_variables: Iterable[str],
    ):
        """
            Args:
                model: pytorch model
                scalers: scalers for the state variables, keys are prepended with "a_"
                    or "b_" to denote the domain of the scaler, followed by the name of
                    the state variable it scales
                state_variables: name of variables to be used as state variables in
                    the order expected by the model
        """
        self.model = model
        self.scalers = scalers
        self.state_variables = state_variables

    @property
    def generator(self) -> nn.Module:
        return self.model.generator

    @property
    def discriminator(self) -> nn.Module:
        # the trained discriminator operates on single timesteps, but when testing
        # we want to evaluate the entire sequence at once, so we fold the batch
        # and time dimensions together
        return FoldFirstDimension(self.model.discriminator)

    @classmethod
    def load(cls, path: str) -> "FullModelReplacement":
        """Load a serialized model from a directory."""
        return _load_pytorch(cls, path)

    def to(self, device) -> "FullModelReplacement":
        model = self.model.to(device)
        return FullModelReplacement(model, scalers=self.scalers, **self.get_config())

    def dump(self, path: str) -> None:
        _dump_pytorch(self, path)

    def get_config(self):
        return {"state_variables": self.state_variables}

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
        tensor = _pack_to_tensor(
            ds=ds,
            timesteps=timesteps,
            state_variables=self.state_variables,
            scalers=self.scalers,
        )
        return tensor.permute([0, 1, 2, 5, 3, 4])

    def unpack_tensor(self, data: torch.Tensor) -> xr.Dataset:
        """
        Unpacks the tensor into a dataset.

        Args:
            data: tensor of shape [window, time, tile, x, y, feature]

        Returns:
            xarray dataset with values of shape [window, time, tile, x, y, feature]
        """
        return _unpack_tensor(
            data.permute([0, 1, 2, 4, 5, 3]),
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
        with torch.no_grad():
            output = self.generator(state, ntime=timesteps + 1)
        return output

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
