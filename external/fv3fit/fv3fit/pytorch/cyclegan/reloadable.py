from fv3fit._shared import io
from fv3fit._shared.predictor import Reloadable
from fv3fit._shared.scaler import StandardScaler
from fv3fit.pytorch.predict import (
    _dump_pytorch,
    _load_pytorch,
    _pack_to_tensor,
    _unpack_tensor,
)
from .generator import Generator
from .discriminator import Discriminator
from typing import Mapping, Iterable
import torch
import xarray as xr


class CycleGANModule(torch.nn.Module):
    """
    Torch module containing the components of a CycleGAN.

    All modules expect inputs and produce outputs of shape
    (batch, tile, channels, x, y).
    """

    # we package this in this way so we can easily transform the model
    # to different devices, and save/load the model as one module
    def __init__(
        self,
        generator_a_to_b: Generator,
        generator_b_to_a: Generator,
        discriminator_a: Discriminator,
        discriminator_b: Discriminator,
    ):
        super(CycleGANModule, self).__init__()
        self.generator_a_to_b = generator_a_to_b
        self.generator_b_to_a = generator_b_to_a
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b


@io.register("cycle_gan")
class CycleGAN(Reloadable):

    _MODEL_FILENAME = "weight.pt"
    _CONFIG_FILENAME = "config.yaml"
    _SCALERS_FILENAME = "scalers.zip"

    def __init__(
        self,
        model: CycleGANModule,
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
    def generator_a_to_b(self) -> torch.nn.Module:
        return self.model.generator_a_to_b

    @property
    def generator_b_to_a(self) -> torch.nn.Module:
        return self.model.generator_b_to_a

    @property
    def discriminator_a(self) -> torch.nn.Module:
        return self.model.discriminator_a

    @property
    def discriminator_b(self) -> torch.nn.Module:
        return self.model.discriminator_b

    @classmethod
    def load(cls, path: str) -> "CycleGAN":
        """Load a serialized model from a directory."""
        return _load_pytorch(cls, path)

    def to(self, device) -> "CycleGAN":
        model = self.model.to(device)
        return CycleGAN(model, self.scalers, **self.get_config())

    def dump(self, path: str) -> None:
        _dump_pytorch(self, path)

    def get_config(self):
        return {"state_variables": self.state_variables}

    def pack_to_tensor(self, ds: xr.Dataset, domain: str = "a") -> torch.Tensor:
        """
        Packs the dataset into a tensor to be used by the pytorch model.

        Subdivides the dataset evenly into windows
        of size (timesteps + 1) with overlapping start and end points.
        Overlapping the window start and ends is necessary so that every
        timestep (evolution from one time to the next) is included within
        one of the windows.

        Args:
            ds: dataset containing values to pack
            domain: one of "a" or "b"

        Returns:
            tensor of shape [window, time, tile, x, y, feature]
        """
        scalers = {
            name[2:]: scaler
            for name, scaler in self.scalers.items()
            if name.startswith(f"{domain}_")
        }
        tensor = _pack_to_tensor(
            ds=ds, timesteps=0, state_variables=self.state_variables, scalers=scalers,
        )
        return tensor.permute([0, 1, 4, 2, 3])

    def unpack_tensor(self, data: torch.Tensor, domain: str = "b") -> xr.Dataset:
        """
        Unpacks the tensor into a dataset.

        Args:
            data: tensor of shape [window, time, tile, x, y, feature]
            domain: one of "a" or "b"

        Returns:
            xarray dataset with values of shape [window, time, tile, x, y, feature]
        """
        scalers = {
            name[2:]: scaler
            for name, scaler in self.scalers.items()
            if name.startswith(f"{domain}_")
        }
        return _unpack_tensor(
            data.permute([0, 1, 3, 4, 2]),
            varnames=self.state_variables,
            scalers=scalers,
            dims=["time", "tile", "x", "y", "z"],
        )

    def predict(self, X: xr.Dataset, reverse: bool = False) -> xr.Dataset:
        """
        Predict a state in the output domain from a state in the input domain.

        Args:
            X: input dataset
            reverse: if True, transform from the output domain to the input domain

        Returns:
            predicted: predicted dataset
        """
        if reverse:
            input_domain, output_domain = "b", "a"
        else:
            input_domain, output_domain = "a", "b"

        tensor = self.pack_to_tensor(X, domain=input_domain)
        if reverse:
            generator = self.generator_b_to_a
        else:
            generator = self.generator_a_to_b
        n_batch = 100
        outputs = torch.zeros_like(tensor)
        with torch.no_grad():
            for i in range(0, tensor.shape[0], n_batch):
                new: torch.Tensor = generator(tensor[i : i + n_batch])
                outputs[i : i + new.shape[0]] = new
        predicted = self.unpack_tensor(outputs, domain=output_domain)
        return predicted
