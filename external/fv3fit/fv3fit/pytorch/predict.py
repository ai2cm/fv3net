import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from fv3fit._shared import (
    Predictor,
    match_prediction_to_input_coords,
    SAMPLE_DIM_NAME,
    stack,
)
from typing import Any, Dict, Hashable, Iterable, Sequence


class PytorchModel(Predictor):

    _MODEL_FILENAME = "weight.pt"
    _CONFIG_FILENAME = "config.yaml"
    custom_objects: Dict[str, Any] = {}

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        model: nn.Module,
        normalizers,
        unstacked_dims: Sequence[str],
    ):
        """Initialize the predictor
        Args:
            input_variables: names of input variables
            output_variables: names of output variables
            model: pytorch model to wrap
            unstacked_dims: non-sample dimensions of model output
        """
        super().__init__(input_variables, output_variables)
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.model = model
        self.normalizers = normalizers
        self._unstacked_dims = unstacked_dims

    def pack_to_tensor(self, ds: xr.Dataset) -> torch.Tensor:
        """
        Args:
            ds

        Returns:
            tensor of shape [sample, time, tile, x, y, feature]
        """
        pass

    def unpack_tensor(self, data: torch.Tensor) -> xr.Dataset:
        pass

    def _array_prediction_to_dataset(
        self, names, outputs, stacked_coords
    ) -> xr.Dataset:
        ds = xr.Dataset()
        for name, output in zip(names, outputs):
            dims = [SAMPLE_DIM_NAME] + list(self._unstacked_dims)
            scalar_singleton_dim = (
                len(output.shape) == len(dims) and output.shape[-1] == 1
            )
            if scalar_singleton_dim:  # remove singleton dimension
                output = output[..., 0]
                dims = dims[:-1]
            da = xr.DataArray(
                data=output, dims=dims, coords={SAMPLE_DIM_NAME: stacked_coords},
            ).unstack(SAMPLE_DIM_NAME)
            dim_order = [dim for dim in self._unstacked_dims if dim in da.dims]
            ds[name] = da.transpose(*dim_order, ...)
        return ds

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        X = X.transpose("grid", "z")
        X_stacked = stack(X, unstacked_dims=self._unstacked_dims)
        inputs = [X_stacked[name].values for name in self.input_variables]
        outputs = self.model(torch.as_tensor(inputs).float())
        if isinstance(outputs, np.ndarray):
            outputs = [outputs]
        return_ds = self._array_prediction_to_dataset(
            self.output_variables,
            outputs.detach().cpu().numpy(),
            X_stacked.coords[SAMPLE_DIM_NAME],
        )
        # turn outputs into an xarray dataset
        return match_prediction_to_input_coords(X, return_ds)

    @classmethod
    def load(cls, path: str) -> "PytorchModel":
        raise NotImplementedError()
        # """Load a serialized model from a directory."""
        # with get_dir(path) as path:
        #     model_filename = os.path.join(path, cls._MODEL_FILENAME)
        #     model = self.model.load_state_dict(torch.load(model_filename))
        #     self.model.eval()

        #     with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
        #         config = yaml.load(f, Loader=yaml.Loader)
        #     obj = cls(
        #         config["input_variables"],
        #         config["output_variables"],
        #         model,
        #         unstacked_dims=config.get("unstacked_dims", None),
        #     )
        #     return obj

    def dump(self, path: str) -> None:
        raise NotImplementedError()
        # with put_dir(path) as path:
        #     if self.model is not None:
        #         model_filename = os.path.join(path, self._MODEL_FILENAME)
        #         torch.save(self.model.state_dict(), model_filename)
        #     with open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
        #         f.write(
        #             yaml.dump(
        #                 {
        #                     "input_variables": self.input_variables,
        #                     "output_variables": self.output_variables,
        #                     "unstacked_dims": self._unstacked_dims,
        #                 }
        #             )
        #         )
