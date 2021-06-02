from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Union
from ._shared import Predictor, io
import numpy as np
import xarray as xr
import os
import yaml


@io.register("constant-output")
class ConstantOutputPredictor(Predictor):
    """
    A simple predictor meant to be used for testing.
    
    Supports scalar and vector outputs, where the vector outputs are all
    of the same shape and assigned a dimension name of "z".
    """

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
    ):
        """Initialize the predictor
        
        Args:
            sample_dim_name: name of sample dimension
            input_variables: names of input variables
            output_variables: names of output variables
        
        """
        super().__init__(
            sample_dim_name=sample_dim_name,
            input_variables=input_variables,
            output_variables=output_variables,
        )
        self._outputs: Dict[Hashable, Union[np.ndarray, float]] = {}

    def set_outputs(self, **outputs: Union[np.ndarray, float]):
        """
        Args:
            outputs: column output for each name. For scalar
                values, use float, and for column values use 1D arrays.
        """
        self._outputs.update(outputs)  # type: ignore

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        n_samples = len(X[self.sample_dim_name])
        data_vars = {}
        for name in self.output_variables:
            output = self._outputs[name]
            if isinstance(output, np.ndarray):
                array = np.repeat(output[None, :], repeats=n_samples, axis=0)
                data_vars[name] = xr.DataArray(
                    data=array, dims=[self.sample_dim_name, "z"]
                )
            else:
                array = np.full([n_samples], float(output))
                data_vars[name] = xr.DataArray(data=array, dims=[self.sample_dim_name])
        if self.sample_dim_name in X:
            coords: Optional[Mapping[Hashable, Any]] = {
                self.sample_dim_name: X[self.sample_dim_name]
            }
        else:
            coords = None
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def dump(self, path: str) -> None:
        np.savez(os.path.join(path, "_outputs.npz"), **self._outputs)
        with open(os.path.join(path, "attrs.yaml"), "w") as f:
            yaml.safe_dump(
                {
                    "sample_dim_name": self.sample_dim_name,
                    "input_variables": self.input_variables,
                    "output_variables": self.output_variables,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> object:
        """Load a serialized model from a directory."""
        outputs = np.load(os.path.join(path, "_outputs.npz"))
        with open(os.path.join(path, "attrs.yaml"), "r") as f:
            attrs = yaml.safe_load(f)
        obj = cls(**attrs)
        obj.set_outputs(**outputs)
        return obj
