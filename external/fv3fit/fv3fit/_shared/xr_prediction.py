import abc
import numpy as np
import os
from typing import Sequence, Iterable, Hashable
import xarray as xr
import yaml

from fv3fit._shared.halos import append_halos, append_halos_using_mpi
from fv3fit._shared import (
    match_prediction_to_input_coords,
    SAMPLE_DIM_NAME,
    stack,
    io,
    Predictor,
    get_dir,
    put_dir,
)


class ArrayPredictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, inputs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        pass

    @abc.abstractmethod
    def dump(self, path: str) -> None:
        """Serialize to a directory."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "ArrayPredictor":
        """Load a serialized model from a directory."""
        pass


def _array_prediction_to_dataset(
    names, outputs, stacked_coords, unstacked_dims,
) -> xr.Dataset:
    ds = xr.Dataset()
    for name, output in zip(names, outputs):
        if len(unstacked_dims) > 0:
            dims = [SAMPLE_DIM_NAME] + list(unstacked_dims)
            scalar_singleton_dim = (
                len(output.shape) == len(dims) and output.shape[-1] == 1
            )
            if scalar_singleton_dim:  # remove singleton dimension
                output = output[..., 0]
                dims = dims[:-1]
        else:
            dims = [SAMPLE_DIM_NAME]
            output = output[..., 0]
        da = xr.DataArray(
            data=output, dims=dims, coords={SAMPLE_DIM_NAME: stacked_coords},
        ).unstack(SAMPLE_DIM_NAME)
        dim_order = [dim for dim in unstacked_dims if dim in da.dims]
        ds[name] = da.transpose(*dim_order, ...)
    return ds


def predict_on_dataset(
    model: ArrayPredictor,
    X: xr.Dataset,
    input_variables: Iterable[Hashable],
    output_variables: Iterable[Hashable],
    n_halo: int,
    unstacked_dims: Sequence[str],
) -> xr.Dataset:
    """Predict an output xarray dataset from an input xarray dataset."""
    if n_halo > 0:
        if "tile" not in X.dims:
            try:
                X = append_halos_using_mpi(ds=X, n_halo=n_halo)
            except RuntimeError as err:
                raise ValueError(
                    "either dataset must have tile dimension or MPI must be present"
                ) from err
        else:
            X = append_halos(ds=X, n_halo=n_halo)
    X_stacked = stack(X, unstacked_dims=unstacked_dims)
    inputs = [X_stacked[name].values for name in input_variables]
    outputs = model.predict(inputs)
    if isinstance(outputs, np.ndarray):
        outputs = [outputs]
    return_ds = _array_prediction_to_dataset(
        names=output_variables,
        outputs=outputs,
        stacked_coords=X_stacked.coords[SAMPLE_DIM_NAME],
        unstacked_dims=unstacked_dims,
    )
    return match_prediction_to_input_coords(X, return_ds)


@io.register("dataset-predictor")
class DatasetPredictor(Predictor):
    """
    Base model is an ArrayPredictor which takes in a sequence of
    input arrays and outputs a sequence of output arrays.
    Normalization is contained within the base ArrayPredictor.

    Assumes wrapped model accepts [sample, feature] arrays.
    """

    _BASE_MODEL_NAME = "base_model"
    _CONFIG_FILENAME = "config.yaml"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        model: ArrayPredictor,
        unstacked_dims: Sequence[str],
        n_halo: int = 0,
    ):
        """Initialize the predictor

        Args:
            input_variables: names of input variables
            output_variables: names of output variables
            model: base model to wrap. Must be in fv3fit.io registry
            unstacked_dims: non-sample dimensions of model output
            n_halo: number of halo points required in input data
        """
        super().__init__(input_variables, output_variables)
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.model = model
        self._n_halo = n_halo
        self._unstacked_dims = unstacked_dims

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        return predict_on_dataset(
            model=self.model,
            X=X,
            input_variables=self.input_variables,
            output_variables=self.output_variables,
            n_halo=self._n_halo,
            unstacked_dims=self._unstacked_dims,
        )

    @classmethod
    def load(cls, path: str) -> "DatasetPredictor":
        """Load a serialized model from a directory."""
        with get_dir(path) as path:
            base_model_filename = os.path.join(path, cls._BASE_MODEL_NAME)
            base_model: ArrayPredictor = io.load(base_model_filename)  # type: ignore
            with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            obj = cls(
                input_variables=config["input_variables"],
                output_variables=config["output_variables"],
                model=base_model,
                unstacked_dims=config.get("unstacked_dims", None),
                n_halo=config.get("n_halo", 0),
            )

            return obj

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self.model is not None:
                io.dump(
                    self.model,  # type: ignore
                    os.path.join(path, self._BASE_MODEL_NAME),
                )
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
