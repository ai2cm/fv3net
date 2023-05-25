import numpy as np
from typing import Sequence, Iterable, Hashable, Protocol
import xarray as xr

from fv3fit._shared.halos import append_halos, append_halos_using_mpi
from fv3fit._shared import (
    match_prediction_to_input_coords,
    SAMPLE_DIM_NAME,
    stack,
)


class ArrayPredictor(Protocol):
    def predict(self, inputs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        ...


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
