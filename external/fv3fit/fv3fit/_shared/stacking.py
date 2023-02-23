from typing import Sequence, Tuple, Optional
import xarray as xr

from vcm import safe


SAMPLE_DIM_NAME = "_fv3fit_sample"
DATASET_DIM_NAME = "dataset"
Z_DIM_NAMES = ["z", "pfull"]


def stack(ds: xr.Dataset, unstacked_dims: Optional[Sequence[str]] = None):
    if unstacked_dims is None:
        unstacked_dims = []
    stack_dims = [dim for dim in ds.dims if dim not in unstacked_dims]
    unstacked_dims = [dim for dim in ds.dims if dim in unstacked_dims]
    unstacked_dims.sort()  # needed to always get [x, y, z] dimensions
    if len(stack_dims) == 0:
        ds_stacked = ds.expand_dims(dim=SAMPLE_DIM_NAME, axis=0)
    else:
        ds_stacked = safe.stack_once(
            ds,
            SAMPLE_DIM_NAME,
            stack_dims,
            allowed_broadcast_dims=list(unstacked_dims) + ["time", "dataset"],
        )
    return ds_stacked.transpose(SAMPLE_DIM_NAME, *unstacked_dims)


def _infer_dimension_order(ds: xr.Dataset) -> Tuple:
    # add check here for cases when the dimension order is inconsistent between arrays?
    dim_order = []
    for variable in ds:
        for dim in ds[variable].dims:
            if dim not in dim_order:
                dim_order.append(dim)
    return tuple(dim_order)


def match_prediction_to_input_coords(
    input: xr.Dataset, prediction: xr.Dataset
) -> xr.Dataset:
    # ensure the output coords are the same and dims are same order
    # stack/unstack adds coordinates if none exist before
    input_coords = input.coords
    for key in prediction.coords:
        if key in input_coords:
            prediction.coords[key] = input_coords[key]
        else:
            del prediction.coords[key]
    dim_order = [dim for dim in _infer_dimension_order(input) if dim in prediction.dims]
    return prediction.transpose(*dim_order)
