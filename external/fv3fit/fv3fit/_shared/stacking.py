from typing import Sequence, Tuple
import xarray as xr

from vcm import safe


SAMPLE_DIM_NAME = "_fv3fit_sample"
DATASET_DIM_NAME = "dataset"
Z_DIM_NAMES = ["z", "pfull"]


def stack(ds: xr.Dataset, unstacked_dims: Sequence[str]):
    stack_dims = [dim for dim in ds.dims if dim not in unstacked_dims]
    unstacked_dims = [dim for dim in ds.dims if dim in unstacked_dims]
    unstacked_dims.sort()  # needed to always get [x, y, z] dimensions
    ds_stacked = safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=list(unstacked_dims) + ["time", "dataset"],
    )
    return ds_stacked.transpose(SAMPLE_DIM_NAME, *unstacked_dims)


def stack_non_vertical(ds: xr.Dataset) -> xr.Dataset:
    """
    Stack all dimensions except for the Z dimensions into a sample

    Args:
        ds: dataset with geospatial dimensions
    """
    if len(set(ds.dims).intersection(Z_DIM_NAMES)) > 1:
        raise ValueError("Data cannot have >1 feature dimension in {Z_DIM_NAMES}.")
    return stack(ds=ds, unstacked_dims=Z_DIM_NAMES)


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
