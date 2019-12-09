from typing import Hashable, Union

import numpy as np
import xarray as xr


def isclose(
    a: xr.DataArray,
    b: xr.DataArray,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> xr.DataArray:
    """An xarray wrapper of np.isclose.

    Args:
        a: xr.DataArray
        b: xr.DataArray
        rtol: The relative tolerance parameter (float).
        atol: The absolute tolerance parameter (float).
        equal_nan: Whether to compare NaN’s as equal. If True, NaN’s in a will
            be considered equal to NaN’s in b in the output array (bool).

    Returns:
        xr.DataArray
    """
    return xr.apply_ufunc(
        np.isclose,
        a,
        b,
        dask="parallelized",
        output_dtypes=[np.bool],
        kwargs={"rtol": rtol, "atol": atol, "equal_nan": equal_nan},
    )


def _repeat_dataarray(obj: xr.DataArray, factor: int, dim: Hashable) -> xr.DataArray:
    """Upsample a DataArray by uniformly repeating values n times along a dimension."""

    def func(arr, repeats=1):
        return arr.repeat(repeats, axis=-1)

    if dim not in obj.dims:
        return obj
    else:
        return xr.apply_ufunc(
            func,
            obj,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            exclude_dims={dim},
            dask="allowed",
            kwargs={"repeats": factor},
        )


def repeat(
    obj: Union[xr.Dataset, xr.DataArray], factor: int, dim: Hashable
) -> Union[xr.Dataset, xr.DataArray]:
    """Upsample an object by uniformly repeating values n times along a
    dimension.
    
    Args:
        obj: Input xr.Dataset or xr.DataArray.
        factor: Integer repetition factor.
        dim: Dimension name to repeat along.

    Returns:
        xr.Dataset or xr.DataArray
    """
    obj_dimensions = list(obj.dims)
    if dim not in obj.dims:
        raise ValueError(
            "Cannot repeat over {!r}; expected one of {!r}.".format(dim, obj_dimensions)
        )

    if isinstance(obj, xr.Dataset):
        return obj.apply(_repeat_dataarray, args=(factor, dim))
    else:
        return _repeat_dataarray(obj, factor, dim)
    return obj
