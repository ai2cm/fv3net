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
        equal_nan: Whether to compare NaN’s as equal. If True, NaN's in a will
            be considered equal to NaN's in b in the output array (bool).

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


def _repeat_dataarray(obj: xr.DataArray, repeats: int, dim: Hashable) -> xr.DataArray:
    """Repeat elements of an array."""

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
            kwargs={"repeats": repeats},
        )


def repeat(
    obj: Union[xr.Dataset, xr.DataArray], repeats: Union[int, np.array], dim: Hashable
) -> Union[xr.Dataset, xr.DataArray]:
    """Repeat elements of an array.

    Args:
        obj: Input xr.Dataset or xr.DataArray.
        repeats: The number of repetitions for each element. repeats is
            broadcasted to fit the shape of the given axis.  Note that an array
            repeats argument is not supported for dask arrays.
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
        return obj.apply(_repeat_dataarray, args=(repeats, dim))
    else:
        return _repeat_dataarray(obj, repeats, dim)
    return obj


def assert_identical_including_dtype(
    a: Union[xr.Dataset, xr.DataArray], b: Union[xr.Dataset, xr.DataArray]
):
    """Check whether two xarray objects are identical up to the dtype.

    xarray's internal xr.testing.assert_identical does not check whether variable
    dtypes are equal.

    Args:
        a: Input xr.Dataset or xr.DataArray
        b: Input xr.Dataset or xr.DataArray
    """
    xr.testing.assert_identical(a, b)
    if isinstance(a, xr.Dataset):
        for variable in a.variables:
            assert a[variable].dtype == b[variable].dtype
    else:
        assert a.dtype == b.dtype
        for coord in a.coords:
            assert a[coord].dtype == b[coord].dtype
