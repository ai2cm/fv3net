from typing import cast, Sequence, Hashable
import xarray as xr


def get_variables(ds: xr.Dataset, variables: Sequence[Hashable]) -> xr.Dataset:
    """ds[...] is very confusing function from a typing perspective and should be
    avoided in long-running pipeline codes. This function introduces a type-stable
    alternative that works better with mypy.

    In particular, ds[('a' , 'b' ,'c')] looks for a variable named ('a', 'b', 'c') which
    usually doesn't exist, so it causes a key error. but ds[['a', 'b', 'c']] makes a
    dataset only consisting of the variables 'a', 'b', and 'c'. This causes tons of
    hard to find errors.
    """
    variables = list(variables)
    return cast(xr.Dataset, ds[variables])


def _validate_stack_dims(ds, dims, allowed_broadcast_dims=()):
    """Don't broadcast arrays"""
    for variable in ds:
        var_dims = ds[variable].dims
        broadcast_dims = set(dims) - (set(var_dims) | set(allowed_broadcast_dims))
        if len(broadcast_dims) > 0:
            raise ValueError(
                f"{variable} will be broadcast to include unallowed dimensions {broadcast_dims}. "
                "This could greatly increase the size of dataset."
            )


def stack_once(
    ds: xr.Dataset,
    dim,
    dims: Sequence[Hashable],
    allowed_broadcast_dims: Sequence[Hashable] = (),
):
    """Stack once raising ValueError if any unexpected broadcasting occurs"""
    _validate_stack_dims(ds, dims, allowed_broadcast_dims)
    return ds.stack({dim: dims})
