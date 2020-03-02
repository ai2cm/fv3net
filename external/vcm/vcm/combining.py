from typing import Any, Iterable, Sequence, Tuple, Mapping
from collections import defaultdict
import pandas as pd

import xarray as xr


def _collect_variables(datasets):
    variables = defaultdict(dict)
    for name, dims, array in datasets:
        variables[name][dims] = array
    return variables



def _datasets_dims(ds):
    return ds.dims

def _combine_arrays(arrays: Mapping[Tuple, xr.Dataset], labels):
    idx = pd.MultiIndex.from_tuples(arrays.keys(), names=labels)
    idx.name = 'concat_dim'
    concat = xr.concat(arrays.values(), dim=idx)
    old_dims = concat.isel({idx.name: 0}).dims
    new_dims = tuple(labels) + tuple(old_dims)
    return concat.unstack(idx.name).transpose(*new_dims)


def _merge_datasets_with_key(datasets):
    output = defaultdict(list)
    for dims, datasets in datasets:
        output[dims].append(datasets)
    
    for key in output:
        output[key] = xr.merge(output[key])
    return output


def combine_dataset_sequence(
    datasets: Iterable[Tuple[Tuple, xr.Dataset]],
    labels: Sequence[Any]
) -> xr.Dataset:
    datasets = _merge_datasets_with_key(datasets)
    return _combine_arrays(datasets, labels)


def combine_array_sequence(
    datasets: Iterable[Tuple[Any, Tuple, xr.DataArray]], labels: Sequence[Any]
) -> xr.Dataset:
    """Combine a sequence of dataarrays into one Dataset

    The input is a sequence of (name, dims, array) tuples and the output is an Dataset
    which combines the arrays assumings.

    This can be viewed as an alternative to some of xarrays built-in merging routines.
    It is more robust than xr.combine_by_coords, and more easy to use than
    xr.combine_nested, since it does not require created a nested list of dataarrays.

    When loading multiple netCDFs from disk it is often possible to parse some
    coordinate info from the file-name. For instance, one can easy build a sequence of
    DataArrays with elements like this::

        (name, (time, location), array)

    This function allows merging this list into a Dataset with variables names given by
    `name`, and the dimensions `time` and `location` combined.


    Args:
        datasets: a sequence of (name, dims, array) tuples.
        labels: the labels corresponding the dims part of the tuple above. This must
            be the same length as each dims.
    Returns:
        merged dataset


    Examples:
        >>> import xarray as xr
        >>> name = 'a'
        >>> arr = xr.DataArray([0.0], dims=["x"])
        >>> arrays = [
        ...         (name, ("a", 1), arr),
        ...         (name, ("b", 1), arr),
        ...         (name, ("a", 2), arr),
        ...         (name, ("b", 2), arr),
        ...     ]
        >>> combine_by_dims(arrays, ["letter", "number"])
        <xarray.Dataset>
        Dimensions:  (letter: 2, number: 2, x: 1)
        Coordinates:
        * number   (number) int64 1 2
        * letter   (letter) object 'a' 'b'
        Dimensions without coordinates: x
        Data variables:
            a        (letter, number, x) float64 0.0 0.0 0.0 0.0
    """
    variables = _collect_variables(datasets)
    return xr.Dataset({
        key: _combine_arrays(variables[key], labels)
        for key in variables

    })
