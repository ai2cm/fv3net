from typing import Any, Iterable, Sequence, Tuple
from collections import defaultdict
import pandas as pd

import xarray as xr


def _reduce_one_key_at_time(seqs, dims):
    """Turn flat dictionary into nested lists by keys"""
    # first groupby last element of key tuple
    if len(dims) == 0:
        return seqs
    else:
        # groupby everything but final key
        output = defaultdict(list)
        labels = defaultdict(list)
        for key, array in seqs.items():
            output[key[:-1]].append(array)
            labels[key[:-1]].append(key[-1])

        # concat the output
        out2 = {}
        for key in output:
            out2[key] = xr.concat(output[key], dim=pd.Index(labels[key], name=dims[-1]))

        return _reduce_one_key_at_time(out2, dims[:-1])


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
    datasets_dict = {(name,) + dims: array for name, dims, array in datasets}
    output = _reduce_one_key_at_time(datasets_dict, labels)
    return xr.Dataset({key[0]: val for key, val in output.items()})
