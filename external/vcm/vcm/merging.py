from typing import Dict, Tuple

import xarray as xr
from toolz import first


def _reduce_one_key_at_time(func, seqs, dims):
    """Turn flat dictionary into nested lists by keys"""
    # first groupby last element of key tuple
    if len(dims) == 0:
        return seqs
    else:
        # groupby everything but final key
        output = {}
        for key, array in seqs.items():
            val = output.get(key[:-1], None)
            output[key[:-1]] = func(val, array, key[-1], dims[-1])

        return _reduce_one_key_at_time(func, output, dims[:-1])


def _concat_binary_op(a, b, coord, dim):
    temp = b.assign_coords({dim: coord})
    if a is None:
        return temp
    else:
        return xr.concat([a, temp], dim=dim)


def combine_by_key(datasets: Dict[Tuple, xr.DataArray], dims=["time", "tile"]):
    """Merge dataarrays contained within a dictionary

    They keys of dictionary are tuples, the elements of which will become the
    coordinates and keys of the result.

    This can be viewed as an alternative to some of xarrays built-in merging routines.
    It is more robust than xr.combine_by_coords, and more easy to use than
    xr.combine_nested, since it does not require created a nested list of dataarrays.

    When loading multiple netCDFs from disk it is often possible to parse some
    coordinate info from the file-name. For instance, one can easy build a dictionary of
    DataArrays like this::

        {(variable, time, tile): array ...}

    This function allows combining the dataarrays along the "time" and "tile" part of
    the key to create an output like::

        {variable: array_with_time_tile_coords ...}


    Args:
        datasets: a dictionary with tuples for keys and DataArrays for values. The
            elements of the key represent coordinates along which the dataarrays will
            be concatenated.
        dims: the dimensions to associate with the keys of `datasets`. If the length
            of dims is `n`, then the final `n` elements of the key tuples will be
            turned into coordinates.


    Examples:
        >>> import pprint
        >>> arr = xr.DataArray([0.0], dims=["x"])
        >>> arrays = {("a", 1): arr, ("a", 2): arr, ("b", 1): arr, ("b", 2): arr}
        >>> arrays
        {('a', 1): <xarray.DataArray (x: 1)>
        array([0.])
        Dimensions without coordinates: x, ('a', 2): <xarray.DataArray (x: 1)>
        array([0.])
        Dimensions without coordinates: x, ('b', 1): <xarray.DataArray (x: 1)>
        array([0.])
        Dimensions without coordinates: x, ('b', 2): <xarray.DataArray (x: 1)>
        array([0.])
        Dimensions without coordinates: x}
        >>>  combine_by_key(arrays, dims=['letter', 'number'])
        <xarray.DataArray (letter: 2, number: 2, x: 1)>
        array([[[0.],
                [0.]],

            [[0.],
                [0.]]])
        Coordinates:
        * number   (number) int64 1 2
        * letter   (letter) object 'a' 'b'
        Dimensions without coordinates: x
        >>> pprint.pprint(combine_by_key(arrays, dims=['number']))
        {'a': <xarray.DataArray (number: 2, x: 1)>
        array([[0.],
            [0.]])
        Coordinates:
        * number   (number) int64 1 2
        Dimensions without coordinates: x,
        'b': <xarray.DataArray (number: 2, x: 1)>
        array([[0.],
            [0.]])
        Coordinates:
        * number   (number) int64 1 2
        Dimensions without coordinates: x}
    """
    output = _reduce_one_key_at_time(_concat_binary_op, datasets, dims)
    if first(output) == ():
        return output[()]
    if len(first(output)) == 1:
        return {key[0]: val for key, val in output.items()}
    else:
        return output
