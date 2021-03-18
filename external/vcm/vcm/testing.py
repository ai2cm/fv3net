from typing import Hashable, List, Tuple, Mapping
import contextlib
import pytest
import numpy as np
import joblib
import xarray

import io


@contextlib.contextmanager
def no_warning(*args):
    """Raise error if any errors occur. Takes the same arguments as 
    ``pytest.warns``.

    Example:

        >>> import warnings
        >>> from vcm.testing import no_warning
        >>> with no_warning(UserWarning):
        ...     warnings.warn(UserWarning("A warning"))
        ...
        Traceback (most recent call last):
        File "<ipython-input-9-c178a20fa539>", line 2, in <module>
            warnings.warn(UserWarning("A warning"))
        File "/Users/noah/.pyenv/versions/miniconda3-latest/envs/fv3net/lib/python3.7/contextlib.py", line 119, in __exit__
            next(self.gen)
        File "/Users/noah/workspace/fv3net/external/vcm/vcm/testing.py", line 14, in no_warning
            assert len(record) == 0
        AssertionError

    """  # noqa
    with pytest.warns(*args) as record:
        yield

    assert len(record) == 0


def checksum_dataarray(xobj) -> str:
    return joblib.hash(np.asarray(xobj))


def checksum_dataarray_mapping(
    d: Mapping[Hashable, xarray.DataArray]
) -> List[Tuple[Hashable, str]]:
    """Checksum a mapping of datarrays

    Returns:
        sorted list of (key, hash) combinations. This is sorted to simplify
        regression testing.
    
    """
    sorted_keys = sorted(d.keys())
    return [(key, checksum_dataarray(d[key])) for key in sorted_keys]


def regression_data(
    array: xarray.DataArray, attrs: bool = True, coords: bool = True
) -> str:
    f = io.StringIO()
    print("Array hash:", file=f)
    print(checksum_dataarray(array), file=f)
    if coords:
        print("Coordinate info:", file=f)
        for coord in array.coords:
            print("Coordinate ", coord, ":", np.asarray(array[coord]), file=f)
    if attrs:
        print("CDL Description of Data:")
        # This ensures the metadata is correct
        array.to_dataset(name="a").info(f)
    return f.getvalue()
