import pytest
import xarray as xr
import numpy as np

import synth

from fv3net.regression.loaders._nudged import (
    _rename_ds_variables,
    _get_batch_func_args,
)


@pytest.fixture
def xr_dataset():

    dat = np.arange(120).reshape(5, 2, 3, 4)
    ds = xr.Dataset(
        data_vars={"data": (("time", "x", "y", "z"), dat)},
        coords={
            "time": np.arange(5),
            "x": np.arange(2),
            "y": np.arange(3),
            "z": np.arange(4),
        }
    )

    return ds


@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture to help load data files into a test based on the name of the
    module containing the test function.

    For example, if the name of the test file is named
    ``path/to/test_integration.py``, then and data in
    ``path/to/test_integration/`` will be copied into the temporary directory
    returned by this fixture.

    Returns:
        tmpdir (a temporary directory)

    Credit:
        https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


@pytest.mark.regression
def test_load_nudging_batches(datadir):

    pass


@pytest.mark.parametrize(
    "num_samples,samples_per_batch,num_batches",
    [(5, 2, None), (5, 4, 2)]
)
def test__get_batch_func_args(num_samples, samples_per_batch, num_batches):

    expected = [(slice(0, 2),), (slice(2, 4),)]
    args = _get_batch_func_args(num_samples, samples_per_batch, num_batches=num_batches)
    assert args == expected


@pytest.mark.parametrize(
    "num_samples,samples_per_batch,num_batches",
    [(5, 6, None), (5, 2, 6)]
)
def test__get_batch_func_args_failure(num_samples, samples_per_batch, num_batches):
    with pytest.raises(ValueError):
        _get_batch_func_args(num_samples, samples_per_batch, num_batches=num_batches)


def test__rename_ds_variables(xr_dataset):

    rename_vars = {"data": "new_data", "nonexistent": "doesntmatter"}

    renamed = _rename_ds_variables(xr_dataset, rename_vars)

    assert "new_data" in renamed
    assert "doesntmatter" not in renamed
