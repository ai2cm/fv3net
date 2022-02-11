import pytest
from loaders._utils import nonderived_variables, shuffle, SAMPLE_DIM_NAME
import xarray as xr
import numpy as np


@pytest.mark.parametrize(
    "requested, available, nonderived",
    (
        [["dQ1", "dQ2"], ["dQ1", "dQ2"], ["dQ1", "dQ2"]],
        [
            ["dQ1", "dQ2", "dQu", "dQv", "cos_zenith_angle"],
            ["dQ1", "dQ2"],
            ["dQ1", "dQ2", "dQxwind", "dQywind"],
        ],
        [
            ["dQ1", "dQ2", "dQu", "dQv", "cos_zenith_angle"],
            ["dQ1", "dQ2", "dQu", "dQv"],
            ["dQ1", "dQ2", "dQu", "dQv"],
        ],
    ),
)
def test_nonderived_variable_names(requested, available, nonderived):
    assert set(nonderived_variables(requested, available)) == set(nonderived)


def test_shuffle_retains_values():
    ds = xr.Dataset(  # random zeros and ones
        data_vars={
            "a": xr.DataArray(
                np.random.randint(low=0, high=2, size=[100, 10], dtype=int),
                dims=[SAMPLE_DIM_NAME, "z"],
            )
        }
    )
    shuffled = shuffle(ds)
    assert not np.all(ds["a"].values == shuffled["a"].values)
    assert np.sum(ds["a"].values) == np.sum(shuffled["a"].values)
