import pytest
from loaders._utils import (
    nonderived_variables,
    shuffle,
    SAMPLE_DIM_NAME,
    select_first_samples,
)
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


@pytest.mark.parametrize(
    "input_samples, fraction, expected_samples",
    [
        pytest.param(5, 1, 5, id="keep_all"),
        pytest.param(20, 0.75, 15, id="discard_some"),
    ],
)
def test_select_first_samples(
    input_samples: int, fraction: float, expected_samples: int
):
    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(
                np.repeat(np.arange(0, input_samples)[:, None], 10, axis=1),
                dims=[SAMPLE_DIM_NAME, "non_sample_dim"],
            )
        }
    )
    result = select_first_samples(fraction)(ds)
    assert len(result[SAMPLE_DIM_NAME]) == expected_samples
    assert result["a"].values.max() == expected_samples - 1
    assert result["a"].values.min() == 0
