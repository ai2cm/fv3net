import pytest
from loaders._utils import (
    shuffle,
    SAMPLE_DIM_NAME,
    select_fraction,
)
import xarray as xr
import numpy as np


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
def test_select_fraction(input_samples: int, fraction: float, expected_samples: int):
    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(
                np.repeat(np.arange(0, input_samples)[:, None], 10, axis=1),
                dims=[SAMPLE_DIM_NAME, "non_sample_dim"],
            )
        }
    )
    result = select_fraction(fraction)(ds)
    assert len(result[SAMPLE_DIM_NAME]) == expected_samples
    assert len(np.unique(result[SAMPLE_DIM_NAME].values)) == expected_samples
    assert np.array_equal(sorted(result[SAMPLE_DIM_NAME]), result[SAMPLE_DIM_NAME])
