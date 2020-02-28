import numpy as np
import xarray as xr
import pytest
from datetime import datetime
from dask.delayed import delayed
from dask.array import Array

from vcm.cubedsphere.constants import TIME_FMT
from vcm.convenience import (
    open_delayed,
    parse_timestep_from_path,
    parse_time_from_string,
)


@pytest.fixture()
def dataset():
    arr = np.random.rand(100, 10)
    coords = dict(time=np.arange(100), x=np.arange(10),)
    return xr.Dataset(
        {"a": (["time", "x"], arr), "b": (["time", "x"], arr)}, coords=coords
    )


def test_open_delayed(dataset):
    a_delayed = delayed(lambda x: x)(dataset)
    ds = open_delayed(a_delayed, schema=dataset)

    xr.testing.assert_equal(dataset, ds.compute())
    assert isinstance(ds["a"].data, Array)


def test_open_delayed_fills_nans(dataset):
    ds_no_b = dataset[["a"]]
    # wrap idenity with delated object
    a_delayed = delayed(lambda x: x)(ds_no_b)
    ds = open_delayed(a_delayed, schema=dataset)

    # test that b is filled with anans
    b = ds["b"].compute()
    assert np.all(np.isnan(b))
    assert b.dims == dataset["b"].dims
    assert b.dtype == dataset["b"].dtype


def test_extract_timestep_from_path():

    timestep = "20160801.001500"
    good_path = f"gs://path/to/timestep/{timestep}/"
    assert parse_timestep_from_path(good_path) == timestep


def test_extract_timestep_from_path_with_no_timestep_in_path():

    with pytest.raises(ValueError):
        bad_path = "gs://path/to/not/a/timestep/"
        parse_timestep_from_path(bad_path)


def test_datetime_from_string():

    year = 2008
    month = 3
    day = 24
    hour = 16
    minute = 50
    second = 41
    time_str = f"{year:04d}{month:02d}{day:02d}.{hour:02d}{minute:02d}{second:02d}"

    parsed_datetime = parse_time_from_string(time_str)
    assert parsed_datetime.year == year
    assert parsed_datetime.month == month
    assert parsed_datetime.day == day
    assert parsed_datetime.hour == hour
    assert parsed_datetime.minute == minute
    assert parsed_datetime.second == second
