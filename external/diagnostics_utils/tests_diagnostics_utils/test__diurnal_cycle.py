import numpy as np
from datetime import datetime
import pytest
import xarray as xr

import loaders
from diagnostics_utils._diurnal_cycle import (
    _local_time,
    bin_diurnal_cycle,
    create_diurnal_cycle_dataset,
    DIURNAL_CYCLE_DIM,
    SURFACE_TYPE_DIM,
)

ADDITIONAL_DIM = "additional_dim"
ADDITIONAL_COORDS = ["extra_dim_0", "extra_dim_1", "extra_dim_2"]


def _generate_time_coords(time_of_day: tuple):
    h, m, s = time_of_day
    return datetime(2016, 1, 1, h, m, s)


@pytest.fixture
def da_lon():
    return xr.DataArray([0.0, 180.0], dims=["x"], coords={"x": [0, 1]})


@pytest.fixture
def da_sfc_type():
    return xr.DataArray([0, 1], dims=["x"], coords={"x": [0, 1]},).rename(
        SURFACE_TYPE_DIM
    )


@pytest.fixture
def ds(request):
    values, times_of_day = request.param
    time_coords = list(map(_generate_time_coords, times_of_day))
    da = xr.DataArray(
        [values for time in time_coords],
        dims=["time", "x"],
        coords={"x": [0, 1], "time": time_coords},
    ).rename("test_var")
    da_additional_dim = xr.DataArray(
        [
            [i * np.array(values) for time in time_coords]
            for i in range(len(ADDITIONAL_COORDS))
        ],
        dims=[ADDITIONAL_DIM, "time", "x"],
        coords={"x": [0, 1], "time": time_coords, ADDITIONAL_DIM: ADDITIONAL_COORDS},
    ).rename("test_var_additional_dim")

    return xr.Dataset({"test_var": da, "test_var_additional_dim": da_additional_dim})


@pytest.fixture
def ds_with_dataset_dim(request):
    values, times_of_day = request.param
    time_coords = list(map(_generate_time_coords, times_of_day))
    da = xr.DataArray(
        [values for time in time_coords],
        dims=["time", "x", "dataset"],
        coords={"x": [0, 1], "time": time_coords},
    ).rename("test_var")
    da_additional_dim = xr.DataArray(
        [
            [i * np.array(values) for time in time_coords]
            for i in range(len(ADDITIONAL_COORDS))
        ],
        dims=[ADDITIONAL_DIM, "time", "x", "dataset"],
        coords={"x": [0, 1], "time": time_coords, ADDITIONAL_DIM: ADDITIONAL_COORDS},
    ).rename("test_var_additional_dim")

    return xr.Dataset(
        {"test_var": da, "test_var_additional_dim": da_additional_dim}
    ).stack(time_dataset_dim=["time", "dataset"])


@pytest.mark.parametrize(
    "time_gmt, expected",
    (
        [datetime(2016, 1, 1, 0, 0, 0), [0, 6, 12, 18, 0]],
        [datetime(2017, 2, 10, 0, 0, 0), [0, 6, 12, 18, 0]],
        [datetime(2016, 1, 1, 12, 30, 0), [12.5, 18.5, 0.5, 6.5, 12.5]],
    ),
)
def test__local_time(time_gmt, expected):
    da_lon = xr.DataArray(
        [0.0, 90.0, 180.0, -90.0, 360], dims=["x"], coords={"x": range(5)},
    )
    da_time = xr.DataArray([time_gmt], dims=["time"], coords={"time": [time_gmt]})
    assert np.allclose(_local_time(da_lon, da_time).values, expected)


@pytest.mark.parametrize(
    "ds, diurnal_bin_means",
    (
        [[[1.0, 2.0], [(0, 30, 0), (6, 30, 0)]], [1, 1, 2, 2]],
        [[[1.0, 2.0], [(0, 30, 0), (0, 30, 0)]], [1.0, np.nan, 2.0, np.nan]],
        [[[1.0, 2.0], [(0, 30, 0), (12, 30, 0)]], [1.5, np.nan, 1.5, np.nan]],
    ),
    indirect=["ds"],
)
def test_bin_diurnal_cycle(ds, diurnal_bin_means, da_lon):
    da_var = ds["test_var"]
    assert np.allclose(
        bin_diurnal_cycle(da_var, da_lon, n_bins=4), diurnal_bin_means, equal_nan=True
    )


@pytest.mark.parametrize(
    "ds_with_dataset_dim, diurnal_bin_means",
    (
        [[[[1.0, 1.0], [2.0, 2.0]], [(0, 30, 0), (6, 30, 0)]], [1, 1, 2, 2]],
        [
            [[[1.0, 1.0], [2.0, 2.0]], [(0, 30, 0), (0, 30, 0)]],
            [1.0, np.nan, 2.0, np.nan],
        ],
        [
            [[[1.0, 1.0], [2.0, 2.0]], [(0, 30, 0), (12, 30, 0)]],
            [1.5, np.nan, 1.5, np.nan],
        ],
    ),
    indirect=["ds_with_dataset_dim"],
)
def test_bin_diurnal_cycle_with_dataset_dim(
    ds_with_dataset_dim, diurnal_bin_means, da_lon
):
    da_var = ds_with_dataset_dim["test_var"]
    assert np.allclose(
        bin_diurnal_cycle(da_var, da_lon, n_bins=4), diurnal_bin_means, equal_nan=True
    )


@pytest.mark.parametrize(
    "ds", ([[[1.0, 2.0], [(0, 30, 0), (6, 30, 0)]]]), indirect=True
)
def test_create_diurnal_cycle_dataset_correct_dims(ds, da_lon, da_sfc_type):
    ds_diurnal = create_diurnal_cycle_dataset(
        ds, da_lon, da_sfc_type, diurnal_vars=["test_var", "test_var_additional_dim"]
    )
    assert set(ds_diurnal["test_var"].dims) == {DIURNAL_CYCLE_DIM, SURFACE_TYPE_DIM}
    assert set(ds_diurnal["test_var_additional_dim"].dims) == {
        DIURNAL_CYCLE_DIM,
        ADDITIONAL_DIM,
        SURFACE_TYPE_DIM,
    }
    assert set(ds_diurnal[ADDITIONAL_DIM].values) == set(ADDITIONAL_COORDS)


@pytest.mark.parametrize(
    "ds", ([[[1.0, 1.0], [(0, 30, 0), (6, 30, 0)]]]), indirect=True
)
def test_create_diurnal_cycle_dataset_correct_additional_coords(
    ds, da_lon, da_sfc_type
):
    ds_diurnal = create_diurnal_cycle_dataset(
        ds,
        da_lon,
        da_sfc_type,
        diurnal_vars=["test_var", "test_var_additional_dim"],
        n_bins=1,
    )
    for i, coord in enumerate(ADDITIONAL_COORDS):
        da = ds_diurnal["test_var_additional_dim"].sel({ADDITIONAL_DIM: coord})
        assert da.values[0] == i
