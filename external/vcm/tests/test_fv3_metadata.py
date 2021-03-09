import pytest
import xarray as xr
import numpy as np
import cftime

from vcm.fv3.metadata import (
    standardize_fv3_diagnostics,
    gfdl_to_standard,
    standard_to_gfdl,
)


DIM_NAMES = {"x", "y", "tile", "time"}
DATA_VAR = "a"
TILE_RANGE = np.arange(6)
TIME_DIM = 10


def _create_raw_dataset(
    dims, tile_range, time_coord_random, attrs,
):
    spatial_dims = {dim for dim in dims if dim not in ["tile", "time"]}
    coords = {dim: np.arange(i + 1) for i, dim in enumerate(sorted(spatial_dims))}
    sizes = {dim: len(coords[dim]) for dim in spatial_dims}
    coords.update({"tile": tile_range})
    sizes["tile"] = tile_range.shape[0]
    if time_coord_random:
        time_coord = [
            cftime.DatetimeJulian(2016, 1, n + 1, 0, 0, 0, np.random.randint(100))
            for n in range(TIME_DIM)
        ]
    else:
        time_coord = [
            cftime.DatetimeJulian(2016, 1, n + 1, 0, 0, 0) for n in range(TIME_DIM)
        ]
    coords.update({"time": time_coord})
    sizes["time"] = TIME_DIM
    arr = np.ones([sizes[k] for k in sorted(sizes)])
    data_array = xr.DataArray(arr, dims=sorted(dims))
    data_array.attrs.update(attrs)
    return xr.Dataset({DATA_VAR: data_array}, coords=coords)


@pytest.mark.parametrize(
    ["dims", "tile_range", "time_coord_random", "attrs", "expected_dims"],
    [
        pytest.param(DIM_NAMES, TILE_RANGE, False, {}, DIM_NAMES, id="dims_no_change"),
        pytest.param(
            {"grid_xt", "grid_yt", "tile", "time"},
            TILE_RANGE,
            False,
            {},
            DIM_NAMES,
            id="dims_change_xy",
        ),
        pytest.param(
            {"grid_xt", "grid_yt", "tile", "time", "some_other"},
            TILE_RANGE,
            False,
            {},
            {"x", "y", "tile", "time", "some_other"},
            id="dims_change_xy_not_other",
        ),
        pytest.param(
            DIM_NAMES, np.arange(1, 7), False, {}, DIM_NAMES, id="change_tile_range"
        ),
        pytest.param(
            DIM_NAMES, TILE_RANGE, True, {}, DIM_NAMES, id="random_time_values"
        ),
        pytest.param(
            DIM_NAMES,
            TILE_RANGE,
            False,
            {"units": "best units"},
            DIM_NAMES,
            id="attrs_units_only",
        ),
        pytest.param(
            DIM_NAMES,
            TILE_RANGE,
            False,
            {"long_name": "name is long!"},
            DIM_NAMES,
            id="attrs_long_name_only",
        ),
        pytest.param(
            DIM_NAMES,
            TILE_RANGE,
            False,
            {"units": "trees", "long_name": "number of U.S. trees"},
            DIM_NAMES,
            id="attrs_both_only",
        ),
        pytest.param(
            DIM_NAMES,
            TILE_RANGE,
            False,
            {"description": "a description will be converted to a longname"},
            DIM_NAMES,
            id="attrs_description_only",
        ),
    ],
)
def test_standardize_fv3_diagnostics(
    dims, tile_range, time_coord_random, attrs, expected_dims, regtest,
):
    diag = _create_raw_dataset(dims, tile_range, time_coord_random, attrs)
    standardized_ds = standardize_fv3_diagnostics(diag)

    assert set(standardized_ds.dims) == expected_dims
    xr.testing.assert_equal(
        standardized_ds.tile,
        xr.DataArray(TILE_RANGE, dims=["tile"], coords={"tile": TILE_RANGE}),
    )
    assert "long_name" in standardized_ds[DATA_VAR].attrs
    assert "units" in standardized_ds[DATA_VAR].attrs
    if "description" in attrs:
        assert standardized_ds[DATA_VAR].attrs["long_name"] == attrs["description"]

    standardized_ds.info(regtest)
    with regtest:
        print(f"\n{standardized_ds.time}")


def test_gfdl_to_standard_dims_correct():
    data = xr.Dataset(
        {
            "2d": (["tile", "grid_yt", "grid_xt"], np.ones((1, 1, 1))),
            "3d": (["tile", "pfull", "grid_yt", "grid_xt"], np.ones((1, 1, 1, 1))),
        }
    )

    ans = gfdl_to_standard(data)
    assert set(ans.dims) == {"tile", "x", "y", "z"}


def test_gfdl_to_standard_is_inverse_of_standard_to_gfdl():

    data = xr.Dataset(
        {
            "2d": (["tile", "grid_yt", "grid_xt"], np.ones((1, 1, 1))),
            "3d": (["tile", "pfull", "grid_yt", "grid_xt"], np.ones((1, 1, 1, 1))),
        }
    )

    back = standard_to_gfdl(gfdl_to_standard(data))
    xr.testing.assert_equal(data, back)
