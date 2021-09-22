import pytest
from datetime import timedelta
import cftime
import xarray as xr
from dataclasses import asdict

from fv3net.diagnostics._shared import transform
from fv3net.diagnostics._shared.constants import DiagArg

# Transform params structure
# key - transform name, value Tuple(transform_args, transform_kwargs)
TRANSFORM_PARAMS = {
    "resample_time": (["1H"], {"time_slice": slice(0, -2)}),
    "daily_mean": ([timedelta(hours=2)], {}),
    "mask_to_sfc_type": (["sea"], {}),
    "subset_variables": ([("temperature")], {}),
    "mask_area": (["sea"], {}),
    "insert_absent_3d_output_placeholder": ([], {}),
    "select_2d_variables": ([], {}),
    "select_3d_variables": ([], {}),
    "regrid_zdim_to_pressure_levels": ([], {}),
}


def test_transform_default_params_present_here():
    """
    Test that all transforms have default parameters specified.
    Requires devs to pass basic test of transforms not adjusting
    input datasets in place.
    """

    for transform_name in transform._TRANSFORM_FNS.keys():
        assert transform_name in TRANSFORM_PARAMS


@pytest.fixture
def input_args():
    mask = [[[0, 1], [0, 2]]]
    area = [[[1, 2], [3, 4]]]
    latitude = [[[0, 0], [15, 15]]]
    p = [[[[10000, 10000], [10000, 10000]]], [[[20000, 20000], [20000, 20000]]]]

    ntimes = 5
    temp = [[[[0.5, 1.5], [2.5, 3.5]]]] * ntimes
    time_coord = [cftime.DatetimeJulian(2016, 4, 2, i + 1, 0, 0) for i in range(ntimes)]

    ds = xr.Dataset(
        data_vars={
            "SLMSKsfc": (["tile", "x", "y"], mask),
            "temperature": (["time", "tile", "x", "y"], temp),
            "var_3d": (["time", "z", "tile", "x", "y"], [p] * ntimes),
        },
        coords={"time": time_coord},
    )

    grid = xr.Dataset(
        data_vars={
            "lat": (["tile", "x", "y"], latitude),
            "area": (["tile", "x", "y"], area),
            "land_sea_mask": (["tile", "x", "y"], mask),
        }
    )
    delp = xr.DataArray(
        data=[p] * ntimes,
        dims=["time", "z", "tile", "x", "y"],
        name="pressure_thickness_of_atmospheric_layer",
        coords={"time": time_coord},
    )
    return DiagArg(ds, ds.copy(), grid, delp)


def test_transform_no_input_side_effects(input_args):
    """Test that all transforms do not operate on input datasets in place"""

    copied_args = {key: ds.copy() for key, ds in asdict(input_args).items()}

    for func_name, (t_args, t_kwargs) in TRANSFORM_PARAMS.items():

        transform_func = transform._TRANSFORM_FNS[func_name]
        transform_func(*t_args, input_args, **t_kwargs)

        for key, ds in asdict(input_args).items():
            xr.testing.assert_equal(ds, copied_args[key])


def test_subset_variables(input_args):
    output = transform.subset_variables(["SLMSKsfc", "other_var"], input_args)
    for subsetted_dataset in ["prediction", "verification"]:
        assert "SLMSKsfc" in getattr(output, subsetted_dataset)
        assert "temperature" not in getattr(output, subsetted_dataset)


def test_daily_mean_split_short_input(input_args):
    transform.daily_mean(timedelta(hours=10), input_args)


@pytest.mark.parametrize("region", [("global"), ("land"), ("sea"), ("tropics")])
def test__mask_array_global(input_args, region):
    grid = input_args.grid
    transform._mask_array(region, grid.area, grid.lat, grid.land_sea_mask)


def test_select_3d_variables(input_args):
    output = transform.select_3d_variables(input_args)
    for subsetted_dataset in ["prediction", "verification"]:
        print(getattr(output, subsetted_dataset))
        assert len(getattr(output, subsetted_dataset)) == 1
        assert "var_3d" in getattr(output, subsetted_dataset)


def test_select_2d_variables(input_args):
    output = transform.select_2d_variables(input_args)
    for subsetted_dataset in ["prediction", "verification"]:
        ds = getattr(output, subsetted_dataset)
        assert set(ds.data_vars) == {"SLMSKsfc", "temperature"}
