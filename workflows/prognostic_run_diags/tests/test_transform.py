import pytest
from datetime import timedelta
import cftime
import xarray as xr

from fv3net.diagnostics.prognostic_run import transform

# Transform params structure
# key - transform name, value Tuple(transform_args, transform_kwargs)
TRANSFORM_PARAMS = {
    "resample_time": (["1H"], {"time_slice": slice(0, -2)}),
    "daily_mean": ([timedelta(hours=2)], {}),
    "mask_to_sfc_type": (["sea"], {}),
    "subset_variables": ([("temperature")], {}),
    "mask_area": (["sea"], {}),
    "insert_absent_3d_output_placeholder": ([], {}),
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

    ntimes = 5
    temp = [[[[0.5, 1.5], [2.5, 3.5]]]] * ntimes
    time_coord = [cftime.DatetimeJulian(2016, 4, 2, i + 1, 0, 0) for i in range(ntimes)]

    ds = xr.Dataset(
        data_vars={
            "SLMSKsfc": (["tile", "x", "y"], mask),
            "temperature": (["time", "tile", "x", "y"], temp),
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

    return (ds, ds.copy(), grid)


def test_transform_no_input_side_effects(input_args):
    """Test that all transforms do not operate on input datasets in place"""

    copied_args = [ds.copy() for ds in input_args]

    for func_name, (t_args, t_kwargs) in TRANSFORM_PARAMS.items():

        transform_func = transform._TRANSFORM_FNS[func_name]
        transform_func(*t_args, input_args, **t_kwargs)

        for i, ds in enumerate(input_args):
            xr.testing.assert_equal(ds, copied_args[i])


def test_subset_variables(input_args):
    output = transform.subset_variables(["SLMSKsfc", "other_var"], input_args)
    for i in range(2):
        assert "SLMSKsfc" in output[i]
        assert "temperature" not in output[i]


def test_daily_mean_split_short_input(input_args):
    transform.daily_mean(timedelta(hours=10), input_args)


@pytest.mark.parametrize("region", [("global"), ("land"), ("sea"), ("tropics")])
def test__mask_array_global(input_args, region):
    ds, _, grid = input_args
    transform._mask_array(region, grid.area, grid.lat, grid.land_sea_mask)
