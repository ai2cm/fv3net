import pytest
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from vcm.visualize.masking import (
    _mask_antimeridian_quads,
    _periodic_equal_or_less_than,
    _periodic_greater_than,
    _periodic_difference,
)
from vcm.visualize.plot_cube import mappable_var, plot_cube_axes, plot_cube
from vcm.visualize.plot_helpers import _get_var_label


@pytest.mark.parametrize(
    "lonb,central_longitude,expected",
    [
        (
            np.moveaxis(
                np.tile(
                    np.array([[0.0, 0.0], [120.0, 120.0], [240.0, 240.0]]), [6, 1, 1]
                ),
                0,
                -1,
            ),
            0.0,
            np.moveaxis(np.tile(np.array([[True], [False]]), [6, 1, 1]), 0, -1),
        )
    ],
)
def test__mask_antimeridian_quads(lonb, central_longitude, expected):
    np.testing.assert_array_equal(
        _mask_antimeridian_quads(lonb, central_longitude), expected
    )


@pytest.mark.parametrize(
    "x1,x2,period,expected",
    [
        (0.0, 5.0, 360.0, np.array(True)),
        (355.0, 0.0, 360.0, np.array(True)),
        (5.0, 355.0, 360.0, np.array(False)),
    ],
)
def test__periodic_equal_or_less_than(x1, x2, period, expected):
    np.testing.assert_array_equal(
        _periodic_equal_or_less_than(x1, x2, period), expected
    )


@pytest.mark.parametrize(
    "x1,x2,period,expected",
    [
        (0.0, 5.0, 360.0, np.array(False)),
        (355.0, 0.0, 360.0, np.array(False)),
        (5.0, 355.0, 360.0, np.array(True)),
    ],
)
def test__periodic_greater_than(x1, x2, period, expected):
    np.testing.assert_array_equal(_periodic_greater_than(x1, x2, period), expected)


@pytest.mark.parametrize(
    "x1,x2,period,expected",
    [
        (0.0, 5.0, 360.0, np.array(-5.0)),
        (355.0, 0.0, 360.0, np.array(-5.0)),
        (5.0, 355.0, 360.0, np.array(10.0)),
    ],
)
def test__periodic_difference(x1, x2, period, expected):
    np.testing.assert_allclose(_periodic_difference(x1, x2, period), expected)


@pytest.fixture()
def latb():
    return np.array(
        [
            [
                [-35.26439, -35.988922, -36.69255],
                [-33.79504, -34.505943, -35.197224],
                [-32.32569, -33.02107, -33.6981],
            ],
            [
                [-35.26439, -35.988922, -36.69255],
                [-33.79504, -34.505943, -35.197224],
                [-32.32569, -33.02107, -33.6981],
            ],
            [
                [35.26439, 35.988922, 36.69255],
                [35.988922, 36.76087, 37.512653],
                [36.69255, 37.512653, 38.313576],
            ],
            [
                [35.26439, 33.79504, 32.32569],
                [35.988922, 34.505943, 33.02107],
                [36.69255, 35.197224, 33.6981],
            ],
            [
                [35.26439, 33.79504, 32.32569],
                [35.988922, 34.505943, 33.02107],
                [36.69255, 35.197224, 33.6981],
            ],
            [
                [-35.26439, -35.988922, -36.69255],
                [-35.988922, -36.76087, -37.512653],
                [-36.69255, -37.512653, -38.313576],
            ],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def lonb():
    return np.array(
        [
            [
                [305.0, 306.5727, 308.1741],
                [305.0, 306.5727, 308.1741],
                [305.0, 306.5727, 308.1741],
            ],
            [
                [35.0, 36.572693, 38.174114],
                [35.0, 36.572693, 38.174114],
                [35.0, 36.572693, 38.174114],
            ],
            [
                [35.0, 36.572693, 38.174114],
                [33.427307, 35.0, 36.606304],
                [31.825886, 33.393696, 35.0],
            ],
            [
                [125.0, 125.0, 125.0],
                [126.57269, 126.57269, 126.57269],
                [128.17412, 128.17412, 128.17412],
            ],
            [
                [215.0, 215.0, 215.0],
                [216.5727, 216.5727, 216.5727],
                [218.17412, 218.17412, 218.17412],
            ],
            [
                [215.0, 213.4273, 211.82588],
                [216.5727, 215.0, 213.39369],
                [218.17412, 216.60631, 215.0],
            ],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def lon():
    return np.array(
        [
            [[305.7829, 307.3699], [305.78317, 307.37018]],
            [[35.782913, 37.369915], [35.78316, 37.370167]],
            [[35.0, 36.588547], [33.411453, 35.0]],
            [[125.78291, 125.783165], [127.36991, 127.37016]],
            [[215.78291, 215.78316], [217.36992, 217.37016]],
            [[215.0, 213.41145], [216.58855, 215.0]],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def lat():
    return np.array(
        [
            [[-34.891106, -35.59881], [-33.414417, -34.10818]],
            [[-34.891106, -35.59881], [-33.414417, -34.10818]],
            [[36.00591, 36.74402], [36.74402, 37.530376]],
            [[34.891106, 33.414417], [35.59881, 34.10818]],
            [[34.891106, 33.414417], [35.59881, 34.10818]],
            [[-36.00591, -36.74402], [-36.74402, -37.530376]],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def t2m():
    return np.array(
        [
            [
                [[285.24548, 285.91785], [286.58337, 286.31308]],
                [[289.17456, 288.05328], [289.89584, 289.19724]],
                [[300.79932, 297.65076], [293.8577, 293.46573]],
                [[300.42297, 301.45743], [305.09097, 301.1763]],
                [[293.6815, 293.9053], [293.52594, 293.69046]],
                [[287.85144, 287.42148], [287.58282, 287.13138]],
            ],
            [
                [[285.24548, 285.91785], [286.58337, 286.31308]],
                [[289.17456, 288.05328], [289.89584, 289.19724]],
                [[300.79932, 297.65076], [293.8577, 293.46573]],
                [[300.42297, 301.45743], [305.09097, 301.1763]],
                [[293.6815, 293.9053], [293.52594, 293.69046]],
                [[287.85144, 287.42148], [287.58282, 287.13138]],
            ],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def sample_dataset(latb, lonb, lat, lon, t2m):
    dataset = xr.Dataset(
        {
            "t2m": (["time", "tile", "grid_yt", "grid_xt"], t2m),
            "lat": (["tile", "grid_yt", "grid_xt"], lat),
            "lon": (["tile", "grid_yt", "grid_xt"], lon),
            "latb": (["tile", "grid_y", "grid_x"], latb),
            "lonb": (["tile", "grid_y", "grid_x"], lonb),
        }
    )
    dataset = dataset.assign_coords(
        {
            "time": np.arange(2),
            "tile": np.arange(6),
            "grid_x": np.arange(3.0),
            "grid_y": np.arange(3.0),
            "grid_xt": np.arange(2.0),
            "grid_yt": np.arange(2.0),
        }
    )
    return dataset


def test_mappable_var_all_sizes(sample_dataset):
    mappable_ds = mappable_var(sample_dataset, "t2m").isel(time=0)
    sizes_expected = {"grid_x": 3, "grid_yt": 2, "grid_y": 3, "grid_xt": 2, "tile": 6}
    assert mappable_ds.sizes == sizes_expected


def test_mappable_var_coords(sample_dataset):
    mappable_ds_coords = set(mappable_var(sample_dataset, "t2m").coords)
    coords_expected = set(["lat", "latb", "lon", "lonb", "tile", "time"])
    assert mappable_ds_coords == coords_expected


def test_mappable_var_sizes(sample_dataset):
    mappable_var_sizes = mappable_var(sample_dataset, "t2m").isel(time=0)["t2m"].sizes
    sizes_expected = {"grid_yt": 2, "grid_xt": 2, "tile": 6}
    assert mappable_var_sizes == sizes_expected


@pytest.mark.parametrize(
    "plotting_function", [("pcolormesh"), ("contour"), ("contourf")]
)
def test_plot_cube_axes(sample_dataset, plotting_function):
    ds = mappable_var(sample_dataset, "t2m").isel(time=0)
    ax = plt.axes(projection=ccrs.Robinson())
    plot_cube_axes(
        ds.t2m.values,
        ds.lat.values,
        ds.lon.values,
        ds.latb.values,
        ds.lonb.values,
        plotting_function,
        ax=ax,
    )


@pytest.mark.parametrize(
    "plotting_function", [("pcolormesh"), ("contour"), ("contourf")]
)
def test_plot_cube_with_facets(sample_dataset, plotting_function):
    f, axes, hs, cbar = plot_cube(
        mappable_var(sample_dataset, "t2m"),
        col="time",
        plotting_function=plotting_function,
    )


@pytest.mark.parametrize(
    "plotting_function", [("pcolormesh"), ("contour"), ("contourf")]
)
def test_plot_cube_on_axis(sample_dataset, plotting_function):
    ax = plt.axes(projection=ccrs.Robinson())
    f, axes, hs, cbar = plot_cube(
        mappable_var(sample_dataset, "t2m").isel(time=0),
        plotting_function=plotting_function,
        ax=ax,
    )


@pytest.mark.parametrize(
    "attrs,var_name,expected_label",
    [
        ({}, "temp", "temp"),
        ({"long_name": "air_temperature"}, "temp", "air_temperature"),
        ({"units": "degK"}, "temp", "temp [degK]"),
        (
            {"long_name": "air_temperature", "units": "degK"},
            "temp",
            "air_temperature [degK]",
        ),
    ],
)
def test__get_var_label(attrs, var_name, expected_label):
    assert _get_var_label(attrs, var_name) == expected_label
