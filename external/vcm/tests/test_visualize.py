import pytest
import numpy as np
import xarray as xr
from vcm.visualize.masking import (
    _mask_antimeridian_quads,
    _periodic_equal_or_less_than,
    _periodic_greater_than,
    _periodic_difference,
)
from vcm.visualize.plot_cube import plot_cube_axes


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
            [[-35.26439, -35.26439], [35.26439, 35.26439]],
            [[-35.26439, -35.26439], [35.26439, 35.26439]],
            [[35.26439, 35.26439], [35.26439, 35.26439]],
            [[35.26439, -35.26439], [35.26439, -35.26439]],
            [[35.26439, -35.26439], [35.26439, -35.26439]],
            [[-35.26439, -35.26439], [-35.26439, -35.26439]],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def lonb():
    return np.array(
        [
            [[305.0, 35.0], [305.0, 35.0]],
            [[35.0, 125.0], [35.0, 125.0]],
            [[35.0, 125.0], [305.0, 215.0]],
            [[125.0, 125.0], [215.0, 215.0]],
            [[215.0, 215.0], [305.0, 305.0]],
            [[215.0, 125.0], [305.0, 35.0]],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def lon():
    return np.array(
        [
            [[351.03876]],
            [[81.03876]],
            [[215.0]],
            [[171.03876]],
            [[261.03876]],
            [[35.0]],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def lat():
    return np.array(
        [
            [[1.038589]],
            [[1.038589]],
            [[88.531136]],
            [[-1.038589]],
            [[-1.038589]],
            [[-88.531136]],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def t2m():
    return np.array(
        [
            [[296.26602]],
            [[301.1429]],
            [[273.25952]],
            [[301.7855]],
            [[294.68396]],
            [[220.30968]],
        ],
        dtype=np.float32,
    )


def test_plot_cube_axes(t2m, lat, lon, latb, lonb):
    plot_cube_axes(t2m, lat, lon, latb, lonb, "pcolormesh")


@pytest.fixture()
def sample_dataset(latb, lonb, lat, lon, t2m):
    dataset = xr.Dataset(
        {
            "t2m": (["tile", "grid_yt", "grid_xt"], t2m),
            "lat": (["tile", "grid_yt", "grid_xt"], lat),
            "lon": (["tile", "grid_yt", "grid_xt"], lon),
            "latb": (["tile", "grid_y", "grid_x"], latb),
            "lonb": (["tile", "grid_y", "grid_x"], lonb),
        }
    )
    dataset = dataset.assign_coords(
        {
            "tile": np.arange(6),
            "grid_x": np.arange(2.0),
            "grid_y": np.arange(2.0),
            "grid_xt": np.arange(1.0),
            "grid_yt": np.arange(1.0),
        }
    )
    return dataset


# def test__plot_cube()
