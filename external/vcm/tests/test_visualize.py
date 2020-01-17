import pytest
import numpy as np
from vcm.visualize.masking import (
    _mask_antimeridian_quads,
    _periodic_equal_or_less_than,
    _periodic_greater_than,
    _periodic_difference,
)


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
