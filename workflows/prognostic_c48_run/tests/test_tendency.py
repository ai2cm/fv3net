import numpy as np
import pytest
import xarray as xr

from runtime.loop import (
    fillna_tendency,
    fillna_tendencies,
    prepare_agrid_wind_tendencies,
    transform_agrid_wind_tendencies,
)
from runtime.names import (
    EASTWARD_WIND_TENDENCY,
    NORTHWARD_WIND_TENDENCY,
    X_WIND_TENDENCY,
)


TENDENCY_ATTRS = {"units": "m/s/s"}


def test_fillna_tendency():
    tendency = xr.DataArray([1.0, np.nan, 2.0], dims=["z"], name="dQ1")
    expected_filled_tendency = xr.DataArray([1.0, 0.0, 2.0], dims=["z"], name="dQ1")
    expected_filled_fraction = xr.DataArray(1.0 / 3.0, name="dQ1_filled_frac")
    result_filled_tendency, result_filled_fraction = fillna_tendency(tendency)
    xr.testing.assert_identical(result_filled_tendency, expected_filled_tendency)
    xr.testing.assert_identical(result_filled_fraction, expected_filled_fraction)


def test_fillna_tendencies():
    tendencies = {"dQ1": xr.DataArray([1.0, np.nan, 2.0], dims=["z"], name="dQ1")}
    expected_filled_tendencies = {
        "dQ1": xr.DataArray([1.0, 0.0, 2.0], dims=["z"], name="dQ1")
    }
    expected_tendencies_filled_frac = {
        "dQ1_filled_frac": xr.DataArray(1.0 / 3.0, name="dQ1_filled_frac")
    }
    result_filled_tendencies, result_tendencies_filled_frac = fillna_tendencies(
        tendencies
    )

    for name in expected_filled_tendencies:
        expected = expected_filled_tendencies[name]
        result = result_filled_tendencies[name]
        xr.testing.assert_identical(result, expected)

    for name in expected_tendencies_filled_frac:
        expected = expected_tendencies_filled_frac[name]
        result = result_tendencies_filled_frac[name]
        xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    ("tendencies", "expected_dQu", "expected_dQv"),
    [
        (
            {EASTWARD_WIND_TENDENCY: xr.DataArray(5.0)},
            xr.DataArray(5.0, attrs=TENDENCY_ATTRS),
            xr.DataArray(0.0, attrs=TENDENCY_ATTRS),
        ),
        (
            {NORTHWARD_WIND_TENDENCY: xr.DataArray(3.0)},
            xr.DataArray(0.0, attrs=TENDENCY_ATTRS),
            xr.DataArray(3.0, attrs=TENDENCY_ATTRS),
        ),
        (
            {
                EASTWARD_WIND_TENDENCY: xr.DataArray(5.0),
                NORTHWARD_WIND_TENDENCY: xr.DataArray(3.0),
            },
            xr.DataArray(5.0, attrs=TENDENCY_ATTRS),
            xr.DataArray(3.0, attrs=TENDENCY_ATTRS),
        ),
        (
            {NORTHWARD_WIND_TENDENCY: xr.DataArray(np.float32(3.0))},
            xr.DataArray(0.0, attrs=TENDENCY_ATTRS),
            xr.DataArray(3.0, attrs=TENDENCY_ATTRS),
        ),
    ],
    ids=["only-dQu", "only-dQv", "both", "float32-input"],
)
def test_prepare_agrid_wind_tendencies(tendencies, expected_dQu, expected_dQv):
    result_dQu, result_dQv = prepare_agrid_wind_tendencies(tendencies)
    xr.testing.assert_identical(result_dQu, expected_dQu)
    xr.testing.assert_identical(result_dQv, expected_dQv)

    # assert_identical doesn't check the dtype.  We could use
    # vcm.xarray_utils.assert_identical_including_dtype, but this is a little
    # more explicit.
    assert result_dQu.dtype == np.float64
    assert result_dQv.dtype == np.float64


def test_transform_agrid_wind_tendencies_mixed_coordinates_error():
    tendencies = {
        EASTWARD_WIND_TENDENCY: xr.DataArray(
            [1.0, np.nan, 2.0], dims=["z"], name=EASTWARD_WIND_TENDENCY
        ),
        X_WIND_TENDENCY: xr.DataArray(
            [1.0, np.nan, 3.0], dims=["z"], name=X_WIND_TENDENCY
        ),
    }
    with pytest.raises(ValueError, match="Simultaneously"):
        transform_agrid_wind_tendencies(tendencies)
