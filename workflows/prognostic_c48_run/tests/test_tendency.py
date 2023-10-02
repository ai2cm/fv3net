import numpy as np
import pytest
import xarray as xr

from runtime.tendency import (
    add_tendency,
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


def test_add_tendency():
    state = {"air_temperature": xr.DataArray([0.0, 1.0, 2.0], dims=["z"])}
    tendency = {"dQ1": xr.DataArray([1.0, 0.0, 1.0], dims=["z"])}
    updated_state = add_tendency(state, tendency, 1.0)
    expected_after = {"air_temperature": xr.DataArray([1.0, 1.0, 3.0], dims=["z"])}
    for name, state in updated_state.items():
        xr.testing.assert_allclose(expected_after[name], state)


def test_fillna_tendencies():
    dQ1 = xr.DataArray([1.0, np.nan, 2.0], dims=["z"], name="dQ1")
    dQ1_filled = xr.DataArray([1.0, 0.0, 2.0], dims=["z"], name="dQ1")
    dQ1_filled_frac = xr.DataArray(1.0 / 3.0, name="dQ1_filled_frac")
    tendencies = {"dQ1": dQ1}
    expected_filled_tendencies = {"dQ1": dQ1_filled}
    expected_tendencies_filled_frac = {"dQ1_filled_frac": dQ1_filled_frac}
    results = fillna_tendencies(tendencies)
    result_filled_tendencies, result_tendencies_filled_frac = results

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
        EASTWARD_WIND_TENDENCY: xr.DataArray([1.0, np.nan, 2.0], dims=["z"]),
        X_WIND_TENDENCY: xr.DataArray([1.0, np.nan, 3.0], dims=["z"]),
    }
    with pytest.raises(ValueError, match="Simultaneously"):
        transform_agrid_wind_tendencies(tendencies)
