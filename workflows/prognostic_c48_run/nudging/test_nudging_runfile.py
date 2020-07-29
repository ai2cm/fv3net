from nudging_runfile import add_humidity_nudging_to_precip, total_precipitation
from datetime import timedelta
from fv3util import Quantity
import pytest
import xarray as xr
import numpy as np

DELP = "pressure_thickness_of_atmospheric_layer"
SPHUM_TENDENCY = "specific_humidity_tendency_due_to_nudging"
PRECIP = "total_precipitation"


@pytest.fixture()
def xr_darray():
    return xr.DataArray(
        np.array([[2.0, 6.0], [6.0, 2.0]]), dims=["x", "z"], attrs={"units": "m"}
    )


@pytest.fixture()
def quantity(xr_darray):
    return Quantity.from_data_array(xr_darray)


@pytest.fixture()
def timestep():
    return timedelta(seconds=1)


@pytest.fixture()
def state(quantity):
    return {DELP: quantity, PRECIP: quantity}


@pytest.fixture()
def tendencies(quantity):
    return {SPHUM_TENDENCY: quantity}


def test_total_precipitation_positive(xr_darray, timestep):
    model_precip = 0.0 * xr_darray
    column_moistening = xr_darray
    total_precip = total_precipitation(model_precip, column_moistening, timestep)
    assert total_precip.min().values >= 0


def test_add_humidity_nudging_to_precip(state, tendencies, timestep):
    state_copy = state.copy()
    add_humidity_nudging_to_precip(state, tendencies, timestep)
    assert (state[PRECIP].view[:] != state_copy[PRECIP].view[:]).any()


def test_add_humidity_nudging_only_acts_if_nudging_humidity(state, timestep):
    state_copy = state.copy()
    add_humidity_nudging_to_precip(state, {}, timestep)
    np.testing.assert_allclose(state[PRECIP].view[:], state_copy[PRECIP].view[:])
