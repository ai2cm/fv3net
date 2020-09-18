from nudging_runfile import implied_precipitation, total_precipitation, sst_from_reference
from datetime import timedelta
from fv3gfs.util import Quantity
import pytest
import xarray as xr
import numpy as np


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


def test_total_precipitation_positive(xr_darray, timestep):
    model_precip = 0.0 * xr_darray
    column_moistening = xr_darray
    total_precip = total_precipitation(model_precip, column_moistening, timestep)
    assert total_precip.min().values >= 0


def test_implied_precipitation(quantity, timestep):
    output = implied_precipitation(quantity, quantity, quantity, timestep)
    assert isinstance(output, np.ndarray)


def test_sst_set_to_reference(quantity):
    land_sea_mask = Quantity.from_data_array(xr.DataArray(
        np.array([0., 1., 2.]), dims=["x"], attrs={"units": None}))
    reference_sfc_temp = Quantity.from_data_array(xr.DataArray(
        np.array([1., 1., 1.]), dims=["x"], attrs={"units": "degK"}))
    model_sfc_temp = Quantity.from_data_array(xr.DataArray(
        np.array([-1., -1., -1.]), dims=["x"], attrs={"units": "degK"}))
    assert np.allclose(
        sst_from_reference(reference_sfc_temp, model_sfc_temp, land_sea_mask),
        [1., -1., -1.]
    )
