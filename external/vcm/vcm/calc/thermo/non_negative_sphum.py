import xarray as xr
from typing import Tuple, Optional
from .local import moist_static_energy_tendency, temperature_tendency


def non_negative_sphum(
    sphum: xr.DataArray, dQ1: xr.DataArray, dQ2: xr.DataArray, dt: float
) -> Tuple[xr.DataArray, xr.DataArray]:
    delta = dQ2 * dt
    reduction_ratio = (-sphum) / (dt * dQ2)  # type: ignore
    dQ1_updated = xr.where(sphum + delta >= 0, dQ1, reduction_ratio * dQ1)
    dQ2_updated = xr.where(sphum + delta >= 0, dQ2, reduction_ratio * dQ2)
    return dQ1_updated, dQ2_updated


def update_moisture_tendency_to_ensure_non_negative_humidity(
    sphum: xr.DataArray, q2: xr.DataArray, dt: float
) -> xr.DataArray:
    return xr.where(sphum + q2 * dt >= 0, q2, -sphum / dt)


def update_temperature_tendency_to_conserve_mse(
    q1: xr.DataArray, q2_old: xr.DataArray, q2_new: xr.DataArray
) -> xr.DataArray:
    mse_tendency = moist_static_energy_tendency(q1, q2_old)
    q1_new = temperature_tendency(mse_tendency, q2_new)
    return q1_new


def non_negative_sphum_mse_conserving(
    sphum: xr.DataArray, q2: xr.DataArray, dt: float, q1: Optional[xr.DataArray] = None
) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    q2_new = update_moisture_tendency_to_ensure_non_negative_humidity(sphum, q2, dt)
    if q1 is not None:
        q1_new = update_temperature_tendency_to_conserve_mse(q1, q2, q2_new)
    else:
        q1_new = None
    return q2_new, q1_new
