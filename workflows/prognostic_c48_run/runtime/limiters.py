from typing import MutableMapping, Hashable
import xarray as xr

State = MutableMapping[Hashable, xr.DataArray]

gravity = 9.81
m_per_mm = 1 / 1000


def precipitation_sum(
    physics_precip: xr.DataArray, column_dq2: xr.DataArray, dt: float
) -> xr.DataArray:
    """Return sum of physics precipitation and ML-induced precipitation. Output is
    thresholded to enforce positive precipitation.

    Args:
        physics_precip: precipitation from physics parameterizations [m]
        column_dq2: column-integrated moistening from ML [kg/m^2/s]
        dt: physics timestep [s]

    Returns:
        total precipitation [m]"""
    python_precip = -column_dq2 * dt * m_per_mm  # type: ignore
    total_precip = physics_precip + python_precip
    total_precip = total_precip.where(total_precip >= 0, 0)
    total_precip.attrs["units"] = "m"
    return total_precip


def limit_sphum_tendency_for_non_negativity(
    sphum_state: xr.DataArray, sphum_tendency: xr.DataArray, dt: float
) -> xr.DataArray:
    """Reduce ML-derived sphum tendencies that would otherwise produce negative
    specific humidity"""

    delta = sphum_tendency * dt
    sphum_tendency_updated = xr.where(
        sphum_state + delta > 0, sphum_tendency, -sphum_state / dt,  # type: ignore
    )
    return sphum_tendency_updated


def limit_sphum_tendency_for_moisture_conservation(
    total_precip: xr.DataArray,
    sphum_tendency: xr.DataArray,
    delp: xr.DataArray,
    dt: float,
) -> xr.DataArray:
    """Set to zero columns of ML- or nudging-derived sphum tendencies that would
    otherwise produce an atmospheric moisture source without corresponding surface
    evaporation, i.e., PRATE-<dQ2> < 0"""

    column_dq2 = (sphum_tendency * delp / gravity).sum("z")
    python_precip = -column_dq2 * dt * m_per_mm
    total_precip = total_precip + python_precip
    sphum_tendency_updated = xr.where(
        total_precip >= 0,  # type: ignore
        sphum_tendency,
        xr.zeros_like(sphum_tendency),
    )
    return sphum_tendency_updated
