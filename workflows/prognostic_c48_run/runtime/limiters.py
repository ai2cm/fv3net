from typing import MutableMapping, Hashable, Tuple
import xarray as xr

State = MutableMapping[Hashable, xr.DataArray]

GRAVITY = 9.81
M_PER_MM = 1 / 1000


def couple_sphum_tendency_and_surface_precip(
    physics_precip: xr.DataArray,
    sphum_tendency: xr.DataArray,
    delp: xr.DataArray,
    dt: float,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Add column-integrated drying of the atmosphere by nudging/ML to physics
    precipitation, assuming the resulting quantity is positive. If it is negative,
    scale down column-wise specific humidity tendencies until this quantity is
    zero, such that moisture is conserved.

    Args:
        physics_precip: 2-D precipitation from physics parameterizations [m]
        sphum_tendency: vertically-resolved specific humidity tendencies [kg/kg/s]
        delp: pressure thicknesses of atmospheric model levels [Pa]
        dt: physics timestep [s]

    Returns:
        sum of physics precipitation and nudging/ML-induced precipitation
        updated specific humidity tendencies
        """

    column_dq2 = (sphum_tendency * delp / GRAVITY).sum("z")
    python_precip = -column_dq2 * dt * M_PER_MM  # type: ignore
    total_precip_init = physics_precip + python_precip
    sphum_tendency_updated = xr.where(
        total_precip_init >= 0,
        sphum_tendency,
        -(physics_precip / python_precip) * sphum_tendency,  # type: ignore
    )
    total_precip = total_precip_init.where(total_precip_init >= 0, 0)
    total_precip.attrs["units"] = "m"
    return total_precip, sphum_tendency_updated


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
