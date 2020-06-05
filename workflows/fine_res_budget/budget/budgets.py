from typing import Optional
from dataclasses import dataclass

import numpy as np
import xarray as xr

from vcm import pressure_at_interface
from vcm.cubedsphere import block_upsample, weighted_block_average, regrid_vertical

import logging

logger = logging.getLogger(__name__)


def _center_to_interface(f: np.ndarray) -> np.ndarray:
    """Interpolate vertically cell centered data to the interface
    with linearly extrapolated inputs"""
    f_low = 2 * f[..., 0] - f[..., 1]
    f_high = 2 * f[..., -1] - f[..., -2]
    pad = np.concatenate([f_low[..., np.newaxis], f, f_high[..., np.newaxis]], axis=-1)
    return (pad[..., :-1] + pad[..., 1:]) / 2


def _convergence(eddy: np.ndarray, delp: np.ndarray) -> np.ndarray:
    """Compute vertical convergence of a cell-centered flux.
    
    This flux is assumed to vanish at the vertical boundaries
    """
    padded = _center_to_interface(eddy)
    # pad interfaces assuming eddy = 0 at edges
    return -np.diff(padded, axis=-1) / delp


def convergence(eddy: xr.DataArray, delp: xr.DataArray, dim: str = "p") -> xr.DataArray:
    return xr.apply_ufunc(
        _convergence,
        eddy,
        delp,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[eddy.dtype],
    )


@dataclass
class Grid:
    """Convenience interface for doing grid-math
    
    This class supplies the dimension name arguments to the cubedsphere methods,
    which are the same for all computations done here.
    
    """

    x: str
    y: str
    z: str
    xi: str
    yi: str
    zi: str

    def pressure_at_interface(self, p):
        return pressure_at_interface(p, dim_center=self.z, dim_outer=self.zi)

    def weighted_block_average(self, delp, weight, factor):
        return weighted_block_average(delp, weight, factor, x_dim=self.x, y_dim=self.y)

    def block_upsample(self, coarse, factor):
        return block_upsample(coarse, factor, [self.x, self.y])

    def regrid_vertical(self, *args, **kwargs):
        """regrid_vertical(
            p_in: xarray.core.dataarray.DataArray,
            f_in: xarray.core.dataarray.DataArray,
            p_out: xarray.core.dataarray.DataArray,
        """
        return regrid_vertical(
            *args, z_dim_center=self.z, z_dim_outer=self.zi, **kwargs
        )

    def pressure_level_average(self, delp_fine, delp_coarse, area, field, factor):
        pi = self.pressure_at_interface(delp_fine)
        pi_c = self.pressure_at_interface(delp_coarse)
        pi_c_up = self.block_upsample(pi_c, factor=factor)

        fg = self.regrid_vertical(pi, field, pi_c_up)
        avg = self.weighted_block_average(fg, area, factor)
        return avg.drop_vars([self.x, self.y, self.z], errors="ignore").rename(
            field.name
        )

    def vertical_convergence(self, f, delp):
        return convergence(f, delp, dim=self.z)


def storage(field: xr.DataArray, time_step: float) -> xr.DataArray:
    return (field.sel(step="end") - field.sel(step="begin")) / time_step


def eddy_flux_coarse(unresolved_flux, total_resolved_flux, omega, field):
    """Compute re-coarsened eddy flux divergence from re-coarsed data
    """
    return unresolved_flux + (total_resolved_flux - omega * field)


def compute_recoarsened_budget_field(
    area: xr.DataArray,
    delp_fine: xr.DataArray,
    delp_coarse: xr.DataArray,
    omega_fine: xr.DataArray,
    omega_coarse: xr.DataArray,
    field_fine: xr.DataArray,
    unresolved_flux: xr.DataArray,
    storage: xr.DataArray,
    saturation_adjustment: xr.DataArray,
    physics: xr.DataArray,
    nudging: xr.DataArray = None,
    factor: int = 8,
    field: Optional[str] = None,
):
    """Compute the recoarse-grained budget information


    Returns:

        xr.Dataset with keys: storage, eddy, field, resolved, convergence,
            microphysics, physics, fv_sat_adj
    Note:
        Need to pass in coarsened omega and delp to save computational cost

    """
    logger.info("Re-coarsegraining the budget")

    field = field_fine.name if field is None else field

    storage_name = "storage"
    unresolved_flux_name = "eddy"
    field_place_holder_name = "field"
    resolved_flux_name = "resolved"
    convergence_name = "convergence"

    grid = Grid("grid_xt", "grid_yt", "pfull", "grid_x", "grid_y", "pfulli")

    # Make iterator of all the variables to average
    def variables_to_average():
        yield saturation_adjustment.rename("saturation_adjustment")
        yield physics.rename("physics")
        if nudging is not None:
            yield nudging.rename("nudging")
        yield unresolved_flux.rename(unresolved_flux_name)
        yield (field_fine * omega_fine).rename(resolved_flux_name)
        yield storage.rename(storage_name)
        yield field_fine.rename(field_place_holder_name)

    def averaged_variables():
        for array in variables_to_average():
            yield grid.pressure_level_average(
                delp_fine, delp_coarse, area, array, factor=factor
            )

    averaged = xr.merge(averaged_variables())

    eddy_flux = eddy_flux_coarse(
        averaged[unresolved_flux_name],
        averaged[resolved_flux_name],
        omega_coarse,
        averaged[field_place_holder_name],
    )

    convergence = grid.vertical_convergence(eddy_flux, delp_coarse).rename(
        convergence_name
    )

    return xr.merge([convergence, averaged])


def add_budget_metadata(budget: xr.Dataset, units: str, field_name: str) -> xr.Dataset:
    tendency_units = units + "/s"
    budget.convergence.attrs.update(
        {"long_name": f"eddy flux convergence of {field_name}", "units": tendency_units}
    )

    budget.saturation_adjustment.attrs.update(
        {
            "long_name": (
                f"tendency of {field_name} due to dynamical core "
                "saturation adjustment"
            ),
            "units": tendency_units,
        }
    )

    budget.physics.attrs.update(
        {
            "long_name": f"tendency of {field_name} due to physics",
            "description": "sum of microphysics and any other parameterized process",
            "units": tendency_units,
        }
    )

    if "nudging" in budget:
        budget.nudging.attrs.update(
            {
                "long_name": f"tendency of {field_name} due to SHiELD nudging",
                "units": tendency_units,
            }
        )

    budget.storage.attrs.update(
        {
            "long_name": f"storage of {field_name}",
            "description": (
                f"partial time derivative of {field_name} for fixed x, y, "
                "and output model level. Sum of all the budget tendencies."
            ),
            "units": tendency_units,
        }
    )

    budget.field.attrs.update({"units": units})


def rename_recoarsened_budget(budget: xr.Dataset, field_name: str) -> str:
    rename = {}
    rename["field"] = field_name
    for variable in budget:
        if variable == "field":
            rename[variable] = field_name
        else:
            rename[variable] = field_name + "_" + variable

    return budget.rename(rename)


def compute_recoarsened_budget(merged: xr.Dataset, dt=15 * 60, factor=8):
    """Compute the recoarse-grained budgets of temperature and humidity

    Example output for a single tile::

        <xarray.Dataset>
        Dimensions:                         (grid_xt: 6, grid_yt: 6, pfull: 79)
        Coordinates:
            step                            object ...
            tile                            int64 ...
            time                            object ...
        Dimensions without coordinates: grid_xt, grid_yt, pfull
        Data variables:
            air_temperature                 (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            air_temperature_convergence     (grid_yt, grid_xt, pfull) float32 dask.array<chunksize=(6, 6, 79), meta=np.ndarray>
            air_temperature_eddy            (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            air_temperature_microphysics    (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            air_temperature_nudging         (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            air_temperature_physics         (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            air_temperature_resolved        (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            air_temperature_storage         (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            specific_humidity               (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            specific_humidity_convergence   (grid_yt, grid_xt, pfull) float32 dask.array<chunksize=(6, 6, 79), meta=np.ndarray>
            specific_humidity_eddy          (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            specific_humidity_microphysics  (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            specific_humidity_physics       (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            specific_humidity_resolved      (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            specific_humidity_storage       (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>

    """  # noqa

    logger.info("Re-coarsegraining the budget")

    grid = Grid("grid_xt", "grid_yt", "pfull", "grid_x", "grid_y", "pfulli")

    middle = merged.sel(step="middle")

    omega_fine = middle.omega_coarse
    area = middle.area_coarse
    delp_fine = middle.delp
    delp_coarse = grid.weighted_block_average(delp_fine, area, factor=factor)
    omega_coarse = grid.pressure_level_average(
        delp_fine, delp_coarse, area, omega_fine, factor=factor
    )

    t_budget_coarse = compute_recoarsened_budget_field(
        area,
        delp_fine,
        delp_coarse,
        omega_fine,
        omega_coarse,
        middle["T"],
        storage=storage(merged["T"], dt),
        unresolved_flux=middle["eddy_flux_omega_temp"],
        saturation_adjustment=middle["t_dt_gfdlmp_coarse"],
        nudging=middle["t_dt_nudge_coarse"],
        physics=middle["t_dt_phys_coarse"],
        factor=factor,
        field="air_temperature",
    )

    q_budget_coarse = compute_recoarsened_budget_field(
        area,
        delp_fine,
        delp_coarse,
        omega_fine,
        omega_coarse,
        middle["sphum"],
        storage=storage(merged["sphum"], dt),
        unresolved_flux=middle["eddy_flux_omega_sphum"],
        saturation_adjustment=middle["qv_dt_gfdlmp_coarse"],
        physics=middle["qv_dt_phys_coarse"],
        factor=factor,
    )

    # metadata adjustments
    add_budget_metadata(t_budget_coarse, "K", "air_temperature")
    t_budget_coarse = rename_recoarsened_budget(t_budget_coarse, "air_temperature")

    add_budget_metadata(q_budget_coarse, "kg/kg", "specific_humidity")
    q_budget_coarse = rename_recoarsened_budget(q_budget_coarse, "specific_humidity")

    omega_coarse = omega_coarse.assign_attrs(
        {"long_name": "Lagrangian derivative of hydrostatic pressure", "units": "Pa/s"}
    ).rename("omega")

    delp_coarse = delp_coarse.rename("delp")

    return xr.merge([t_budget_coarse, q_budget_coarse, omega_coarse, delp_coarse])
