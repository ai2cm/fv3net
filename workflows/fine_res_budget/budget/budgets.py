from dataclasses import dataclass

import numpy as np
import xarray as xr

from vcm import pressure_at_interface
from vcm.cubedsphere import block_upsample, weighted_block_average, regrid_vertical


def _divergence(eddy, delp):
    """Eddy is cell centered here"""
    padding = [(0, 0)] * eddy.ndim
    padding[-1] = (1, 1)
    padded = np.pad(eddy, pad_width=padding)
    eddy_interface = (padded[..., 1:] + padded[..., :-1]) / 2
    return -np.diff(eddy_interface, axis=-1) / delp


def divergence(eddy, delp, dim="p"):
    return xr.apply_ufunc(
        _divergence,
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

    def pressure_level_average(self, delp, delp_c, area, arg, factor):
        """
        
        Returns
            total flux, coarse-grained w, coarse-grained f, delpc
        
        """
        pi = self.pressure_at_interface(delp)
        pi_c = self.pressure_at_interface(delp_c)
        pi_c_up = self.block_upsample(pi_c, factor=factor)

        fg = self.regrid_vertical(pi, arg, pi_c_up)
        avg = self.weighted_block_average(fg, area, factor)
        return avg.drop([self.x, self.y, self.z], errors="ignore")

    def vertical_divergence(self, f, delp):
        return divergence(f, delp, dim=self.z)


def dict_to_array(d, dim):
    return xr.concat(d.values(), dim=dim).assign_coords({dim: list(d.keys())})


def storage(qv, dt):
    return (qv.sel(step="end") - qv.sel(step="begin")) / dt


def compute_recoarsened_budget(merged: xr.Dataset, dt=15 * 60, factor=8):
    """Compute the recoarse-grained budget information

    merged needs to be in the following format:

    """

    grid = Grid("grid_xt", "grid_yt", "pfull", "grid_x", "grid_y", "pfulli")
    VARIABLES = ["t_dt_gfdlmp", "t_dt_nudge", "t_dt_phys", "qv_dt_gfdlmp", "qv_dt_phys"]

    middle = merged.sel(step="middle")

    omega = middle.omega
    area = middle.area
    delp = middle.delp
    delp_c = grid.weighted_block_average(middle.delp, middle.area, factor=factor)

    # Collect all variables
    variables_to_average = {}
    for key in VARIABLES:
        variables_to_average[key] = middle[key]

    variables_to_average["storage_T"] = storage(merged.T, dt=dt)
    variables_to_average["storage_q"] = storage(merged.sphum, dt=dt)

    variables_to_average["omega"] = omega
    variables_to_average["sphum"] = middle.sphum
    variables_to_average["T"] = middle.T
    variables_to_average["wq"] = middle.sphum * omega
    variables_to_average["wT"] = middle.T * omega
    variables_to_average["eddy_q"] = middle.eddy_flux_omega_sphum
    variables_to_average["eddy_T"] = middle.eddy_flux_omega_temp

    averaged_vars = {}
    for key in variables_to_average:
        averaged_vars[key] = grid.pressure_level_average(
            delp, delp_c, area, variables_to_average[key], factor=8
        )

    eddy_flux_q = (
        averaged_vars["eddy_q"]
        + averaged_vars["wq"]
        - averaged_vars["omega"] * averaged_vars["sphum"]
    )
    eddy_flux_t = (
        averaged_vars["eddy_T"]
        + averaged_vars["wT"]
        - averaged_vars["omega"] * averaged_vars["T"]
    )

    div_q = grid.vertical_divergence(eddy_flux_q, delp_c)
    div_T = grid.vertical_divergence(eddy_flux_t, delp_c)

    t_budget_coarse = {
        "storage": averaged_vars["storage_T"],
        "microphysics": averaged_vars["t_dt_gfdlmp"],
        "nudging": averaged_vars["t_dt_nudge"],
        "physics": averaged_vars["t_dt_phys"],
        "vertical_eddy_fluxdiv": div_T,
    }

    q_budget_coarse = {
        "storage": averaged_vars["storage_q"],
        "microphysics": averaged_vars["qv_dt_gfdlmp"],
        "physics": averaged_vars["qv_dt_phys"],
        "vertical_eddy_fluxdiv": div_q,
    }

    return xr.Dataset(
        {
            "air_temperature": averaged_vars["T"],
            "specific_humidity": averaged_vars["sphum"],
            "omega": averaged_vars["omega"],
            "air_temperature_tendency": dict_to_array(
                t_budget_coarse, "budget"
            ).assign_attrs(units="K/s"),
            "specific_humidity_tendency": dict_to_array(
                q_budget_coarse, "budget"
            ).assign_attrs(units="kg/kg/s"),
            "delp": delp_c,
        }
    )
