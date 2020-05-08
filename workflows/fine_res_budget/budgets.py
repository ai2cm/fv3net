import os
from dataclasses import dataclass

import fsspec
import numpy as np
import xarray as xr

import vcm
from vcm.calc.thermo import pressure_at_interface
from vcm.cubedsphere.coarsen import (
    block_coarsen,
    block_upsample,
    block_upsample_like,
    weighted_block_average,
)

# TODO make a PR which exposes this function as public API
from vcm.cubedsphere.regridz import (
    _regrid_given_delp,
    block_upsample_like,
    pressure_at_interface,
    regrid_to_area_weighted_pressure,
    regrid_vertical,
)


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
        dask='parallelized',
        output_dtypes=[eddy.dtype]
    )


@dataclass
class Grid:
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

    def pressure_level_average(self, delp, delp_c, area, *args, factor):
        """
        
        Returns
            total flux, coarse-grained w, coarse-grained f, delpc
        
        """
        pi = self.pressure_at_interface(delp)
        pi_c = self.pressure_at_interface(delp_c)
        pi_c_up = self.block_upsample(pi_c, factor=factor)

        for arg in args:
            fg = self.regrid_vertical(pi, arg, pi_c_up)
            avg = self.weighted_block_average(fg, area, factor)
            yield avg.drop([self.x, self.y, self.z], errors="ignore")

    def vertical_divergence(self, f, delp):
        return divergence(f, delp, dim=self.z)


def dict_to_array(d, dim):
    return xr.concat(d.values(), dim=dim).assign_coords({dim: list(d.keys())})


def storage(qv, dt):
    return (qv.sel(step="end") - qv.sel(step="begin")) / dt


merged_schema = """
    <xarray.Dataset>
Dimensions:                  (grid_x: 385, grid_xt: 384, grid_y: 385, grid_yt: 384, nv: 2, pfull: 79, soil_layer: 4, step: 3, tile: 6)
Coordinates:
  * pfull                    (pfull) float64 4.514 8.301 12.45 ... 994.3 998.3
  * tile                     (tile) int64 0 1 2 3 4 5
    lat                      (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    lon                      (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
  * nv                       (nv) float32 1.0 2.0
    time                     object 2016-08-05 23:37:30
  * soil_layer               (soil_layer) float32 1.0 2.0 3.0 4.0
  * step                     (step) <U6 'begin' 'middle' 'end'
Dimensions without coordinates: grid_x, grid_xt, grid_y, grid_yt
Data variables:
    HGTsfc                   (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    area                     (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    average_DT               timedelta64[ns] dask.array<chunksize=(), meta=np.ndarray>
    average_T1               datetime64[ns] dask.array<chunksize=(), meta=np.ndarray>
    average_T2               datetime64[ns] dask.array<chunksize=(), meta=np.ndarray>
    delp                     (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    delp_dt_nudge            (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    dx                       (tile, grid_y, grid_xt) float32 dask.array<chunksize=(1, 385, 384), meta=np.ndarray>
    dy                       (tile, grid_yt, grid_x) float32 dask.array<chunksize=(1, 384, 385), meta=np.ndarray>
    eddy_flux_omega_ice_wat  (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    eddy_flux_omega_liq_wat  (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    eddy_flux_omega_sphum    (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    eddy_flux_omega_temp     (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    ice_wat_dt_gfdlmp        (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    ice_wat_dt_phys          (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    init_cond_eddy_qv_40     (grid_yt, grid_xt, tile) float32 dask.array<chunksize=(192, 192, 3), meta=np.ndarray>
    int_delp_dt_nudge        (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_ice_wat_dt_gfdlmp    (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_ice_wat_dt_phys      (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_liq_wat_dt_gfdlmp    (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_liq_wat_dt_phys      (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qg_dt_gfdlmp         (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qg_dt_phys           (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qi_dt_gfdlmp         (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qi_dt_phys           (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_ql_dt_gfdlmp         (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_ql_dt_phys           (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qr_dt_gfdlmp         (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qr_dt_phys           (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qs_dt_gfdlmp         (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qs_dt_phys           (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qv_dt_gfdlmp         (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_qv_dt_phys           (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_t_dt_gfdlmp          (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_t_dt_nudge           (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_t_dt_phys            (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_u_dt_gfdlmp          (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_u_dt_nudge           (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_u_dt_phys            (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_v_dt_gfdlmp          (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_v_dt_nudge           (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    int_v_dt_phys            (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    latb                     (tile, grid_y, grid_x) float32 dask.array<chunksize=(1, 385, 385), meta=np.ndarray>
    liq_wat_dt_gfdlmp        (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    liq_wat_dt_phys          (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    lonb                     (tile, grid_y, grid_x) float32 dask.array<chunksize=(1, 385, 385), meta=np.ndarray>
    omega                    (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    ps                       (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    ps_dt_nudge              (tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 384, 384), meta=np.ndarray>
    qg_dt_gfdlmp             (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    qg_dt_phys               (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    qi_dt_gfdlmp             (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    qi_dt_phys               (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    ql_dt_gfdlmp             (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    ql_dt_phys               (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    qr_dt_gfdlmp             (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    qr_dt_phys               (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    qs_dt_gfdlmp             (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    qs_dt_phys               (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    qv_dt_gfdlmp             (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    qv_dt_phys               (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    t_dt_gfdlmp              (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    t_dt_nudge               (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    t_dt_phys                (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    time_bnds                (nv) timedelta64[ns] dask.array<chunksize=(2,), meta=np.ndarray>
    u_dt_gfdlmp              (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    u_dt_nudge               (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    u_dt_phys                (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    v_dt_gfdlmp              (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    v_dt_nudge               (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    v_dt_phys                (tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 79, 128, 128), meta=np.ndarray>
    DZ                       (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    T                        (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    W                        (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    alnsf                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    alnwf                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    alvsf                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    alvwf                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    canopy                   (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    cld_amt                  (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    f10m                     (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    facsf                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    facwf                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    ffhh                     (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    ffmm                     (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    fice                     (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    graupel                  (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    hice                     (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    ice_wat                  (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    liq_wat                  (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    o3mr                     (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    phis                     (step, tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    q2m                      (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    rainwat                  (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    sgs_tke                  (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    shdmax                   (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    shdmin                   (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    sheleg                   (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    slc                      (step, tile, soil_layer, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 4, 384, 384), meta=np.ndarray>
    slmsk                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    slope                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    smc                      (step, tile, soil_layer, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 4, 384, 384), meta=np.ndarray>
    sncovr                   (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    snoalb                   (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    snowwat                  (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    snwdph                   (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    sphum                    (step, tile, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 384, 384), meta=np.ndarray>
    srflag                   (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    stc                      (step, tile, soil_layer, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 4, 384, 384), meta=np.ndarray>
    stype                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    t2m                      (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    tg3                      (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    tisfc                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    tprcp                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    tsea                     (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    u                        (step, tile, pfull, grid_y, grid_xt) float32 dask.array<chunksize=(1, 1, 79, 385, 384), meta=np.ndarray>
    u_srf                    (step, tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    uustar                   (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    v                        (step, tile, pfull, grid_yt, grid_x) float32 dask.array<chunksize=(1, 1, 79, 384, 385), meta=np.ndarray>
    v_srf                    (step, tile, grid_yt, grid_xt) float32 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    vfrac                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
    vtype                    (step, tile, grid_yt, grid_xt) float64 dask.array<chunksize=(1, 1, 384, 384), meta=np.ndarray>
"""


def compute_recoarsened_budget(merged: xr.Dataset, dt=15 * 60, factor=8):
    f"""Compute the recoarse-grained budget information

    merged needs to be in the following format:

    {merged_schema}
    """

    grid = Grid("grid_xt", "grid_yt", "pfull", "grid_x", "grid_y", "pfulli")
    VARIABLES = ["t_dt_gfdlmp", "t_dt_nudge", "t_dt_phys", "qv_dt_gfdlmp", "qv_dt_phys"]

    middle = merged.sel(step="middle")

    area = middle.area
    delp = middle.delp
    delp_c = grid.weighted_block_average(middle.delp, middle.area, factor=factor)

    # Collect all variables
    variables_to_average = {}
    for key in VARIABLES:
        variables_to_average[key] = middle[key]

    variables_to_average["storage_T"] = storage(merged.T, dt=dt)
    variables_to_average["storage_q"] = storage(merged.sphum, dt=dt)

    variables_to_average["omega"] = middle.omega
    variables_to_average["sphum"] = middle.sphum
    variables_to_average["T"] = middle.T
    variables_to_average["wq"] = middle.sphum * middle.omega
    variables_to_average["wT"] = middle.sphum * middle.T
    variables_to_average["eddy_q"] = middle.eddy_flux_omega_sphum
    variables_to_average["eddy_T"] = middle.eddy_flux_omega_temp

    def avg_dict(d, delp, delp_c, area, factor):
        return dict(
            zip(
                d.keys(),
                grid.pressure_level_average(
                    delp, delp_c, area, *d.values(), factor=factor
                ),
            )
        )

    averaged_vars = avg_dict(variables_to_average, delp, delp_c, area, factor=factor)
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

    div_q = grid.vertical_divergence(eddy_flux_q, delp_c.drop("pfull"))
    div_T = grid.vertical_divergence(eddy_flux_t, delp_c.drop("pfull"))

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

   