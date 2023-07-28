"""
Utilities for coarse-graining restart data and directories
"""
import logging
from typing import Dict, Hashable, Mapping, Callable

import dask
import numpy as np
import xarray as xr

from .. import xarray_utils
from ..calc.thermo.vertically_dependent import (
    dz_and_top_to_phis,
    height_at_interface,
    hydrostatic_dz,
    pressure_at_midpoint_log,
    surface_pressure_from_delp,
)
from .coarsen import (
    block_coarsen,
    block_edge_coarsen,
    block_edge_sum,
    block_median,
    block_upsample_like,
    edge_weighted_block_average,
    weighted_block_average,
)
from .constants import (
    COORD_X_CENTER,
    COORD_X_OUTER,
    COORD_Y_CENTER,
    COORD_Y_OUTER,
    FV_CORE_X_CENTER,
    FV_CORE_X_OUTER,
    FV_CORE_Y_CENTER,
    FV_CORE_Y_OUTER,
    FV_SRF_WND_X_CENTER,
    FV_SRF_WND_Y_CENTER,
    FV_TRACER_X_CENTER,
    FV_TRACER_Y_CENTER,
    RESTART_Z_CENTER,
    SFC_DATA_X_CENTER,
    SFC_DATA_Y_CENTER,
)
from .regridz import regrid_to_area_weighted_pressure, regrid_to_edge_weighted_pressure
from .xgcm import create_fv3_grid

FREEZING_TEMPERATURE = 273.16
SHDMIN_THRESHOLD = 0.011
STYPE_LAND_ICE = 16.0
VTYPE_LAND_ICE = 15.0
X_DIM = "xaxis_1"
Y_DIM = "yaxis_1"
SIGMA_BLEND = 0.9
FRACTION_TRACERS = ["cld_amt"]  # Always are area-weighted.
NON_FRACTION_TRACERS = [
    "sphum",
    "liq_wat",
    "rainwat",
    "ice_wat",
    "snowwat",
    "graupel",
    "o3mr",
    "sgs_tke",
]  # Are area-weighted or mass-weighted depending on the user's choice.


dask.config.set(scheduler="single-threaded")

TILES = range(1, 7)
CATEGORY_LIST = ["fv_core.res", "fv_srf_wnd.res", "fv_tracer.res", "sfc_data"]

logger = logging.getLogger("vcm.coarsen")


def coarsen_restarts_on_sigma(
    coarsening_factor: int,
    grid_spec: xr.Dataset,
    restarts: Mapping[str, xr.Dataset],
    coarsen_agrid_winds: bool = False,
    mass_weighted: bool = True,
) -> Mapping[str, xr.Dataset]:
    """Coarsen a complete set of restart data, averaging on model levels and
    using the 'complex' surface coarsening method

    Args:
        coarsening_factor: the amount of coarsening to apply. C384 to C48 is a factor
            of 8.
        grid_spec: Dataset containing the variables area, dx, dy.
        restarts: dictionary of restart data. Must have the keys
            "fv_core.res", "fv_srf_wnd.res", "fv_tracer.res", and "sfc_data".
        coarsen_agrid_winds: flag indicating whether to coarsen A-grid winds in
            "fv_core.res" restart files (default False).

    Returns:
        restarts_coarse: a dictionary with the same format as restarts but
            coarsening_factor times coarser.

    """
    coarsened = {}

    coarsened["fv_core.res"] = _coarse_grain_fv_core(
        restarts["fv_core.res"],
        restarts["fv_core.res"].delp,
        grid_spec.area.rename(
            {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
        ),
        grid_spec.dx.rename(
            {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_OUTER: FV_CORE_Y_OUTER}
        ),
        grid_spec.dy.rename(
            {COORD_X_OUTER: FV_CORE_X_OUTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
        ),
        coarsening_factor,
        coarsen_agrid_winds,
        mass_weighted,
    )

    coarsened["fv_srf_wnd.res"] = _coarse_grain_fv_srf_wnd(
        restarts["fv_srf_wnd.res"],
        grid_spec.area.rename(
            {COORD_X_CENTER: FV_SRF_WND_X_CENTER, COORD_Y_CENTER: FV_SRF_WND_Y_CENTER}
        ),
        coarsening_factor,
    )

    coarsened["fv_tracer.res"] = _coarse_grain_fv_tracer(
        restarts["fv_tracer.res"],
        restarts["fv_core.res"].delp.rename({FV_CORE_Y_CENTER: FV_TRACER_Y_CENTER}),
        grid_spec.area.rename(
            {COORD_X_CENTER: FV_TRACER_X_CENTER, COORD_Y_CENTER: FV_TRACER_Y_CENTER}
        ),
        coarsening_factor,
        mass_weighted,
    )

    coarsened["sfc_data"] = _coarse_grain_sfc_data_complex(
        restarts["sfc_data"],
        grid_spec.area.rename(
            {COORD_X_CENTER: SFC_DATA_X_CENTER, COORD_Y_CENTER: SFC_DATA_Y_CENTER}
        ),
        coarsening_factor,
    )

    for category in CATEGORY_LIST:
        _sync_dimension_order(coarsened[category], restarts[category])

    return coarsened


def coarsen_restarts_on_pressure(
    coarsening_factor: int,
    grid_spec: xr.Dataset,
    toa_pressure: float,
    restarts: Mapping[str, xr.Dataset],
    coarsen_agrid_winds: bool = False,
    extrapolate: bool = False,
) -> Mapping[str, xr.Dataset]:
    """Coarsen a complete set of restart files, averaging on pressure levels and
    using the 'complex' surface coarsening method

    Args:
        coarsening_factor: the amount of coarsening to apply. C384 to C48 is a factor
            of 8.
        grid_spec: Dataset containing the variables area, dx, dy.
        toa_pressure: pressure at the top of the atmosphere in units of Pascals.
        restarts: dictionary of restart data. Must have the keys
            "fv_core.res", "fv_srf_wnd.res", "fv_tracer.res", and "sfc_data".
        coarsen_agrid_winds: flag indicating whether to coarsen A-grid winds in
            "fv_core.res" restart files (default False).
        extrapolate (optional): whether to allow for limited nearest-neighbor
            extrapolation at points in fine-grid columns whose surface pressure
            is at least greater than the coarse layer midpoint's pressure.
            Otherwise do not allow any nearest-neighbor extrapolation (the
            setting by default).

    Returns:
        restarts_coarse: a dictionary with the same format as restarts but
            coarsening_factor times coarser.
    """

    coarsened = {}

    coarsened["fv_core.res"] = _coarse_grain_fv_core_on_pressure(
        restarts["fv_core.res"],
        restarts["fv_core.res"].delp,
        grid_spec.area.rename(
            {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
        ),
        grid_spec.dx.rename(
            {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_OUTER: FV_CORE_Y_OUTER}
        ),
        grid_spec.dy.rename(
            {COORD_X_OUTER: FV_CORE_X_OUTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
        ),
        toa_pressure,
        coarsening_factor,
        coarsen_agrid_winds,
        extrapolate=extrapolate,
    )

    coarsened["fv_srf_wnd.res"] = _coarse_grain_fv_srf_wnd(
        restarts["fv_srf_wnd.res"],
        grid_spec.area.rename(
            {COORD_X_CENTER: FV_SRF_WND_X_CENTER, COORD_Y_CENTER: FV_SRF_WND_Y_CENTER}
        ),
        coarsening_factor,
    )

    coarsened["fv_tracer.res"] = _coarse_grain_fv_tracer_on_pressure(
        restarts["fv_tracer.res"],
        restarts["fv_core.res"].delp.rename({FV_CORE_Y_CENTER: FV_TRACER_Y_CENTER}),
        grid_spec.area.rename(
            {COORD_X_CENTER: FV_TRACER_X_CENTER, COORD_Y_CENTER: FV_TRACER_Y_CENTER}
        ),
        toa_pressure,
        coarsening_factor,
        extrapolate=extrapolate,
    )

    coarsened["sfc_data"] = _coarse_grain_sfc_data_complex(
        restarts["sfc_data"],
        grid_spec.area.rename(
            {COORD_X_CENTER: SFC_DATA_X_CENTER, COORD_Y_CENTER: SFC_DATA_Y_CENTER}
        ),
        coarsening_factor,
    )

    coarsened["fv_core.res"] = _impose_hydrostatic_balance(
        coarsened["fv_core.res"], coarsened["fv_tracer.res"], toa_pressure
    )

    for category in CATEGORY_LIST:
        _sync_dimension_order(coarsened[category], restarts[category])

    return coarsened


def coarsen_restarts_via_blended_method(
    coarsening_factor: int,
    grid_spec: xr.Dataset,
    toa_pressure: float,
    restarts: Mapping[str, xr.Dataset],
    coarsen_agrid_winds: bool = False,
    mass_weighted: bool = True,
) -> Mapping[str, xr.Dataset]:
    """Coarsen a complete set of restart files using the blended pressure-level
    / model-level coarse-graining method for 3D fields and the 'complex' surface
    coarsening method for surface data.

    Args:
        coarsening_factor: the amount of coarsening to apply. C384 to C48 is a
            factor of 8.
        grid_spec: Dataset containing the variables area, dx, dy.
        toa_pressure: pressure at the top of the atmosphere in units of Pascals.
        restarts: dictionary of restart data. Must have the keys
            "fv_core.res", "fv_srf_wnd.res", "fv_tracer.res", and "sfc_data".
        coarsen_agrid_winds: flag indicating whether to coarsen A-grid winds in
            "fv_core.res" restart files (default False).
        mass_weighted: flag indicating whether model-level averages are
            mass-weighted for the vertical velocity, temperature, and mixing
            ratio tracers and area-weighted for all other 3D variables, or
            area-weighted for all 3D variables.

    Returns:
        restarts_coarse: a dictionary with the same format as restarts but
            coarsening_factor times coarser.
    """

    coarsened = {}

    coarsened["fv_core.res"] = _coarse_grain_fv_core_via_blended_method(
        restarts["fv_core.res"],
        restarts["fv_core.res"].delp,
        grid_spec.area.rename(
            {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
        ),
        grid_spec.dx.rename(
            {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_OUTER: FV_CORE_Y_OUTER}
        ),
        grid_spec.dy.rename(
            {COORD_X_OUTER: FV_CORE_X_OUTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
        ),
        toa_pressure,
        coarsening_factor,
        coarsen_agrid_winds,
        mass_weighted,
    )

    coarsened["fv_srf_wnd.res"] = _coarse_grain_fv_srf_wnd(
        restarts["fv_srf_wnd.res"],
        grid_spec.area.rename(
            {COORD_X_CENTER: FV_SRF_WND_X_CENTER, COORD_Y_CENTER: FV_SRF_WND_Y_CENTER}
        ),
        coarsening_factor,
    )

    coarsened["fv_tracer.res"] = _coarse_grain_fv_tracer_via_blended_method(
        restarts["fv_tracer.res"],
        restarts["fv_core.res"].delp.rename({FV_CORE_Y_CENTER: FV_TRACER_Y_CENTER}),
        grid_spec.area.rename(
            {COORD_X_CENTER: FV_TRACER_X_CENTER, COORD_Y_CENTER: FV_TRACER_Y_CENTER}
        ),
        toa_pressure,
        coarsening_factor,
        mass_weighted,
    )

    coarsened["sfc_data"] = _coarse_grain_sfc_data_complex(
        restarts["sfc_data"],
        grid_spec.area.rename(
            {COORD_X_CENTER: SFC_DATA_X_CENTER, COORD_Y_CENTER: SFC_DATA_Y_CENTER}
        ),
        coarsening_factor,
    )

    coarsened["fv_core.res"] = _impose_hydrostatic_balance(
        coarsened["fv_core.res"], coarsened["fv_tracer.res"], toa_pressure
    )

    for category in CATEGORY_LIST:
        _sync_dimension_order(coarsened[category], restarts[category])

    return coarsened


def _integerize(x):
    return np.round(x).astype(x.dtype)


def _coarse_grain_fv_core(
    ds,
    delp,
    area,
    dx,
    dy,
    coarsening_factor,
    coarsen_agrid_winds=False,
    mass_weighted=True,
):
    """Coarse grain a set of fv_core restart files on model levels.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_core restart files
    delp : xr.DataArray
        Pressure thicknesses
    area : xr.DataArray
        Area weights
    dx : xr.DataArray
        x edge lengths
    dy : xr.DataArray
        y edge lengths
    coarsening_factor : int
        Coarsening factor to use
    coarsen_agrid_winds : bool
        Whether to coarse-grain A-grid winds (default False)
    mass_weighted : bool
        Whether to weight temperature and vertical velocity using mass instead
        of area (default True)

    Returns
    -------
    xr.Dataset
    """
    if mass_weighted:
        area_weighted_vars = ["phis", "delp", "DZ"]
        mass_weighted_vars = ["W", "T"]
    else:
        area_weighted_vars = ["phis", "delp", "DZ", "W", "T"]
        mass_weighted_vars = []

    if coarsen_agrid_winds:
        if not ("ua" in ds and "va" in ds):
            raise ValueError(
                "If 'coarsen_agrid_winds' is active, 'ua' and 'va' "
                "must be present in the 'fv_core.res' restart files."
            )
        if mass_weighted:
            mass_weighted_vars.extend(["ua", "va"])
        else:
            area_weighted_vars.extend(["ua", "va"])

    dx_edge_weighted_vars = ["u"]
    dy_edge_weighted_vars = ["v"]

    area_weighted = weighted_block_average(
        ds[area_weighted_vars],
        area,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )

    mass = delp * area
    mass_weighted = weighted_block_average(
        ds[mass_weighted_vars],
        mass,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )

    edge_weighted_x = edge_weighted_block_average(
        ds[dx_edge_weighted_vars],
        dx,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_OUTER,
        edge="x",
    )

    edge_weighted_y = edge_weighted_block_average(
        ds[dy_edge_weighted_vars],
        dy,
        coarsening_factor,
        x_dim=FV_CORE_X_OUTER,
        y_dim=FV_CORE_Y_CENTER,
        edge="y",
    )

    return xr.merge([area_weighted, mass_weighted, edge_weighted_x, edge_weighted_y])


def _coarse_grain_fv_core_on_pressure(
    ds,
    delp,
    area,
    dx,
    dy,
    toa_pressure,
    coarsening_factor,
    coarsen_agrid_winds=False,
    extrapolate=False,
):
    """Coarse grain a set of fv_core restart files, averaging on surfaces of
    constant pressure (except for delp, DZ and phis which are averaged on model
    surfaces).

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_core restart files
    delp : xr.DataArray
        Pressure thicknesses
    area : xr.DataArray
        Area weights
    dx : xr.DataArray
        x edge lengths
    dy : xr.DataArray
        y edge lengths
    toa_pressure : float
        Pressure at the top of the atmosphere in units of Pascals.
    coarsening_factor : int
        Coarsening factor to use
    coarsen_agrid_winds : bool
        Whether to coarse-grain A-grid winds (default False)
    extrapolate : bool
        Whether to allow for limited nearest-neighbor extrapolation at points in
        fine-grid columns whose surface pressure is at least greater than the
        coarse layer midpoint's pressure.  Otherwise do not allow any
        nearest-neighbor extrapolation (the setting by default).

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ["phis", "delp", "DZ"]
    masked_area_weighted_vars = ["W", "T"]
    if coarsen_agrid_winds:
        if not ("ua" in ds and "va" in ds):
            raise ValueError(
                "If 'coarsen_agrid_winds' is active, 'ua' and 'va' "
                "must be present in the 'fv_core.res' restart files."
            )
        masked_area_weighted_vars.extend(["ua", "va"])
    dx_edge_weighted_vars = ["u"]
    dy_edge_weighted_vars = ["v"]

    area_pressure_regridded, masked_area = regrid_to_area_weighted_pressure(
        ds[masked_area_weighted_vars],
        delp,
        area,
        toa_pressure,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
        extrapolate=extrapolate,
    )

    dx_pressure_regridded, masked_dx = regrid_to_edge_weighted_pressure(
        ds[dx_edge_weighted_vars],
        delp,
        dx,
        toa_pressure,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_OUTER,
        edge="x",
        extrapolate=extrapolate,
    )

    dy_pressure_regridded, masked_dy = regrid_to_edge_weighted_pressure(
        ds[dy_edge_weighted_vars],
        delp,
        dy,
        toa_pressure,
        coarsening_factor,
        x_dim=FV_CORE_X_OUTER,
        y_dim=FV_CORE_Y_CENTER,
        edge="y",
        extrapolate=extrapolate,
    )

    area_weighted = weighted_block_average(
        ds[area_weighted_vars],
        area,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )

    masked_area_weighted = weighted_block_average(
        area_pressure_regridded,
        masked_area,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )

    edge_weighted_x = edge_weighted_block_average(
        dx_pressure_regridded,
        masked_dx,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_OUTER,
        edge="x",
    )

    edge_weighted_y = edge_weighted_block_average(
        dy_pressure_regridded,
        masked_dy,
        coarsening_factor,
        x_dim=FV_CORE_X_OUTER,
        y_dim=FV_CORE_Y_CENTER,
        edge="y",
    )

    return xr.merge(
        [area_weighted, masked_area_weighted, edge_weighted_x, edge_weighted_y]
    )


def compute_blending_weights(blending_pressure, ps_coarse, pfull_coarse):
    """Compute the weights for blending pressure level and model level coarsened
    3D fields.

    Args:
        blending_pressure: xr.DataArray
            Pressure above which we start blending in the model level coarsened
            field.
        ps_coarse: xr.DataArray
            Coarse surface pressure
        pfull_coarse:
            Coarse pressure at vertical level midpoints

    Returns:
        blending_weights: xr.DataArray
    """
    blending_weights = (ps_coarse - pfull_coarse) / (ps_coarse - blending_pressure)
    return xr.where(pfull_coarse > blending_pressure, blending_weights, 1.0)


def _compute_blending_weights_agrid(
    delp,
    area,
    toa_pressure,
    coarsening_factor,
    x_dim=FV_CORE_X_CENTER,
    y_dim=FV_CORE_Y_CENTER,
):
    """Compute the blending weights on the A-grid.

    This follows the approach Chris describes in Section 7 of `this document
    <https://drive.google.com/file/d/1FyLTnR1C5_Ab5Tdbbuhtxm52VC-NroJG/view>`_.
    Here the blending pressure is computed to be 0.9 times the value of the
    minimum surface pressure of the columns contained in a coarse grid cell, and
    the blending weights are then given by the following equation:

    .. math::
        b_k^c = \begin{cases}
            1 & p_k^c \leq p_b^c \\
            \frac{p_s^c - p_k^c}{p_s^c - p_b^c} & p_k^c > p_b^c
        \end{cases}

    Here :math:`b_k^c` is the blending weight at a level :math:`k`,
    :math:`p_k^c` is the pressure at the midpoint of level :math:`k` on the
    coarse grid, :math:`p_b^c` is the blending pressure, and :math:`p_s^c` is
    the surface pressure on the coarse grid.
    """  # noqa: W605
    delp_coarse = weighted_block_average(
        delp, area, coarsening_factor, x_dim=x_dim, y_dim=y_dim
    )
    pfull_coarse = pressure_at_midpoint_log(
        delp_coarse, toa_pressure=toa_pressure, dim=RESTART_Z_CENTER
    )
    ps = surface_pressure_from_delp(
        delp, p_toa=toa_pressure, vertical_dim=RESTART_Z_CENTER
    )
    ps_coarse = surface_pressure_from_delp(
        delp_coarse, p_toa=toa_pressure, vertical_dim=RESTART_Z_CENTER
    )
    blending_pressure = SIGMA_BLEND * block_coarsen(
        ps, coarsening_factor, x_dim=x_dim, y_dim=y_dim, method="min"
    )
    return compute_blending_weights(blending_pressure, ps_coarse, pfull_coarse)


def _compute_blending_weights_dgrid(
    delp, length, toa_pressure, coarsening_factor, edge, x_dim, y_dim
):
    """This follows the approach Chris describes in Section 7 of `this document
    <https://drive.google.com/file/d/1FyLTnR1C5_Ab5Tdbbuhtxm52VC-NroJG/view>`_,
    adapted to grid cell edges rather than grid cell centers. Here the blending
    pressure is computed to be 0.9 times the value of the minimum surface
    pressure along the edge of a coarse grid cell, and the blending weights are
    then given by the following equation:

    .. math::
        b_k^c = \begin{cases}
            1 & p_k^c \leq p_b^c \\
            \frac{p_s^c - p_k^c}{p_s^c - p_b^c} & p_k^c > p_b^c
        \end{cases}

    Here :math:`b_k^c` is the blending weight at a level :math:`k`,
    :math:`p_k^c` is the pressure at the midpoint of level :math:`k` on the
    coarse grid edge, :math:`p_b^c` is the blending pressure, and :math:`p_s^c`
    is the surface pressure on the coarse grid edge.
    """  # noqa: W605
    delp_edge = compute_edge_delp(delp, edge, x_dim=x_dim, y_dim=y_dim)
    delp_edge_coarse = edge_weighted_block_average(
        delp_edge, length, coarsening_factor, x_dim=x_dim, y_dim=y_dim, edge=edge
    )
    pfull_coarse = pressure_at_midpoint_log(
        delp_edge_coarse, toa_pressure=toa_pressure, dim=RESTART_Z_CENTER
    )
    ps = surface_pressure_from_delp(
        delp_edge, p_toa=toa_pressure, vertical_dim=RESTART_Z_CENTER
    )
    ps_coarse = surface_pressure_from_delp(
        delp_edge_coarse, p_toa=toa_pressure, vertical_dim=RESTART_Z_CENTER
    )
    blending_pressure = SIGMA_BLEND * block_edge_coarsen(
        ps, coarsening_factor, edge=edge, x_dim=x_dim, y_dim=y_dim, method="min"
    )
    return compute_blending_weights(blending_pressure, ps_coarse, pfull_coarse)


def blend(weights, pressure_level, model_level):
    """Blend two coarse-grained Datasets or DataArrays using the provided weights.

    Args:
        weights: xr.DataArray
            Weights used to blend the two sources together.
        pressure_level: xr.DataArray or xr.Dataset
            Pressure-level coarsened data.
        model_level: xr.DataArray or xr.Dataset
            Model-level coarsened data.

    Returns:
        xr.DataArray or xr.Dataset
            The blended result.
    """
    return weights * pressure_level + (1 - weights) * model_level


def _2d_fv_core_names_agrid(ds):
    return [v for v in ds.data_vars if RESTART_Z_CENTER not in ds[v].dims]


def _3d_fv_core_names_agrid(ds, coarsen_agrid_winds):
    names = []
    ignore_vars = ["u", "v"]
    if not coarsen_agrid_winds:
        ignore_vars.extend(["ua", "va"])
    for v, da in ds.data_vars.items():
        if v not in ignore_vars and RESTART_Z_CENTER in da.dims:
            names.append(v)
    return names


def _coarse_grain_fv_core_via_blended_method(
    ds,
    delp,
    area,
    dx,
    dy,
    toa_pressure,
    coarsening_factor,
    coarsen_agrid_winds=False,
    mass_weighted=True,
):
    """Coarse grain a set of fv_core restart files via the blended pressure-level
    / model-level approach.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_core restart files
    delp : xr.DataArray
        Pressure thicknesses
    area : xr.DataArray
        Area weights
    dx : xr.DataArray
        x edge lengths
    dy : xr.DataArray
        y edge lengths
    toa_pressure : float
        Pressure at the top of the atmosphere in units of Pascals.
    coarsening_factor : int
        Coarsening factor to use
    coarsen_agrid_winds : bool
        Whether to coarse-grain A-grid winds (default False)
    mass_weighted : bool
        Whether to weight temperature and vertical velocity using mass instead
        of area during the model-level coarsening process (default True)

    Returns
    -------
    xr.Dataset
    """
    pressure_level = _coarse_grain_fv_core_on_pressure(
        ds, delp, area, dx, dy, toa_pressure, coarsening_factor, coarsen_agrid_winds,
    )
    model_level = _coarse_grain_fv_core(
        ds, delp, area, dx, dy, coarsening_factor, coarsen_agrid_winds, mass_weighted
    )
    weights_agrid = _compute_blending_weights_agrid(
        delp,
        area,
        toa_pressure,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )
    weights_dgrid_u = _compute_blending_weights_dgrid(
        delp,
        dx,
        toa_pressure,
        coarsening_factor,
        "x",
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_OUTER,
    )
    weights_dgrid_v = _compute_blending_weights_dgrid(
        delp,
        dy,
        toa_pressure,
        coarsening_factor,
        "y",
        x_dim=FV_CORE_X_OUTER,
        y_dim=FV_CORE_Y_CENTER,
    )
    names_2d_agrid = _2d_fv_core_names_agrid(ds)
    names_3d_agrid = _3d_fv_core_names_agrid(ds, coarsen_agrid_winds)

    # 2D fields could come from either the pressure level or model level results
    fields_2d_agrid = model_level[names_2d_agrid]
    fields_3d_agrid = blend(
        weights_agrid, pressure_level[names_3d_agrid], model_level[names_3d_agrid]
    )
    u = blend(weights_dgrid_u, pressure_level.u, model_level.u).rename("u")
    v = blend(weights_dgrid_v, pressure_level.v, model_level.v).rename("v")
    return xr.merge([fields_2d_agrid, fields_3d_agrid, u, v])


def _coarse_grain_fv_tracer_via_blended_method(
    ds, delp, area, toa_pressure, coarsening_factor, mass_weighted=True
):
    """Coarse grain a set of fv_tracer restart files via the blended
    pressure-level / model-level approach.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_tracer restart files
    delp : xr.DataArray
        Pressure thicknesses
    area : xr.DataArray
        Area weights
    toa_pressure : float
        Pressure at the top of the atmosphere in units of Pascals.
    coarsening_factor : int
        Coarsening factor to use
    mass_weighted: bool
        Whether to weight mixing ratios and TKE by mass during the model-level
        coarsening process.

    Returns
    -------
    xr.Dataset
    """
    pressure_level = _coarse_grain_fv_tracer_on_pressure(
        ds, delp, area, toa_pressure, coarsening_factor
    )
    model_level = _coarse_grain_fv_tracer(
        ds, delp, area, coarsening_factor, mass_weighted
    )
    weights = _compute_blending_weights_agrid(
        delp,
        area,
        toa_pressure,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )
    return blend(weights, pressure_level, model_level)


def compute_edge_delp(delp, edge, x_dim=FV_CORE_X_CENTER, y_dim=FV_CORE_Y_CENTER):
    """Compute the pressure thickness on grid cell edges

    Args:
        delp: xr.DataArray
            Pressure thickness at grid cell centers
        edge: str
            Grid cell side to coarse-grain along {"x", "y"}
        x_dim: str (optional)
            Name of x dimension
        y_dim: str (optional)
            Name of y dimension
    """
    hor_dims = {"x": x_dim, "y": y_dim}
    grid = create_fv3_grid(
        xr.Dataset({"delp": delp}),
        x_center=FV_CORE_X_CENTER,
        x_outer=FV_CORE_X_OUTER,
        y_center=FV_CORE_Y_CENTER,
        y_outer=FV_CORE_Y_OUTER,
    )
    interp_dim = "x" if edge == "y" else "y"
    return grid.interp(delp, interp_dim).assign_coords(
        {
            hor_dims[interp_dim]: np.arange(
                1, delp.sizes[hor_dims[edge]] + 2, dtype=np.float32
            )
        }
    )


def _coarse_grain_fv_tracer(ds, delp, area, coarsening_factor, mass_weighted=True):
    """Coarse grain a set of fv_tracer restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_tracer restart files
    delp : xr.DataArray
        Pressure thicknesses
    area : xr.DataArray
        Area weights
    coarsening_factor : int
        Coarsening factor to use
    mass_weighted: bool
        Whether to weight mixing ratios and TKE by mass

    Returns
    -------
    xr.Dataset
    """
    if mass_weighted:
        area_weighted_vars = FRACTION_TRACERS
        mass_weighted_vars = NON_FRACTION_TRACERS
    else:
        area_weighted_vars = FRACTION_TRACERS + NON_FRACTION_TRACERS
        mass_weighted_vars = []

    area_weighted = weighted_block_average(
        ds[area_weighted_vars],
        area,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )

    mass = delp * area
    mass_weighted = weighted_block_average(
        ds[mass_weighted_vars],
        mass,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )

    return xr.merge([area_weighted, mass_weighted])


def _coarse_grain_fv_tracer_on_pressure(
    ds, delp, area, toa_pressure, coarsening_factor, extrapolate=False
):
    """Coarse grain a set of fv_tracer restart files, averaging on surfaces of
    constant pressure.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_tracer restart files
    delp : xr.DataArray
        Pressure thicknesses
    area : xr.DataArray
        Area weights
    toa_pressure : float
        Pressure at the top of the atmosphere in units of Pascals.
    coarsening_factor : int
        Coarsening factor to use
    extrapolate : bool
        Whether to allow for limited nearest-neighbor extrapolation at points in
        fine-grid columns whose surface pressure is at least greater than the
        coarse layer midpoint's pressure.  Otherwise do not allow any
        nearest-neighbor extrapolation (the setting by default).

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = FRACTION_TRACERS
    masked_area_weighted_vars = NON_FRACTION_TRACERS

    ds_regridded, masked_area = regrid_to_area_weighted_pressure(
        ds,
        delp,
        area,
        toa_pressure,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
        extrapolate=extrapolate,
    )

    area_weighted = weighted_block_average(
        ds_regridded[area_weighted_vars],
        masked_area,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )

    masked_area_weighted = weighted_block_average(
        ds_regridded[masked_area_weighted_vars],
        masked_area,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )

    return xr.merge([area_weighted, masked_area_weighted])


def _coarse_grain_fv_srf_wnd(ds, area, coarsening_factor):
    """Coarse grain a set of fv_srf_wnd restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_srf_wnd restart files
    area : xr.DataArray
        Area weights
    coarsening_factor : int
        Coarsening factor to use

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ["u_srf", "v_srf"]
    return weighted_block_average(
        ds[area_weighted_vars],
        area,
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    )


def _impose_hydrostatic_balance(
    ds_fv_core, ds_fv_tracer, toa_pressure, dim=RESTART_Z_CENTER
):
    """Compute layer thicknesses assuming hydrostatic balance and adjust
    surface geopotential in order to maintain same model top height.

    Args:
        ds_fv_core (xr.Dataset): fv_core restart category Dataset
        ds_fv_tracer (xr.Dataset): fv_tracer restart category Dataset
        toa_pressure (float): pressure at the top of the atmosphere in units
            of Pascals
        dim (str): vertical dimension name (default "zaxis_1")

    Returns:
        xr.Dataset: ds_fv_core with hydrostatic DZ and adjusted phis
    """
    height = height_at_interface(
        ds_fv_core["DZ"], ds_fv_core["phis"], dim_center=dim, dim_outer=dim
    )
    height_top = height.isel({dim: 0})
    dz = hydrostatic_dz(
        ds_fv_core["T"],
        ds_fv_tracer["sphum"].rename({FV_TRACER_Y_CENTER: FV_CORE_Y_CENTER}),
        ds_fv_core["delp"],
        toa_pressure,
        dim=dim,
    )
    return ds_fv_core.assign(DZ=dz, phis=dz_and_top_to_phis(height_top, dz, dim=dim))


def _coarse_grain_sfc_data(ds, area, coarsening_factor, version="simple"):
    """Coarse grain a set of sfc_data restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of sfc_data restart files
    area : xr.DataArray
        Area weights
    coarsening_factor : int
        Coarsening factor to use
    version : str
        Version of the method to use {'simple', 'complex'}

    Returns
    -------
    xr.Dataset
    """
    if version == "simple":
        result = block_median(ds, coarsening_factor, x_dim="xaxis_1", y_dim="yaxis_1")
        result["slmsk"] = _integerize(result.slmsk)
        return result
    elif version == "complex":
        return _coarse_grain_sfc_data_complex(ds, area, coarsening_factor)
    else:
        raise ValueError(
            f"Currently the only supported versions are 'simple' and 'complex'. "
            "Got {version}."
        )


def _coarse_grain_grid_spec(
    ds,
    coarsening_factor,
    x_dim_unstaggered=COORD_X_CENTER,
    y_dim_unstaggered=COORD_Y_CENTER,
    x_dim_staggered=COORD_X_OUTER,
    y_dim_staggered=COORD_Y_OUTER,
):
    coarse_dx = block_edge_sum(
        ds.dx, coarsening_factor, x_dim_unstaggered, y_dim_staggered, "x"
    )
    coarse_dy = block_edge_sum(
        ds.dy, coarsening_factor, x_dim_staggered, y_dim_unstaggered, "y"
    )
    coarse_area = block_coarsen(
        ds.area, coarsening_factor, x_dim_unstaggered, y_dim_unstaggered
    )

    return xr.merge(
        [
            coarse_dx.rename(ds.dx.name),
            coarse_dy.rename(ds.dy.name),
            coarse_area.rename(ds.area.name),
        ]
    )


def _sync_dimension_order(a, b):
    for var in a:
        a[var] = a[var].transpose(*b[var].dims)
    return a


def _is_float(x):
    floating_point_types = tuple(
        np.dtype(t) for t in ["float32", "float64", "float128"]
    )
    if x.dtype in floating_point_types:
        return True
    else:
        return False


def _doubles_to_floats(ds: xr.Dataset):
    coords = {}
    data_vars = {}

    for key in ds.coords:
        coord = ds[key]
        if _is_float(coord):
            coords[key] = ds[key].astype(np.float32)

    for key in ds.data_vars:
        var = ds[key]
        if _is_float(var):
            data_vars[key] = ds[key].astype(np.float32).drop_vars(var.coords)

    return xr.Dataset(data_vars, coords=coords)


def _coarse_grain_sfc_data_complex(
    ds: xr.Dataset, area: xr.DataArray, coarsening_factor: int
) -> xr.Dataset:
    """Coarse grain a set of sfc_data restart files using the 'complicated'
    method.

    See:

    https://paper.dropbox.com/doc/Downsampling-restart-fields-for-the-
    Noah-land-surface-model--Ap8z8JnU1OH4HwNZ~PY4z~p~Ag-OgkIEYSn0g4X9CCZxn5dy

    for a description of what this procedure does.

    Args:
        ds: Input Dataset.
        area: DataArray with the surface area of each grid cell; it is assumed
            its horizontal dimension names are 'xaxis_1' and 'yaxis_1'.
        coarsening_factor: Integer coarsening factor to use.

    Returns:
        xr.Dataset
    """
    precomputed_arguments = _compute_arguments_for_complex_sfc_coarsening(
        ds, coarsening_factor
    )

    result = xr.Dataset()
    result["slmsk"] = precomputed_arguments["coarsened_slmsk"]
    result["vtype"] = precomputed_arguments["coarsened_vtype"]
    result["stype"] = precomputed_arguments["coarsened_stype"]
    for data_var in ds:
        if data_var not in result:
            result[data_var] = SFC_DATA_COARSENING_METHOD[data_var](
                data_var=ds[data_var],
                coarsening_factor=coarsening_factor,
                area=area,
                is_dominant_surface_type=precomputed_arguments[
                    "is_dominant_surface_type"
                ],
                is_dominant_vtype=precomputed_arguments["is_dominant_vtype"],
                is_dominant_stype=precomputed_arguments["is_dominant_stype"],
                vfrac=ds.vfrac,
                sncovr=ds.sncovr,
                fice=ds.fice,
                coarsened_slmsk=precomputed_arguments["coarsened_slmsk"],
            )

    result = _apply_surface_chgres_corrections(result)
    return _doubles_to_floats(result)


def _compute_arguments_for_complex_sfc_coarsening(
    ds: xr.Dataset, coarsening_factor: int
) -> Dict[str, xr.DataArray]:
    coarsened_slmsk = block_coarsen(
        ds.slmsk,
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
        method="mode",
        func_kwargs={"nan_policy": "omit"},
    )

    upsampled_slmsk = block_upsample_like(coarsened_slmsk, ds.slmsk)
    is_dominant_surface_type = xarray_utils.isclose(ds.slmsk, upsampled_slmsk)

    coarsened_vtype_and_stype = block_coarsen(
        ds[["vtype", "stype"]].where(is_dominant_surface_type),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
        method="mode",
        func_kwargs={"nan_policy": "omit"},
    )

    upsampled_vtype = block_upsample_like(coarsened_vtype_and_stype.vtype, ds.vtype)
    is_dominant_vtype = xarray_utils.isclose(ds.vtype, upsampled_vtype)

    upsampled_stype = block_upsample_like(coarsened_vtype_and_stype.stype, ds.stype)
    is_dominant_stype = xarray_utils.isclose(ds.stype, upsampled_stype)

    return {
        "coarsened_slmsk": coarsened_slmsk,
        "coarsened_vtype": coarsened_vtype_and_stype.vtype,
        "coarsened_stype": coarsened_vtype_and_stype.stype,
        "is_dominant_surface_type": is_dominant_surface_type,
        "is_dominant_vtype": is_dominant_vtype,
        "is_dominant_stype": is_dominant_stype,
    }


def _area_weighted_mean(
    data_var: xr.DataArray, coarsening_factor: int, area: xr.DataArray, **unused_kwargs,
) -> xr.DataArray:
    return weighted_block_average(
        data_var, area, coarsening_factor, x_dim=X_DIM, y_dim=Y_DIM
    )


def _area_weighted_mean_over_dominant_sfc_type(
    data_var: xr.DataArray,
    coarsening_factor: int,
    area: xr.DataArray,
    is_dominant_surface_type: xr.DataArray,
    **unused_kwargs,
) -> xr.DataArray:
    return weighted_block_average(
        data_var.where(is_dominant_surface_type),
        area.where(is_dominant_surface_type),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
    )


def _area_and_vfrac_weighted_mean_over_dominant_sfc_and_vtype(
    data_var: xr.DataArray,
    coarsening_factor: int,
    area: xr.DataArray,
    is_dominant_surface_type: xr.DataArray,
    is_dominant_vtype: xr.DataArray,
    vfrac: xr.DataArray,
    **unused_kwargs,
) -> xr.DataArray:
    mask = is_dominant_surface_type & is_dominant_vtype
    area_weighted_mean = weighted_block_average(
        data_var.where(mask),
        area.where(mask),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
    )
    area_and_vfrac_weighted_mean = weighted_block_average(
        data_var.where(mask),
        (area * vfrac).where(mask),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
    )
    coarsened_area_times_vfrac = block_coarsen(
        (area * vfrac).where(mask),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
        method="sum",
    )
    return xr.where(
        coarsened_area_times_vfrac > 0.0,
        area_and_vfrac_weighted_mean,
        area_weighted_mean,
    )


def _area_weighted_mean_over_dominant_sfc_and_stype(
    data_var: xr.DataArray,
    coarsening_factor: int,
    area: xr.DataArray,
    is_dominant_surface_type: xr.DataArray,
    is_dominant_stype: xr.DataArray,
    **unused_kwargs,
) -> xr.DataArray:
    mask = is_dominant_surface_type & is_dominant_stype
    return weighted_block_average(
        data_var.where(mask),
        area.where(mask),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
    )


def _mode(
    data_var: xr.DataArray, coarsening_factor: int, **unused_kwargs,
) -> xr.DataArray:
    return block_coarsen(
        data_var,
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
        method="mode",
        func_kwargs={"nan_policy": "omit"},
    )


def _mode_over_dominant_sfc_type(
    data_var: xr.DataArray,
    coarsening_factor: int,
    area: xr.DataArray,
    is_dominant_surface_type: xr.DataArray,
    **unused_kwargs,
) -> xr.DataArray:
    return block_coarsen(
        data_var.where(is_dominant_surface_type),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
        method="mode",
        func_kwargs={"nan_policy": "omit"},
    )


def _area_and_sncovr_weighted_mean(
    data_var: xr.DataArray,
    coarsening_factor: int,
    area: xr.DataArray,
    sncovr: xr.DataArray,
    **unused_kwargs,
) -> xr.DataArray:
    return weighted_block_average(
        data_var, area * sncovr, coarsening_factor, x_dim=X_DIM, y_dim=Y_DIM
    ).fillna(0.0)


def _area_and_fice_weighted_mean(
    data_var: xr.DataArray,
    coarsening_factor: int,
    area: xr.DataArray,
    fice: xr.DataArray,
    **unused_kwargs,
) -> xr.DataArray:
    return weighted_block_average(
        data_var, area * fice, coarsening_factor, x_dim=X_DIM, y_dim=Y_DIM
    ).fillna(0.0)


def _area_or_area_and_fice_weighted_mean(
    data_var: xr.DataArray,
    coarsening_factor: int,
    area: xr.DataArray,
    is_dominant_surface_type: xr.DataArray,
    fice: xr.DataArray,
    coarsened_slmsk: xr.DataArray,
    **unused_kwargs,
) -> xr.DataArray:
    """Special function for handling tisfc.

    Takes the area and ice fraction weighted mean over sea ice and the area
    weighted mean over anything else."""
    sea_ice = weighted_block_average(
        data_var.where(is_dominant_surface_type),
        (area * fice).where(is_dominant_surface_type),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
    )
    land_or_ocean = weighted_block_average(
        data_var.where(is_dominant_surface_type),
        area.where(is_dominant_surface_type),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
    )
    return xr.where(xarray_utils.isclose(coarsened_slmsk, 2.0), sea_ice, land_or_ocean)


# Note that slmsk, vtype, and stype are handled separately (they need to be
# coarsened ahead of coarsening these other variables).
SFC_DATA_COARSENING_METHOD: Mapping[Hashable, Callable] = {
    "tsea": _area_weighted_mean,
    "alvsf": _area_weighted_mean,
    "alvwf": _area_weighted_mean,
    "alnsf": _area_weighted_mean,
    "alnwf": _area_weighted_mean,
    "facsf": _area_weighted_mean,
    "facwf": _area_weighted_mean,
    "f10m": _area_weighted_mean,
    "t2m": _area_weighted_mean,
    "q2m": _area_weighted_mean,
    "uustar": _area_weighted_mean,
    "ffmm": _area_weighted_mean,
    "ffhh": _area_weighted_mean,
    "tprcp": _area_weighted_mean,
    "snwdph": _area_weighted_mean,
    "tg3": _area_weighted_mean_over_dominant_sfc_type,
    "vfrac": _area_weighted_mean_over_dominant_sfc_type,
    "fice": _area_weighted_mean_over_dominant_sfc_type,
    "sncovr": _area_weighted_mean_over_dominant_sfc_type,
    "canopy": _area_and_vfrac_weighted_mean_over_dominant_sfc_and_vtype,
    "zorl": _area_and_vfrac_weighted_mean_over_dominant_sfc_and_vtype,
    "smc": _area_weighted_mean_over_dominant_sfc_and_stype,
    "slc": _area_weighted_mean_over_dominant_sfc_and_stype,
    "stc": _area_weighted_mean_over_dominant_sfc_and_stype,
    "srflag": _mode,
    "slope": _mode_over_dominant_sfc_type,
    "sheleg": _area_and_sncovr_weighted_mean,
    "hice": _area_and_fice_weighted_mean,
    "shdmin": _area_weighted_mean_over_dominant_sfc_type,
    "shdmax": _area_weighted_mean_over_dominant_sfc_type,
    "snoalb": _area_weighted_mean_over_dominant_sfc_type,
    "tisfc": _area_or_area_and_fice_weighted_mean,
}


def _clip_tsea_and_t3g_at_freezing_over_ice(ds: xr.Dataset) -> xr.Dataset:
    """Step (1) of surface_chgres corrections.

    Clip tsea and tg3 at 273.16 K if a cell contains land ice.
    """
    clipped_tsea = ds.tsea.where(
        ds.tsea < FREEZING_TEMPERATURE, other=FREEZING_TEMPERATURE
    )
    clipped_t3g = ds.tg3.where(
        ds.tg3 < FREEZING_TEMPERATURE, other=FREEZING_TEMPERATURE
    )

    is_land_ice = xarray_utils.isclose(ds.vtype, VTYPE_LAND_ICE)

    tsea = xr.where(is_land_ice, clipped_tsea, ds.tsea)
    tg3 = xr.where(is_land_ice, clipped_t3g, ds.tg3)

    ds["tsea"] = tsea
    ds["tg3"] = tg3

    return ds


def _ensure_stype_is_ice_if_vtype_is_ice(ds: xr.Dataset) -> xr.Dataset:
    """Step (2) of surface_chgres corrections.

    If a cell contains land ice, make sure the soil type is ice.
    """
    is_land_ice = xarray_utils.isclose(ds.vtype, VTYPE_LAND_ICE)
    stype = xr.where(is_land_ice, STYPE_LAND_ICE, ds.stype)
    ds["stype"] = stype
    return ds


def _zero_canopy_moisture_content_over_bare_land(ds: xr.Dataset) -> xr.Dataset:
    """Step (3) of surface_chgres corrections.

    If a cell does not contain vegetation, i.e. if shdmin < 0.011,
    then set the canopy moisture content to zero.
    """
    canopy = xr.where(ds.shdmin < SHDMIN_THRESHOLD, 0.0, ds.canopy)
    ds["canopy"] = canopy
    return ds


def _zero_shdmin_over_land_ice(ds: xr.Dataset) -> xr.Dataset:
    """Step (4) of surface_chgres corrections.

    If a cell contains land ice, then shdmin is set to zero.
    """
    is_land_ice = xarray_utils.isclose(ds.vtype, VTYPE_LAND_ICE)
    shdmin = xr.where(is_land_ice, 0.0, ds.shdmin)
    ds["shdmin"] = shdmin
    return ds


def _apply_surface_chgres_corrections(ds: xr.Dataset) -> xr.Dataset:
    """This function applies the corrections noted in surface_chgres.F90
    to the variables in the coarsened surface Dataset.

    See https://github.com/ai2cm/fv3gfs/blob/
    master/sorc/global_chgres.fd/surface_chgres.f90 for more information.
    """
    ds = _clip_tsea_and_t3g_at_freezing_over_ice(ds)
    ds = _ensure_stype_is_ice_if_vtype_is_ice(ds)
    ds = _zero_canopy_moisture_content_over_bare_land(ds)
    return _zero_shdmin_over_land_ice(ds)
