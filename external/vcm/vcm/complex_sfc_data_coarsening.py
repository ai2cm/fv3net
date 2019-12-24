import dask.array as dask_array
import xarray as xr

from typing import Dict, Hashable

from . import xarray_utils
from .casting import doubles_to_floats
from .cubedsphere.coarsen import (
    block_coarsen,
    block_upsample,
    weighted_block_average,
)


FREEZING_TEMPERATURE = 273.16
SHDMIN_THRESHOLD = 0.011
STYPE_LAND_ICE = 16.0
VTYPE_LAND_ICE = 15.0
X_DIM = "xaxis_1"
Y_DIM = "yaxis_1"


def coarse_grain_sfc_data_complex(
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
    from .coarsen import sync_dimension_order

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
    return sync_dimension_order(doubles_to_floats(result), ds)


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

    upsampled_slmsk = _block_upsample_like(coarsened_slmsk, ds.slmsk)
    is_dominant_surface_type = xarray_utils.isclose(ds.slmsk, upsampled_slmsk)

    coarsened_vtype_and_stype = block_coarsen(
        ds[["vtype", "stype"]].where(is_dominant_surface_type),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
        method="mode",
        func_kwargs={"nan_policy": "omit"},
    )

    upsampled_vtype = _block_upsample_like(coarsened_vtype_and_stype.vtype, ds.vtype)
    is_dominant_vtype = xarray_utils.isclose(ds.vtype, upsampled_vtype)

    upsampled_stype = _block_upsample_like(coarsened_vtype_and_stype.stype, ds.stype)
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
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    area: xr.DataArray = None,
    **unused_kwargs,
) -> xr.DataArray:
    return weighted_block_average(
        data_var, area, coarsening_factor, x_dim=X_DIM, y_dim=Y_DIM
    )


def _area_weighted_mean_over_dominant_sfc_type(
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    area: xr.DataArray = None,
    is_dominant_surface_type: xr.DataArray = None,
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
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    area: xr.DataArray = None,
    is_dominant_surface_type: xr.DataArray = None,
    is_dominant_vtype: xr.DataArray = None,
    vfrac: xr.DataArray = None,
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
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    area: xr.DataArray = None,
    is_dominant_surface_type: xr.DataArray = None,
    is_dominant_stype: xr.DataArray = None,
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
    data_var: xr.DataArray = None, coarsening_factor: int = None, **unused_kwargs,
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
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    area: xr.DataArray = None,
    is_dominant_surface_type: xr.DataArray = None,
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
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    area: xr.DataArray = None,
    sncovr: xr.DataArray = None,
    **unused_kwargs,
) -> xr.DataArray:
    return weighted_block_average(
        data_var, area * sncovr, coarsening_factor, x_dim=X_DIM, y_dim=Y_DIM
    ).fillna(0.0)


def _area_and_fice_weighted_mean(
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    area: xr.DataArray = None,
    fice: xr.DataArray = None,
    **unused_kwargs,
) -> xr.DataArray:
    return weighted_block_average(
        data_var, area * fice, coarsening_factor, x_dim=X_DIM, y_dim=Y_DIM
    ).fillna(0.0)


def _minimum_over_dominant_sfc_type(
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    is_dominant_surface_type: xr.DataArray = None,
    **unused_kwargs,
) -> xr.DataArray:
    return block_coarsen(
        data_var.where(is_dominant_surface_type),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
        method="min",
    )


def _maximum_over_dominant_sfc_type(
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    is_dominant_surface_type: xr.DataArray = None,
    **unused_kwargs,
) -> xr.DataArray:
    return block_coarsen(
        data_var.where(is_dominant_surface_type),
        coarsening_factor,
        x_dim=X_DIM,
        y_dim=Y_DIM,
        method="max",
    )


def _area_or_area_and_fice_weighted_mean(
    data_var: xr.DataArray = None,
    coarsening_factor: int = None,
    area: xr.DataArray = None,
    is_dominant_surface_type: xr.DataArray = None,
    fice: xr.DataArray = None,
    coarsened_slmsk: xr.DataArray = None,
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
SFC_DATA_COARSENING_METHOD = {
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
    "shdmin": _minimum_over_dominant_sfc_type,
    "shdmax": _maximum_over_dominant_sfc_type,
    "snoalb": _maximum_over_dominant_sfc_type,
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

    See https://github.com/VulcanClimateModeling/fv3gfs/blob/
    master/sorc/global_chgres.fd/surface_chgres.f90 for more information.
    """
    ds = _clip_tsea_and_t3g_at_freezing_over_ice(ds)
    ds = _ensure_stype_is_ice_if_vtype_is_ice(ds)
    ds = _zero_canopy_moisture_content_over_bare_land(ds)
    return _zero_shdmin_over_land_ice(ds)


def _block_upsample_like(
    da: xr.DataArray,
    reference_da: xr.DataArray,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_1",
) -> xr.DataArray:
    """Upsample a DataArray and sync its chunk and coordinate properties with a
    reference DataArray.

    The purpose of this function is to upsample a coarsened DataArray back out
    to its original resolution.  In doing so it:
      - Ensures the chunk sizes of the upsampled DataArray match the chunk sizes of
        the original DataArray.  This is useful, because block_upsample often
        produces chunk sizes that are too small for good performance when
        applied to dask arrays.
      - Adds horizontal dimension coordinates that match the reference DataArray.

    As other block-related functions, this function assumes that the upsampling
    factor is the same in the x and y dimension.
    """
    upsampling_factor = reference_da.sizes[x_dim] // da.sizes[x_dim]
    result = block_upsample(da, upsampling_factor, [x_dim, y_dim])
    if isinstance(da.data, dask_array.Array):
        result = result.chunk(reference_da.chunks)
    return result.assign_coords(
        {x_dim: reference_da[x_dim], y_dim: reference_da[y_dim]}
    )
