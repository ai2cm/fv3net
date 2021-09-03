"""
Transforms operate on diagnostic function inputs to adjust data before
diagnostic values are calculated.

A transform should take in the transform-specific arguments with a diagnostic
function argument tuple as the final argument and return the adjusted
diagnostic function arguments.
"""

import logging
from typing import Sequence, Tuple
import numpy as np
import xarray as xr

from vcm import interpolate_to_pressure_levels

HORIZONTAL_DIMS = ["x", "y", "tile"]

# argument typehint by multiple modules split out to group operations and simplify
# the diagnostic script
DiagArg = Tuple[xr.Dataset, xr.Dataset, xr.Dataset]


_TRANSFORM_FNS = {}

logger = logging.getLogger(__name__)

SURFACE_TYPE_CODES = {"sea": (0, 2), "land": (1,), "seaice": (2,)}
PRESSURE_DIM = "pressure"
VERTICAL_DIM = "z"
AREA_VAR = "area"
DELP_VAR = "pressure_thickness_of_atmospheric_layer"


def add_to_input_transform_fns(func):

    _TRANSFORM_FNS[func.__name__] = func

    return func


def apply(transform_key: str, *transform_args_partial, **transform_kwargs):
    """
    Wrapper to apply transform to input diagnostic arguments (tuple of three datasets).
    Transform arguments are specified per diagnostic function to enable a query-style
    operation on input data.

    apply -> wraps diagnostic function in save_prognostic_run_diags and
    returns a new function with an input transform prepended to the diagnostic call.
    
    I.e., call to diagnostic_function becomes::

        input_transform(*diag_args):
            adjusted_args = transform(*diagargs)
            diagnostic_function(*adjusted_args)

    Args:
        transform_key: name of transform function to call
        transform_args_partial: All transform function specific arguments preceding the
            final diagnostic argument tuple, e.g., [freq_label] for resample_time
        transform_kwargs: Any transform function keyword arguments
    
    Note: I tried memoizing the current transforms but am unsure
    if it will work on highly mutable datasets.
    """

    def _apply_to_diag_func(diag_func):

        if transform_key not in _TRANSFORM_FNS:
            raise KeyError(
                f"Unrecognized transform, {transform_key} requested "
                f"for {diag_func.__name__}"
            )

        transform_func = _TRANSFORM_FNS[transform_key]

        def transform(*diag_args):

            logger.debug(
                f"Adding transform, {transform_key}, "
                f"to diagnostic function: {diag_func.__name__}"
                f"\n\targs: {transform_args_partial}"
                f"\n\tkwargs: {transform_kwargs}"
            )

            # append diagnostic function input to be transformed
            transform_args = (*transform_args_partial, diag_args)

            transformed_diag_args = transform_func(*transform_args, **transform_kwargs)

            return diag_func(*transformed_diag_args)

        return transform

    return _apply_to_diag_func


def _mask_vars_with_horiz_dims(ds, surface_type, latitude, land_sea_mask):
    """
    Subset data to variables with specified dimensions before masking
    to prevent odd behavior from variables with non-compliant dims
    (e.g., interfaces)
    """

    spatial_ds_varnames = [
        var_name
        for var_name in ds.data_vars
        if set(HORIZONTAL_DIMS).issubset(set(ds[var_name].dims))
    ]
    masked = xr.Dataset()
    for var in spatial_ds_varnames:
        masked[var] = _mask_array(surface_type, ds[var], latitude, land_sea_mask)

    non_spatial_varnames = list(set(ds.data_vars) - set(spatial_ds_varnames))

    return masked.update(ds[non_spatial_varnames])


def _is_3d(da: xr.DataArray, vertical_dim: str = "z"):
    return vertical_dim in da.dims


@add_to_input_transform_fns
def regrid_zdim_to_pressure_levels(arg: DiagArg) -> DiagArg:
    prediction, target, grid, delp = arg
    prediction_regridded, target_regridded = xr.Dataset(), xr.Dataset()
    vertical_prediction_fields = [var for var in prediction if _is_3d(prediction[var])]
    for var in vertical_prediction_fields:
        prediction_regridded[var] = interpolate_to_pressure_levels(
            delp=delp, field=prediction[var], dim=VERTICAL_DIM,
        )
        target_regridded[var] = interpolate_to_pressure_levels(
            delp=delp, field=target[var], dim=VERTICAL_DIM,
        )
    return prediction_regridded, target_regridded, grid, delp


@add_to_input_transform_fns
def mask_to_sfc_type(surface_type: str, arg: DiagArg) -> DiagArg:
    """
    Mask prognostic run and verification data to the specified surface type

    Args:
        arg: input arguments to transform prior to the diagnostic calculation
        surface_type:  Type of grid locations to leave unmasked
    """
    prediction, target, grid, delp = arg
    masked_prediction = _mask_vars_with_horiz_dims(
        prediction, surface_type, grid.lat, grid.land_sea_mask
    )

    masked_target = _mask_vars_with_horiz_dims(
        target, surface_type, grid.lat, grid.land_sea_mask
    )

    return masked_prediction, masked_target, grid, delp


@add_to_input_transform_fns
def mask_area(region: str, arg: DiagArg) -> DiagArg:
    """
    Set area variable to zero everywhere outside of specified region.

    Args:
        region: name of region to leave unmasked. Valid options are "global",
            "land", "sea", and "tropics".
        arg: input arguments to transform prior to the diagnostic calculation
    """
    prediction, target, grid, delp = arg

    masked_area = _mask_array(region, grid.area, grid.lat, grid.land_sea_mask)

    grid_copy = grid.copy()
    return prediction, target, grid_copy.update({"area": masked_area})


def _mask_array(
    region: str, arr: xr.DataArray, latitude: xr.DataArray, land_sea_mask: xr.DataArray,
) -> xr.DataArray:
    """Mask given DataArray to a specific region."""
    if region == "tropics":
        masked_arr = arr.where(abs(latitude) <= 10.0)
    elif region == "global":
        masked_arr = arr.copy()
    elif region in SURFACE_TYPE_CODES:
        masks = [land_sea_mask == code for code in SURFACE_TYPE_CODES[region]]
        mask_union = masks[0]
        for mask in masks[1:]:
            mask_union = np.logical_or(mask_union, mask)
        masked_arr = arr.where(mask_union)
    else:
        raise ValueError(f"Masking procedure for region '{region}' is not defined.")
    return masked_arr


@add_to_input_transform_fns
def subset_variables(variables: Sequence, arg: DiagArg) -> DiagArg:
    """Subset the variables, without failing if a variable doesn't exist"""
    prediction, target, grid, delp = arg
    prediction_vars = [var for var in variables if var in prediction]
    target_vars = [var for var in variables if var in target]
    return prediction[prediction_vars], target[target_vars], grid, delp


@add_to_input_transform_fns
def select_3d_variables(arg: DiagArg) -> DiagArg:
    prediction, target, grid, delp = arg
    prediction_vars = [var for var in prediction if _is_3d(prediction[var])]
    return prediction[prediction_vars], target[prediction_vars], grid, delp


@add_to_input_transform_fns
def select_2d_variables(arg: DiagArg) -> DiagArg:
    prediction, target, grid, delp = arg
    prediction_vars = [var for var in prediction if not _is_3d(prediction[var])]
    return prediction[prediction_vars], target[prediction_vars], grid, delp
