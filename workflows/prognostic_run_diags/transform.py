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
from datetime import timedelta

from constants import HORIZONTAL_DIMS, DiagArg

_TRANSFORM_FNS = {}

logger = logging.getLogger(__name__)

SURFACE_TYPE_CODES = {"sea": (0, 2), "land": (1,), "seaice": (2,)}


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


@add_to_input_transform_fns
def resample_time(
    freq_label: str,
    arg: DiagArg,
    time_slice: slice = slice(None, -1),
    inner_join: bool = False,
    split_timedelta: timedelta = None,
    second_freq_label: str = "1D",
) -> DiagArg:
    """
    Subset times in prognostic and verification data.

    Args:
        arg: input arguments to transform prior to the diagnostic calculation
        freq_label: Time resampling frequency label (should be valid input for xarray's
            resampling function)
        time_slice: Index slice to reduce times after frequency resampling.  Omits final
            time by default to work with crashed simulations.
        inner_join: Subset times to the intersection of prognostic and verification
            data. Defaults to False.
        split_timedelta: time since start of prognostic run after which times will be
            resampled with second_freq_label. Defaults to None, in which case
            the entire run is resampled at the same frequency.
        second_freq_label: time resampling frequency label for portion of run
            after split_timedelta. Defaults to "1D".
    """
    prognostic, verification, grid = arg
    if split_timedelta is None:
        prognostic = prognostic.resample(time=freq_label, label="right").mean()
    else:
        split_time = prognostic.time.values[0] + split_timedelta
        first_segment = prognostic.sel(time=slice(None, split_time))
        second_segment = prognostic.sel(time=slice(split_time, None))
        resampled = [first_segment.resample(time=freq_label, label="right").mean()]
        if second_segment.sizes["time"] != 0:
            resampled.append(
                second_segment.resample(time=second_freq_label, label="right").mean()
            )
        prognostic = xr.concat(resampled, dim="time")

    prognostic = prognostic.isel(time=time_slice)
    if inner_join:
        prognostic, verification = _inner_join_time(prognostic, verification)
    return prognostic, verification, grid


def _inner_join_time(
    prognostic: xr.Dataset, verification: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset]:
    """ Subset times within the prognostic data to be within the verification data,
    as necessary and vice versa, and return the subset datasets
    """

    inner_join_time = xr.merge(
        [
            prognostic.time.rename("prognostic_time"),
            verification.time.rename("verification_time"),
        ],
        join="inner",
    )

    return (
        prognostic.sel(time=inner_join_time.prognostic_time),
        verification.sel(time=inner_join_time.verification_time),
    )


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


@add_to_input_transform_fns
def mask_to_sfc_type(surface_type: str, arg: DiagArg) -> DiagArg:
    """
    Mask prognostic run and verification data to the specified surface type

    Args:
        arg: input arguments to transform prior to the diagnostic calculation
        surface_type:  Type of grid locations to leave unmasked
    """
    prognostic, verification, grid = arg
    masked_prognostic = _mask_vars_with_horiz_dims(
        prognostic, surface_type, grid.lat, grid.land_sea_mask
    )

    masked_verification = _mask_vars_with_horiz_dims(
        verification, surface_type, grid.lat, grid.land_sea_mask
    )

    return masked_prognostic, masked_verification, grid


@add_to_input_transform_fns
def mask_area(region: str, arg: DiagArg) -> DiagArg:
    """
    Set area variable to zero everywhere outside of specified region.

    Args:
        region: name of region to leave unmasked. Valid options are "global",
            "land", "sea", and "tropics".
        arg: input arguments to transform prior to the diagnostic calculation
    """
    prognostic, verification, grid = arg

    masked_area = _mask_array(region, grid.area, grid.lat, grid.land_sea_mask)

    grid_copy = grid.copy()
    return prognostic, verification, grid_copy.update({"area": masked_area})


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
        if len(masks) == 2:
            mask = np.logical_or(masks[0], masks[1])
        else:
            mask = masks[0]
        masked_arr = arr.where(mask)
    else:
        raise ValueError(f"Masking procedure for region '{region}' is not defined.")
    return masked_arr


@add_to_input_transform_fns
def subset_variables(variables: Sequence, arg: DiagArg) -> DiagArg:
    """Subset the variables, without failing if a variable doesn't exist"""
    prognostic, verification, grid = arg
    prognostic_vars = [var for var in variables if var in prognostic]
    verification_vars = [var for var in variables if var in verification]
    return prognostic[prognostic_vars], verification[verification_vars], grid
