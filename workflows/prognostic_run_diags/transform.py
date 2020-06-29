import logging

from toolz import memoize

import save_prognostic_run_diags
import vcm

_TRANSFORM_FNS = {}
HORIZONTAL_DIMS = save_prognostic_run_diags.HORIZONTAL_DIMS

DiagArg = save_prognostic_run_diags.DiagArg
logger = logging.getLogger(__name__)


def add_to_input_transform_fns(func):

    _TRANSFORM_FNS[func.__name__] = func

    return func


def _args_to_hashable_key(args, kwargs):
    # Convert unhashable diags to a hashable string
    # Doesn't explicitly represent full dataset but enough for us to
    # cache the relatively unchanging input datasets to transform operations
    diag_arg = "".join([str(ds) for ds in args[0]])
    hargs = [diag_arg] + list(args[1:])
    hkwargs = [(key, kwargs[key]) for key in sorted(kwargs.keys())]
    hashable_key = tuple(hargs + hkwargs)
    return hashable_key


@add_to_input_transform_fns
@memoize(key=_args_to_hashable_key)
def resample_time(arg: DiagArg, freq_label: str, time_slice=slice(None, -1)) -> DiagArg:
    prognostic, verification, grid = arg
    prognostic = prognostic.resample(time=freq_label, label="right").nearest()
    prognostic = prognostic.isel(time=time_slice)
    if "time" in verification:  # verification might be an empty dataset
        verification = verification.sel(time=prognostic.time)
    return prognostic, verification, grid


def _mask_vars_with_horiz_dims(ds, surface_type, mask_var_name):
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
    spatial = ds[spatial_ds_varnames + [mask_var_name]]
    masked = vcm.mask_to_surface_type(
        spatial, surface_type, surface_type_var=mask_var_name
    )

    return ds.update(masked)


@add_to_input_transform_fns
@memoize(key=_args_to_hashable_key)
def mask_to_sfc_type(
    arg: DiagArg, surface_type: str, mask_var_name: str = "SLMSKsfc"
) -> DiagArg:
    prognostic, verification, grid = arg
    masked_prognostic = _mask_vars_with_horiz_dims(
        prognostic, surface_type, mask_var_name
    )
    try:
        masked_verification = _mask_vars_with_horiz_dims(
            verification, surface_type, mask_var_name
        )
    except KeyError:
        logger.error("Empty verification dataset provided. TODO: fix this by loading")
        masked_verification = verification

    return masked_prognostic, masked_verification, grid


def apply_transform(transform_params, func):
    """
    Wrapper to apply transform to input diagnostic arguments (tuple of three datasets).
    Transform arguments are specified per diagnostic function to enable a query-style
    operation on input data.
    
    Note: I tried memoizing the current transforms but am unsure
    if it will work on highly mutable datasets.
    """

    transform_key, transform_args_partial, transform_kwargs = transform_params
    if transform_key not in _TRANSFORM_FNS:
        raise KeyError(
            f"Unrecognized transform, {transform_key} requested for {func.__name__}"
        )

    transform_func = _TRANSFORM_FNS[transform_key]

    def transform(*diag_args):

        logger.debug(
            f"Adding transform, {transform_key}, "
            f"to diagnostic function: {func.__name__}"
            f"\n\targs: {transform_args_partial}"
            f"\n\tkwargs: {transform_kwargs}"
        )

        # prepend diagnostic function input to be transformed
        transform_args = (diag_args, *transform_args_partial)

        transformed_diag_args = transform_func(*transform_args, **transform_kwargs)

        return func(*transformed_diag_args)

    return transform
