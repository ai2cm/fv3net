#!/usr/bin/env python
# coding: utf-8
"""
This script computes diagnostics for prognostic runs.

Diagnostics are multiple dimensional curves that can be visualized to give
a more detailed look into the data underlying the metrics.

Diagnostics are loaded and saved in groupings corresponding to the source of the data
and its intended use. E.g. the "dycore" grouping includes diagnostics output by the
dynamical core of the model and saved in `atmos_dt_atmos.tile*.nc` while the "physics"
grouping contains outputs from the physics routines (`sfc_dt_atmos.tile*.nc` and
`diags.zarr`).
"""
from argparse import ArgumentParser
import sys

import datetime
import intake
import xarray as xr
from dask.distributed import Client, worker_client
import fsspec


from typing import Optional, Mapping, MutableMapping, Union, Tuple, Sequence, List

import vcm

from fv3net.diagnostics.prognostic_run import load_run_data as load_diags
from fv3net.diagnostics.prognostic_run import diurnal_cycle
from fv3net.diagnostics._shared.constants import (
    DiagArg,
    COL_DRYING,
    WVP,
    HISTOGRAM_BINS,
)
from .constants import (
    GLOBAL_AVERAGE_VARS,
    GLOBAL_BIAS_VARS,
    DIURNAL_CYCLE_VARS,
    TIME_MEAN_VARS,
    RMSE_VARS,
    PRESSURE_INTERPOLATED_VARS,
)
from fv3net.diagnostics._shared.registry import Registry
from fv3net.diagnostics._shared import transform
from fv3net.artifacts.metadata import StepMetadata


import logging

logger = logging.getLogger("SaveDiags")


def itcz_edges(
    psi: xr.DataArray, lat: str = "latitude",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute latitude of ITCZ edges given mass streamfunction at particular level."""
    lat_min = psi.sel({lat: slice(-30, 10)}).idxmin(lat)
    lat_max = psi.sel({lat: slice(-10, 30)}).idxmax(lat)
    return lat_min, lat_max


def _prepare_diag_dict(suffix: str, ds: xr.Dataset) -> Mapping[str, xr.DataArray]:
    """
    Take a diagnostic dataset and add a suffix to all variable names and return as dict.
    """

    diags = {}
    for variable in ds:
        lower = variable.lower()
        da = ds[variable]
        diags[f"{lower}_{suffix}"] = da

    return diags


def _merge_diag_computes(
    input_data: Mapping[str, Tuple[xr.Dataset, xr.Dataset, xr.Dataset]],
    registries: Mapping[str, Registry],
    client: Client,
    n_concurrent_2d_calcs: int = 8,
    n_concurrent_3d_calcs: int = 2,
) -> Mapping[str, xr.DataArray]:
    # Flattens list of all computations across registries before
    # parallelizing the computation.
    merged_input_data: MutableMapping[str, List] = {"2d": [], "3d": []}
    for registry_key, (prog, verif, grid) in input_data.items():

        if len(prog) == 0:
            logger.warn(
                f"prognostic data for {registry_key} missing. "
                "Skipping computation for {registry_key}."
            )
            continue

        if len(verif) == 0:
            logger.warn(
                f"verification data for {registry_key} missing. "
                "Skipping computation for {registry_key}."
            )
            continue

        diag_arg = DiagArg(prog, verif, grid)
        merged_input_data[registry_key] += [
            (func_name, func, registry_key, diag_arg)
            for func_name, func in registries[registry_key].funcs.items()
        ]

    def _compute(arg):
        func_name, func, key, diag_arg = arg
        # Retrieve the parent client to add current tasks to the same scheduler
        with worker_client() as client:  # noqa
            return registries[key].load(func_name, func, diag_arg)

    computed_outputs = []

    # 2D batches
    for i in range(0, len(merged_input_data["2d"]), n_concurrent_2d_calcs):
        futures = client.map(
            _compute, merged_input_data["2d"][i : i + n_concurrent_2d_calcs]
        )
        computed_outputs += client.gather(futures)

    # 3D batches
    for i in range(0, len(merged_input_data["3d"]), n_concurrent_3d_calcs):
        futures = client.map(
            _compute, merged_input_data["3d"][i : i + n_concurrent_3d_calcs]
        )
        computed_outputs += client.gather(futures)

    return merge_diags(computed_outputs)


def merge_diags(diags: Sequence[Tuple[str, xr.Dataset]]) -> Mapping[str, xr.DataArray]:
    out: MutableMapping[str, xr.DataArray] = {}
    for name, ds in diags:
        out.update(_prepare_diag_dict(name, ds))
    return out


# all functions added to these registries must take three xarray Datasets as inputs
# (specifically, the prognostic run output, the verification data and the grid dataset)
# and return an xarray dataset containing one or more diagnostics.
registries = {
    "2d": Registry(merge_diags),
    "3d": Registry(merge_diags),
}

# expressions not allowed in decorator calls, so need explicit variables for each here
registry_2d = registries["2d"]
registry_3d = registries["3d"]


def _is_empty(data: Union[xr.Dataset, xr.DataArray]) -> bool:
    if len(data) == 0 or 0 in data.sizes.values():
        return True
    else:
        return False


def ignore_nan_values(x, y, w):
    if "time" not in w.dims and "time" in x.dims:
        w = w.expand_dims(dim={"time": x.time})
    mask = (x.notnull() | y.notnull()).compute()
    x = x.where(mask, drop=True)
    y = y.where(mask, drop=True)
    w = w.where(mask, drop=True)
    return x, y, w


def rms(x, y, w, dims):
    with xr.set_options(keep_attrs=True):
        return xr.ufuncs.sqrt(((x - y) ** 2 * w).sum(dims) / w.sum(dims))


def bias(truth, prediction):
    with xr.set_options(keep_attrs=True):
        return prediction - truth


def _per_var_wrapper(func, num_ds_args=1):
    # Using to avoid many small tasks being submitted to the cluster
    # by pre-loading individual variable and then performing calculation.
    # Only using for 2d variables for now.  Obviously the full timeseries
    # Needs to fit in worker memory.  For 2-years ~ 1.5GB per variable
    # of scream ne30 data.  Since FV3 segments are shorter, I believe this
    # will be ok.
    def per_variable(ds, *args, **kwargs):
        # handles cases where the function is called with multiple datasets
        if num_ds_args > 1:
            dataset_args = args[: (num_ds_args - 1)]
            args = args[(num_ds_args - 1) :]
        else:
            dataset_args = []

        out = xr.Dataset()
        for var in ds.keys():
            logger.info(f"Computing {func.__name__} for {var}")
            try:
                data_inputs = [d[var].load() for d in [ds] + list(dataset_args)]
            except KeyError:
                logger.info(f"Variable {var} not found in dataset. Skipping.")
                continue
            out[var] = func(*data_inputs, *args, **kwargs)
        return out

    return per_variable


def weighted_mean(ds, weights, dims):
    with xr.set_options(keep_attrs=True):
        return (ds * weights).sum(dims) / weights.sum(dims)


def time_mean(da: xr.DataArray, dim: str = "time") -> xr.Dataset:
    with xr.set_options(keep_attrs=True):
        result = da.mean(dim)
    return _assign_diagnostic_time_attrs(result, da)


def _get_time_attrs(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[Mapping[str, str]]:
    if "time" in ds.coords and not _is_empty(ds):
        start_time = str(ds.time.values[0])
        end_time = str(ds.time.values[-1])
        return {"diagnostic_start_time": start_time, "diagnostic_end_time": end_time}
    else:
        return {}


def _assign_diagnostic_time_attrs(
    diagnostics_da: xr.DataArray, source_da: xr.DataArray
) -> xr.DataArray:
    attrs = _get_time_attrs(source_da)
    diagnostics_da = diagnostics_da.assign_attrs(attrs)
    return diagnostics_da


def _assign_source_attrs(
    diagnostics_ds: xr.Dataset, source_ds: xr.Dataset
) -> xr.Dataset:
    """Get attrs for each variable in diagnostics_ds from corresponding in source_ds."""
    for variable in diagnostics_ds:
        if variable in source_ds:
            attrs = source_ds[variable].attrs
            diagnostics_ds[variable] = diagnostics_ds[variable].assign_attrs(attrs)
    return diagnostics_ds


@registry_2d.register("rms_global")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.daily_mean, datetime.timedelta(days=10))
@transform.apply(transform.subset_variables, RMSE_VARS)
def rms_errors(diag_arg: DiagArg):
    logger.info("Preparing rms errors")
    prognostic, verification, grid = (
        diag_arg.prediction,
        diag_arg.verification,
        diag_arg.grid,
    )

    rms_errors = _per_var_wrapper(rms, num_ds_args=2)(
        prognostic, verification, grid.area.load(), dims=diag_arg.horizontal_dims
    )

    return rms_errors


def _zonal_and_time_mean(prognostic: xr.DataArray, grid: xr.Dataset):
    zonal_mean = vcm.zonal_average_approximate(
        grid.lat, prognostic, lat_name="latitude"
    )
    return time_mean(zonal_mean).load()


def _zonal_mean(da: xr.DataArray, grid: xr.Dataset):
    # mainly using to re-order args so I can use per_var wrapper
    return vcm.zonal_average_approximate(grid.lat, da, lat_name="latitude").load()


def _meridional_mean(da: xr.DataArray, grid: xr.Dataset):
    return vcm.meridional_average_approximate(grid.lon, da, lon_name="longitude").load()


@registry_2d.register("zonal_and_time_mean")
@transform.apply(transform.resample_time, "1H")
@transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
def zonal_means_2d(diag_arg: DiagArg):
    logger.info("Preparing zonal+time means (2d)")
    prognostic, grid = diag_arg.prediction, diag_arg.grid
    return _per_var_wrapper(_zonal_and_time_mean)(prognostic, grid.load())


@registry_3d.register("pressure_level_zonal_time_mean")
@transform.apply(transform.subset_variables, PRESSURE_INTERPOLATED_VARS)
@transform.apply(transform.skip_if_3d_output_absent)
@transform.apply(transform.resample_time, "3H")
def zonal_means_3d(diag_arg: DiagArg):
    logger.info("Preparing zonal+time means (3d)")
    prognostic, grid = diag_arg.prediction, diag_arg.grid
    zonal_means = xr.Dataset()
    for var in prognostic.data_vars:
        logger.info(f"Computing zonal+time means (3d) for {var}")
        with xr.set_options(keep_attrs=True):
            zm = vcm.zonal_average_approximate(
                grid.lat, prognostic[[var]], lat_name="latitude"
            )
            zm_time_mean = time_mean(zm)[var].load()
            zonal_means[var] = zm_time_mean
    return zonal_means


@registry_3d.register("pressure_level_zonal_bias")
@transform.apply(transform.subset_variables, PRESSURE_INTERPOLATED_VARS)
@transform.apply(transform.skip_if_3d_output_absent)
@transform.apply(transform.resample_time, "3H", inner_join=True)
def zonal_bias_3d(diag_arg: DiagArg):
    logger.info("Preparing zonal mean bias (3d)")
    prognostic, verification, grid = (
        diag_arg.prediction,
        diag_arg.verification,
        diag_arg.grid,
    )
    bias_ = bias(verification, prognostic)
    if _is_empty(bias_):
        return xr.Dataset()

    zonal_means = xr.Dataset()
    common_vars = list(set(prognostic.data_vars).intersection(verification.data_vars))
    for var in common_vars:
        logger.info(f"Computing zonal+time mean biases (3d) for {var}")
        with xr.set_options(keep_attrs=True):
            zm_bias = vcm.zonal_average_approximate(
                grid.lat, bias_[[var]], lat_name="latitude",
            )
            zm_bias_time_mean = time_mean(zm_bias)[var].load()
            zonal_means[var] = zm_bias_time_mean
    return zonal_means


@registry_2d.register("zonal_bias")
@transform.apply(transform.resample_time, "1H")
@transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
def zonal_and_time_mean_biases_2d(diag_arg: DiagArg):
    prognostic, verification, grid = (
        diag_arg.prediction,
        diag_arg.verification,
        diag_arg.grid,
    )
    logger.info("Preparing zonal+time mean biases (2d)")

    bias_ = bias(verification, prognostic)
    if _is_empty(bias_):
        return xr.Dataset()

    common_vars = list(set(prognostic.data_vars).intersection(verification.data_vars))
    return _per_var_wrapper(_zonal_and_time_mean)(bias_[common_vars], grid.load())


@registry_2d.register("zonal_mean_value")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.daily_mean, datetime.timedelta(days=10))
@transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
def zonal_mean_hovmoller(diag_arg: DiagArg):
    logger.info(f"Preparing zonal mean values (2d)")
    prognostic, grid = diag_arg.prediction, diag_arg.grid
    return _per_var_wrapper(_zonal_mean)(prognostic, grid.load())


@registry_2d.register("zonal_mean_bias")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.daily_mean, datetime.timedelta(days=10))
@transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
def zonal_mean_bias_hovmoller(diag_arg: DiagArg):

    logger.info(f"Preparing zonal mean biases (2d)")
    prognostic, verification, grid = (
        diag_arg.prediction,
        diag_arg.verification,
        diag_arg.grid,
    )

    bias_ = bias(verification, prognostic)
    if _is_empty(bias_):
        return xr.Dataset()

    common_vars = list(set(prognostic.data_vars).intersection(verification.data_vars))
    return _per_var_wrapper(_zonal_mean)(bias_[common_vars], grid.load())


@registry_2d.register("deep_tropical_meridional_mean_value")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.subset_variables, ["total_precip_to_surface", "ULWRFtoa"])
@transform.apply(transform.mask_to_sfc_type, "tropics20")
def deep_tropical_mean_hovmoller_value(diag_arg: DiagArg):
    logger.info(f"Preparing deep tropical meridional mean values (2d)")
    result = _per_var_wrapper(_meridional_mean)(
        diag_arg.prediction, diag_arg.grid.load()
    )
    return result.rename({"time": "high_frequency_time"})


@registry_2d.register("deep_tropical_meridional_mean_bias")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.subset_variables, ["total_precip_to_surface", "ULWRFtoa"])
@transform.apply(transform.mask_to_sfc_type, "tropics20")
def deep_tropical_mean_hovmoller_bias(diag_arg: DiagArg):
    logger.info(f"Preparing deep tropical meridional mean biases (2d)")
    grid = diag_arg.grid.load()
    trop_bias = bias(diag_arg.verification, diag_arg.prediction)
    return _per_var_wrapper(_meridional_mean)(trop_bias, grid).rename(
        {"time": "high_frequency_time"}
    )


def _per_variable_masked(
    func: str,
    mask: xr.DataArray,
    prognostic: xr.Dataset,
    horizontal_dims: Tuple[str, ...],
) -> xr.Dataset:

    out = xr.Dataset()
    for var, da in prognostic.items():
        da = da.load()
        masked = da.where(mask)
        with xr.set_options(keep_attrs=True):
            out[var] = getattr(masked, func)(dim=horizontal_dims).load()

    return out


for mask_type in ["global", "land", "sea", "tropics"]:

    @registry_2d.register(f"spatial_min_{mask_type}")
    @transform.apply(transform.mask_area, mask_type)
    @transform.apply(transform.resample_time, "3H")
    @transform.apply(transform.daily_mean, datetime.timedelta(days=10))
    @transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
    def spatial_min(diag_arg: DiagArg, mask_type=mask_type):
        logger.info(f"Preparing minimum for variables ({mask_type})")
        prognostic, grid = diag_arg.prediction, diag_arg.grid
        return _per_variable_masked(
            "min", ~grid["area"].isnull().load(), prognostic, diag_arg.horizontal_dims
        )

    @registry_2d.register(f"spatial_max_{mask_type}")
    @transform.apply(transform.mask_area, mask_type)
    @transform.apply(transform.resample_time, "3H")
    @transform.apply(transform.daily_mean, datetime.timedelta(days=10))
    @transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
    def spatial_max(diag_arg: DiagArg, mask_type=mask_type):
        logger.info(f"Preparing maximum for variables ({mask_type})")
        prognostic, grid = diag_arg.prediction, diag_arg.grid
        return _per_variable_masked(
            "min", ~grid["area"].isnull().load(), prognostic, diag_arg.horizontal_dims
        )


for mask_type in ["global", "land", "sea", "tropics"]:

    @registry_2d.register(f"spatial_mean_{mask_type}")
    @transform.apply(transform.mask_area, mask_type)
    @transform.apply(transform.resample_time, "3H")
    @transform.apply(transform.daily_mean, datetime.timedelta(days=10))
    @transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
    def global_averages_2d(diag_arg: DiagArg, mask_type=mask_type):
        logger.info(f"Preparing averages for 2d variables ({mask_type})")
        prognostic, grid = diag_arg.prediction, diag_arg.grid
        return _per_var_wrapper(weighted_mean)(
            prognostic, grid.area.load(), diag_arg.horizontal_dims
        )

    @registry_2d.register(f"mean_bias_{mask_type}")
    @transform.apply(transform.mask_area, mask_type)
    @transform.apply(transform.resample_time, "3H", inner_join=True)
    @transform.apply(transform.daily_mean, datetime.timedelta(days=10))
    @transform.apply(transform.subset_variables, GLOBAL_BIAS_VARS)
    def global_biases_2d(diag_arg: DiagArg, mask_type=mask_type):
        logger.info(f"Preparing average biases for 2d variables ({mask_type})")
        prognostic, verification, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        bias_errors = bias(verification, prognostic)
        mean_bias_errors = _per_var_wrapper(weighted_mean)(
            bias_errors, grid.area.load(), diag_arg.horizontal_dims
        )
        return mean_bias_errors


@registry_2d.register("time_mean_value")
@transform.apply(transform.resample_time, "1H", inner_join=True)
@transform.apply(transform.subset_variables, TIME_MEAN_VARS)
def time_means_2d(diag_arg: DiagArg):
    logger.info("Preparing time means for 2d variables")
    prognostic = diag_arg.prediction
    return _per_var_wrapper(time_mean)(prognostic)


@registry_2d.register("time_mean_bias")
@transform.apply(transform.resample_time, "1H", inner_join=True)
@transform.apply(transform.subset_variables, TIME_MEAN_VARS)
def time_mean_biases_2d(diag_arg: DiagArg):
    logger.info("Preparing time mean biases for 2d variables")
    prognostic, verification = diag_arg.prediction, diag_arg.verification
    return _per_var_wrapper(time_mean)(bias(verification, prognostic))


for mask_type in ["global", "land", "sea"]:

    @registry_2d.register(f"diurnal_{mask_type}")
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    @transform.apply(transform.resample_time, "1H", inner_join=True)
    @transform.apply(transform.subset_variables, DIURNAL_CYCLE_VARS)
    def _diurnal_func(diag_arg: DiagArg, mask_type=mask_type) -> xr.Dataset:
        # mask_type is added as a kwarg solely to give the logging access to the info
        logger.info(
            f"Computing diurnal cycle info for physics variables with mask={mask_type}"
        )
        prognostic, verification, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if _is_empty(prognostic):
            return xr.Dataset({})
        else:
            diag = diurnal_cycle.calc_diagnostics(prognostic, verification, grid)
            return _assign_diagnostic_time_attrs(diag, prognostic)


@registry_2d.register("histogram")
@transform.apply(transform.resample_time, "3H", inner_join=True, method="mean")
@transform.apply(transform.subset_variables, list(HISTOGRAM_BINS.keys()))
def compute_histogram(diag_arg: DiagArg):
    logger.info("Computing histograms for physics diagnostics")
    prognostic = diag_arg.prediction
    counts = xr.Dataset()
    for varname in prognostic.data_vars:
        count, width = vcm.histogram(
            prognostic[varname].load(), bins=HISTOGRAM_BINS[varname], density=True
        )
        counts[varname] = count
        counts[f"{varname}_bin_width"] = width
    return _assign_source_attrs(
        _assign_diagnostic_time_attrs(counts, prognostic), prognostic
    )


@registry_2d.register("hist_bias")
@transform.apply(transform.resample_time, "3H", inner_join=True, method="mean")
@transform.apply(transform.subset_variables, list(HISTOGRAM_BINS.keys()))
def compute_histogram_bias(diag_arg: DiagArg):
    logger.info("Computing histogram biases for physics diagnostics")
    prognostic, verification = diag_arg.prediction, diag_arg.verification
    counts = xr.Dataset()
    for varname in prognostic.data_vars:
        prognostic_count, _ = vcm.histogram(
            prognostic[varname].load(), bins=HISTOGRAM_BINS[varname], density=True
        )
        verification_count, _ = vcm.histogram(
            verification[varname].load(), bins=HISTOGRAM_BINS[varname], density=True
        )
        counts[varname] = bias(verification_count, prognostic_count)
    return _assign_source_attrs(
        _assign_diagnostic_time_attrs(counts, prognostic), prognostic
    )


@registry_2d.register("hist_2d")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.mask_to_sfc_type, "sea")
@transform.apply(transform.mask_to_sfc_type, "tropics20")
def compute_hist_2d(diag_arg: DiagArg):
    logger.info("Computing joint histogram of water vapor path versus Q2")
    hist2d = _compute_wvp_vs_q2_histogram(diag_arg.prediction)
    return _assign_diagnostic_time_attrs(hist2d, diag_arg.prediction)


@registry_2d.register("hist2d_bias")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.mask_to_sfc_type, "sea")
@transform.apply(transform.mask_to_sfc_type, "tropics20")
def compute_hist_2d_bias(diag_arg: DiagArg):
    logger.info("Computing bias of joint histogram of water vapor path versus Q2")
    hist2d_prog = _compute_wvp_vs_q2_histogram(diag_arg.prediction)
    hist2d_verif = _compute_wvp_vs_q2_histogram(diag_arg.verification)
    name = f"{WVP}_versus_{COL_DRYING}"
    error = bias(hist2d_verif[name], hist2d_prog[name])
    hist2d_prog.update({name: error})
    return _assign_diagnostic_time_attrs(hist2d_prog, diag_arg.prediction)


@registry_3d.register("300_700_zonal_mean_value")
@transform.apply(transform.subset_variables, ["northward_wind"])
@transform.apply(transform.skip_if_3d_output_absent)
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.daily_mean, datetime.timedelta(days=10))
def time_dependent_mass_streamfunction(diag_arg: DiagArg):
    logger.info("Computing mid-troposphere averaged mass streamfunction")
    prognostic, grid = diag_arg.prediction, diag_arg.grid
    if _is_empty(prognostic):
        return xr.Dataset()
    northward_wind = prognostic["northward_wind"]
    v_zm = vcm.zonal_average_approximate(grid.lat, northward_wind, lat_name="latitude")
    psi = vcm.mass_streamfunction(v_zm).sel(pressure=slice(30000, 70000))
    with xr.set_options(keep_attrs=True):
        psi_mid_trop = psi.weighted(psi.pressure).mean("pressure")
    return xr.Dataset({"mass_streamfunction": psi_mid_trop})


@registry_3d.register("300_700_zonal_mean_bias")
@transform.apply(transform.subset_variables, ["northward_wind"])
@transform.apply(transform.skip_if_3d_output_absent)
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.daily_mean, datetime.timedelta(days=10))
def time_dependent_mass_streamfunction_bias(diag_arg: DiagArg):
    logger.info("Computing mid-troposphere averaged mass streamfunction bias")
    prognostic, grid = diag_arg.prediction, diag_arg.grid
    if _is_empty(prognostic) or _is_empty(diag_arg.verification):
        return xr.Dataset()
    wind_bias = prognostic["northward_wind"] - diag_arg.verification["northward_wind"]
    v_zm = vcm.zonal_average_approximate(grid.lat, wind_bias, lat_name="latitude")
    psi = vcm.mass_streamfunction(v_zm).sel(pressure=slice(30000, 70000))
    with xr.set_options(keep_attrs=True):
        psi_mid_trop = psi.weighted(psi.pressure).mean("pressure")
    return xr.Dataset({"mass_streamfunction": psi_mid_trop})


def _compute_wvp_vs_q2_histogram(ds: xr.Dataset) -> xr.Dataset:
    counts = xr.Dataset()
    bins = [HISTOGRAM_BINS[WVP], HISTOGRAM_BINS[COL_DRYING]]
    counts, wvp_bins, q2_bins = vcm.histogram2d(
        ds[WVP].load(), ds[COL_DRYING].load(), bins=bins
    )
    return xr.Dataset(
        {
            f"{WVP}_versus_{COL_DRYING}": counts,
            f"{WVP}_bin_width": wvp_bins,
            f"{COL_DRYING}_bin_width": q2_bins,
        }
    )


def add_catalog_and_verification_arguments(parser: ArgumentParser):
    parser.add_argument("--catalog", default=vcm.catalog.catalog_path)
    verification_group = parser.add_mutually_exclusive_group()
    verification_group.add_argument(
        "--verification",
        help="Tag for simulation to use as verification data. Checks against "
        "'simulation' metadata from intake catalog.",
        default="40day_may2020",
    )
    verification_group.add_argument(
        "--verification-url",
        default="",
        type=str,
        help="URL to segmented run. "
        "If not passed then the --verification argument is used.",
    )


def add_time_range_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--start-date", type=str, help="Start date for diagnostics", default=None
    )
    parser.add_argument(
        "--end-date", type=str, help="End date for diagnostics", default=None
    )


def clean_up_diags_attrs(diags: xr.Dataset) -> xr.Dataset:
    """
    Remove attrs from diags that are not serializable.
    Specifically, we cast start_date and end_date which has default of None to str.
    """
    if "start_date" in diags.attrs:
        diags.attrs["start_date"] = str(diags.attrs["start_date"])
    if "end_date" in diags.attrs:
        diags.attrs["end_date"] = str(diags.attrs["end_date"])
    return diags


def register_parser(subparsers):
    parser: ArgumentParser = subparsers.add_parser(
        "save", help="Compute the prognostic run diags."
    )
    parser.add_argument("url", help="Prognostic run output location.")
    parser.add_argument("output", help="Output path including filename.")
    add_catalog_and_verification_arguments(parser)
    add_time_range_arguments(parser)
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Parallelism for dask computation of diagnostics. "
        "Defaults to using all available cores. Can set to a lower fixed value "
        "if you are often running  into read errors when multiple processes "
        "access data concurrently.  Ignored if a scheduler file is passed.",
        default=None,
    )
    parser.add_argument(
        "--scheduler-file",
        type=str,
        help="Path to dask scheduler file. If passed, the client will connect to the "
        "scheduler instead of creating a new one.",
        default="",
    )
    parser.add_argument(
        "--num-concurrent-2d-calcs",
        type=int,
        help="Number of 2d diagnostics functions to compute concurrently.",
        default=8,
    )
    parser.add_argument(
        "--num-concurrent-3d-calcs",
        type=int,
        help="Number of 3d diagnostics functions to compute concurrently.",
        default=2,
    )
    parser.add_argument(
        "--gsrm",
        type=str,
        help="The type of GSRM used to generate the prognostic run,\
              either `fv3gfs` or `scream`",
        default="fv3gfs",
    )
    parser.set_defaults(func=main)


def get_verification(args, catalog, join_2d="outer"):
    if args.verification_url:
        if args.gsrm == "scream":
            return load_diags.ScreamSimulation(args.verification_url)
        else:
            return load_diags.SegmentedRun(
                args.verification_url, catalog, join_2d=join_2d
            )
    else:
        return load_diags.CatalogSimulation(args.verification, catalog, join_2d=join_2d)


def main(args):

    logging.basicConfig(level=logging.INFO)
    if args.scheduler_file:
        client = Client(scheduler_file=args.scheduler_file)
    else:
        client = Client(n_workers=args.n_jobs)
    attrs = vars(args)
    attrs = {k: "null" for k, v in attrs.items() if v is None}
    attrs["history"] = " ".join(sys.argv)

    # begin constructing diags
    diags = {}
    catalog = intake.open_catalog(args.catalog)
    if args.gsrm == "scream":
        prognostic = load_diags.ScreamSimulation(args.url)
    elif args.gsrm == "fv3gfs":
        prognostic = load_diags.SegmentedRun(args.url, catalog)
    else:
        raise ValueError(f"gsrm must be either `fv3gfs` or `scream`")
    verification = get_verification(args, catalog)
    attrs["verification"] = str(verification)

    grid = load_diags.load_grid(catalog, args.gsrm)
    input_data = load_diags.evaluation_pair_to_input_data(
        prognostic, verification, grid, args.start_date, args.end_date
    )

    computed_diags = _merge_diag_computes(
        input_data,
        registries,
        client,
        n_concurrent_2d_calcs=args.num_concurrent_2d_calcs,
        n_concurrent_3d_calcs=args.num_concurrent_3d_calcs,
    )
    diags.update(computed_diags)

    # add grid vars
    diags = xr.Dataset(diags, attrs=attrs)
    diags = diags.merge(grid)

    logger.info("Forcing remaining computation.")
    diags = diags.load()

    logger.info(f"Saving data to {args.output}")
    with fsspec.open(args.output, "wb") as f:
        diags = clean_up_diags_attrs(diags)
        vcm.dump_nc(diags, f)

    StepMetadata(
        job_type="prognostic_run_diags",
        url=args.output,
        dependencies={"prognostic_run": args.url},
        args=sys.argv[1:],
    ).print_json()
