from fv3net.diagnostics._shared.registry import Registry
import fv3net.diagnostics._shared.transform as transform
from fv3net.diagnostics._shared.constants import (
    DiagArg,
    HORIZONTAL_DIMS,
    COL_DRYING,
    WVP,
    HISTOGRAM_BINS,
)
import logging
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Dict, Union, Mapping, Optional
import xarray as xr

import vcm
from vcm import safe, weighted_average, local_time, histogram, histogram2d

logger = logging.getLogger(__name__)

DOMAINS = (
    "land",
    "sea",
    "global",
    "positive_net_precipitation",
    "negative_net_precipitation",
)
SURFACE_TYPE = "land_sea_mask"
SURFACE_TYPE_ENUMERATION = {0.0: "sea", 1.0: "land", 2.0: "sea"}
DERIVATION_DIM = "derivation"
COL_MOISTENING = "column_integrated_Q2"


def _prepare_diag_dict(suffix: str, ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """
    Take a diagnostic dataset and add a suffix to all variable names and return as dict.
    Useful in multiple merge functions passed to registries.
    """

    diags = {}
    for variable in ds:
        lower = str(variable).lower()
        da = ds[variable]
        diags[f"{lower}_{suffix}"] = da

    return diags


def merge_diagnostics(metrics: Sequence[Tuple[str, xr.Dataset]]):
    out: Dict[str, xr.DataArray] = {}
    for (name, ds) in metrics:
        out.update(_prepare_diag_dict(name, ds))
    # ignoring type error that complains if Dataset created from dict
    return xr.Dataset(out)  # type: ignore


diagnostics_registry = Registry(merge_diagnostics)


def compute_diagnostics(
    prediction: xr.Dataset,
    target: xr.Dataset,
    grid: xr.Dataset,
    delp: xr.DataArray,
    n_jobs: int = -1,
):
    return diagnostics_registry.compute(
        DiagArg(prediction, target, grid, delp), n_jobs=n_jobs
    )


def conditional_average_over_domain(
    ds: xr.Dataset,
    grid: xr.Dataset,
    primary_vars: Sequence[str],
    domain: str,
    net_precipitation: Optional[xr.DataArray] = None,
    uninformative_coords: Sequence[str] = ["tile", "z", "y", "x"],
) -> xr.Dataset:
    """Reduce a sequence of batches to a diagnostic dataset
    
    Args:
        ds: xarray datasets with relevant variables batched in time
        grid: xarray dataset containing grid variables
        (latb, lonb, lat, lon, area, land_sea_mask)
        domains: sequence of area domains over which to produce conditional
            averages; optional, defaults to global, land, sea, and positive and
            negative net_precipitation domains
        primary_vars: sequence of variables for which to compute column integrals
            and composite means; optional, defaults to dQs, pQs and Qs
        net_precipitation: xr.DataArray of net_precipitation values for computing
            composites, typically supplied by SHiELD net_precipitation; optional
        uninformative_coords: sequence of names of uninformative (i.e.,
            range(len(dim))), coordinates to be dropped
            
    Returns:
        diagnostic_ds: xarray dataset of reduced diagnostic variables
    """

    ds = ds.drop_vars(names=uninformative_coords, errors="ignore")

    grid = grid.drop_vars(names=uninformative_coords, errors="ignore")
    surface_type_array = _snap_mask_to_type(grid[SURFACE_TYPE])
    if "net_precipitation" in domain:
        net_precipitation_type_array = _snap_net_precipitation_to_type(
            net_precipitation
        )
        cell_type = net_precipitation_type_array.drop_vars(
            names=uninformative_coords, errors="ignore"
        )
    else:
        cell_type = surface_type_array
    domain_average = _conditional_average(
        safe.get_variables(ds, primary_vars), cell_type, domain, grid["area"],
    )
    return domain_average.mean("time")


def _conditional_average(
    ds: Union[xr.Dataset, xr.DataArray],
    cell_type_array: xr.DataArray,
    category: str,
    area: xr.DataArray,
    dims: Sequence[str] = ["tile", "y", "x"],
) -> xr.Dataset:
    """Average over a conditional type
    
    Args:
        ds: xr dataarray or dataset of variables to averaged conditionally
        cell_type_array: xr datarray of cell category strings
        category: str of category over which to conditionally average
        area: xr datarray of grid cell areas for weighted averaging
        dims: dimensions to average over
            
    Returns:
        xr dataarray or dataset of conditionally averaged variables
    """

    if category == "global":
        area_masked = area
    elif category in DOMAINS:
        area_masked = area.where(cell_type_array == category)
    else:
        raise ValueError(
            f"surface type {category} not in provided surface type array "
            f"with types {DOMAINS}."
        )

    return weighted_average(ds, area_masked, dims)


def _snap_mask_to_type(
    float_mask: xr.DataArray,
    enumeration: Mapping[float, str] = SURFACE_TYPE_ENUMERATION,
    atol: float = 1e-7,
) -> xr.DataArray:
    """Convert float surface type array to categorical surface type array
    
    Args:
        float_mask: xr dataarray of float cell types
        enumeration: mapping of surface type str names to float values
        atol: absolute tolerance of float value matching
            
    Returns:
        types: xr dataarray of str categorical cell types
    
    """

    types = np.full(float_mask.shape, np.nan)
    for type_number, type_name in enumeration.items():
        types = np.where(
            np.isclose(float_mask.values, type_number, atol), type_name, types
        )

    types = xr.DataArray(types, float_mask.coords, float_mask.dims)

    return types


def _snap_net_precipitation_to_type(
    net_precipitation: xr.DataArray, type_names: Mapping[str, str] = None
) -> xr.DataArray:
    """Convert net_precipitation array to positive and negative categorical types
    
    Args:
        net_precipitation: xr.DataArray of numerical values
        type_names: Mapping relating the "positive" and "negative" cases to their
            categorical type names
            
    Returns:
        types: xr dataarray of categorical str type
    
    """

    type_names = type_names or {
        "negative": "negative_net_precipitation",
        "positive": "positive_net_precipitation",
    }

    if any(key not in type_names for key in ["negative", "positive"]):
        raise ValueError("'type_names' must include 'positive' and negative' as keys")

    types = np.where(
        net_precipitation.values < 0, type_names["negative"], type_names["positive"]
    )

    return xr.DataArray(types, net_precipitation.coords, net_precipitation.dims)


def _calc_ds_diurnal_cycle(ds):
    """
    Calculates the diurnal cycle for all variables.  Expects
    time dimension and longitude variable "lon".
    """
    local_time_ = local_time(ds, time="time", lon_var="lon")
    local_time_.attrs = {"long_name": "local time", "units": "hour"}
    ds["local_time"] = np.floor(local_time_)  # equivalent to hourly binning
    with xr.set_options(keep_attrs=True):
        diurnal_cycles = ds.drop("lon").groupby("local_time").mean()
    return diurnal_cycles


def _predicts_sphum_tendency(ds):
    for var in ds.data_vars:
        if "q2" in var.lower():
            return True
    return False


def mse(x, y, w, dims):
    with xr.set_options(keep_attrs=True):
        return ((x - y) ** 2 * w).sum(dims) / w.sum(dims)


def bias(truth, prediction, w, dims):
    with xr.set_options(keep_attrs=True):
        return prediction - truth


def weighted_mean(ds, weights, dims):
    with xr.set_options(keep_attrs=True):
        return (ds * weights).sum(dims) / weights.sum(dims)


def zonal_mean(
    ds: xr.Dataset, latitude: xr.DataArray, bins=np.arange(-90, 91, 2)
) -> xr.Dataset:
    with xr.set_options(keep_attrs=True):
        zm = (
            ds.groupby_bins(latitude, bins=bins)
            .mean(skipna=True)
            .rename(lat_bins="latitude")
        )
    latitude_midpoints = [x.item().mid for x in zm["latitude"]]
    return zm.assign_coords(latitude=latitude_midpoints)


def _compute_wvp_vs_q2_histogram(ds: xr.Dataset) -> xr.Dataset:
    counts = xr.Dataset()
    col_drying = -ds[COL_MOISTENING].rename(COL_DRYING)
    col_drying.attrs["units"] = ds[COL_MOISTENING].attrs.get("units")
    bins = [HISTOGRAM_BINS[WVP], np.linspace(-50, 150, 101)]

    counts, wvp_bins, q2_bins = histogram2d(ds[WVP], col_drying, bins=bins)
    return xr.Dataset(
        {
            f"{WVP}_versus_{COL_DRYING}": counts,
            f"{WVP}_bin_width": wvp_bins,
            f"{COL_DRYING}_bin_width": q2_bins,
        }
    )


def _assign_source_attrs(
    diagnostics_ds: xr.Dataset, source_ds: xr.Dataset
) -> xr.Dataset:
    """Get attrs for each variable in diagnostics_ds from corresponding in source_ds."""
    for variable in diagnostics_ds:
        if variable in source_ds:
            attrs = source_ds[variable].attrs
            diagnostics_ds[variable] = diagnostics_ds[variable].assign_attrs(attrs)
    return diagnostics_ds


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"mse_2d_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def mse_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing mean squared errors for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        mse_area_weighted_avg = weighted_mean(
            (predicted - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return mse_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"mse_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_area, mask_type)
    def mse_3d(diag_arg, mask_type=mask_type):
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        logger.info(f"Preparing mean squared errors for 3D variables, {mask_type}")
        if len(predicted) == 0:
            return xr.Dataset()
        mse_area_weighted_avg = weighted_mean(
            (predicted - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return mse_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"variance_2d_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def variance_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing variance for 2D variables, {mask_type}")
        target, grid = diag_arg.verification, diag_arg.grid
        mean = weighted_mean(target, weights=grid.area, dims=HORIZONTAL_DIMS).mean(
            "time"
        )
        return weighted_mean(
            (mean - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        ).mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"variance_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_area, mask_type)
    def variance_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing variance for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        mean = weighted_mean(target, weights=grid.area, dims=HORIZONTAL_DIMS).mean(
            "time"
        )
        return weighted_mean(
            (mean - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        ).mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"bias_2d_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def bias_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing biases for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        biases_area_weighted_avg = weighted_mean(
            predicted - target, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return biases_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"bias_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_area, mask_type)
    def bias_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing biases for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        biases_area_weighted_avg = weighted_mean(
            predicted - target, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return biases_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"bias_2d_zonal_avg_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def bias_zonal_avg_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg biases for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        zonal_avg_bias = zonal_mean(predicted - target, grid.lat)
        return zonal_avg_bias.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"bias_pressure_level_zonal_avg_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def bias_zonal_avg_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg biases for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        zonal_avg_bias = zonal_mean(predicted - target, grid.lat)
        return zonal_avg_bias.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"mse_pressure_level_zonal_avg_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def mse_zonal_avg_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg MSEs for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        zonal_avg_mse = zonal_mean((predicted - target) ** 2, grid.lat)
        return zonal_avg_mse.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"variance_pressure_level_zonal_avg_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    @transform.apply(transform.mask_area, mask_type)
    def variance_zonal_avg_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg variance for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        mean = weighted_mean(target, weights=grid.area, dims=HORIZONTAL_DIMS).mean(
            "time"
        )
        zonal_avg_variance = zonal_mean((mean - target) ** 2, grid.lat)
        return zonal_avg_variance.mean("time")


for mask_type in ["global", "land", "sea"]:

    @diagnostics_registry.register(f"diurnal_cycle_{mask_type}")
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    @transform.apply(transform.select_2d_variables)
    def diurnal_cycle(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing diurnal cycle over sfc type {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        ds = xr.concat(
            [predicted, target],
            dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
        )
        ds["lon"] = grid["lon"]
        return _calc_ds_diurnal_cycle(ds)


for domain in ["positive_net_precipitation", "negative_net_precipitation"]:

    @diagnostics_registry.register(f"time_domain_mean_{domain}")
    @transform.apply(transform.subset_variables, ["dQ1", "dQ2", "Q2"])
    def precip_domain_time_mean(diag_arg, domain=domain):
        logger.info(f"Preparing conditional averages over domain {domain}")
        predicted, target, grid, delp = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
            diag_arg.delp,
        )
        if len(predicted) > 0 and _predicts_sphum_tendency(predicted):
            net_precip = vcm.minus_column_integrated_moistening(target["Q2"], delp)
            ds = xr.concat(
                [predicted, target],
                dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
            )
            domain_avg = conditional_average_over_domain(
                ds,
                grid,
                predicted.data_vars,
                domain=domain,
                net_precipitation=net_precip,
            )
            return domain_avg
        else:
            return xr.Dataset()


for domain in ["global", "land", "sea"]:

    @diagnostics_registry.register(f"time_domain_mean_{domain}")
    @transform.apply(transform.select_3d_variables)
    def surface_type_domain_time_mean(diag_arg, domain=domain):
        logger.info(f"Preparing conditional averages over domain {domain}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) > 0:
            ds = xr.concat(
                [predicted, target],
                dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
            )
            domain_avg = conditional_average_over_domain(
                ds, grid, predicted.data_vars, domain=domain,
            )
            return domain_avg
        else:
            return xr.Dataset()


@diagnostics_registry.register(f"time_mean_global")
def time_mean(diag_arg):
    logger.info(f"Preparing global time means")
    predicted, target = (
        diag_arg.prediction,
        diag_arg.verification,
    )
    if len(predicted) > 0:
        ds = xr.concat(
            [predicted, target],
            dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
        )
        return ds.mean("time")


@diagnostics_registry.register("hist_2d")
@transform.apply(transform.mask_to_sfc_type, "sea")
@transform.apply(transform.mask_to_sfc_type, "tropics20")
def compute_hist_2d(diag_arg: DiagArg):
    histograms = []
    if COL_MOISTENING in diag_arg.prediction and WVP in diag_arg.prediction:
        for ds in [diag_arg.prediction, diag_arg.verification]:
            logger.info("Computing joint histogram of water vapor path versus Q2")
            hist = _compute_wvp_vs_q2_histogram(ds).squeeze()
            histograms.append(_assign_source_attrs(hist, ds))
        return xr.concat(
            histograms, dim=pd.Index(["predict", "target"], name=DERIVATION_DIM)
        )
    else:
        return xr.Dataset()


@diagnostics_registry.register("histogram")
@transform.apply(transform.subset_variables, [WVP, COL_MOISTENING])
def compute_histogram(diag_arg: DiagArg):
    logger.info("Computing histograms for physics diagnostics")
    histograms = []
    if COL_MOISTENING in diag_arg.prediction and WVP in diag_arg.prediction:
        for ds in [diag_arg.prediction, diag_arg.verification]:
            ds[COL_DRYING] = -ds[COL_MOISTENING]
            ds[COL_DRYING].attrs["units"] = getattr(
                ds[COL_MOISTENING], "units", "mm/day"
            )
            counts = xr.Dataset()
            for varname in ds.data_vars:
                count, width = histogram(
                    ds[varname], bins=HISTOGRAM_BINS[varname.lower()], density=True
                )
                counts[varname] = count
                counts[f"{varname}_bin_width"] = width
            histograms.append(_assign_source_attrs(counts, ds))
        return xr.concat(
            histograms, dim=pd.Index(["predict", "target"], name=DERIVATION_DIM)
        )
    else:
        return xr.Dataset()
