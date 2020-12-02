import logging
from typing import Sequence, Callable, Union
import warnings
import xarray as xr


from vcm import safe, interpolate_to_pressure_levels
from vcm.select import zonal_average_approximate
import copy


logging.getLogger(__name__)

# ignore warning from interpolating pressures when lowest level > sea level
warnings.filterwarnings(
    "ignore", message="Interpolation point out of data bounds encountered"
)

# Dimension/ coord/ variable parameter defaults
PREDICT_COORD = "predict"
TARGET_COORD = "target"
DERIVATION_DIM = "derivation"
PRESSURE_DIM = "pressure"
VERTICAL_DIM = "z"
AREA_VAR = "area"
DELP_VAR = "pressure_thickness_of_atmospheric_layer"
TOA_PRESSURE = 300.0  # Pa
VERTICAL_PROFILE_MEAN_DIMS = ("sample", "x", "y", "tile")

metric_sets = {"scalar_metrics_column_integrated_vars": {"dim_tag": "scalar"}}

MetricFunctionInputs = Union[xr.DataArray, xr.DataArray, xr.DataArray, Sequence[str]]


def calc_metrics(
    ds: xr.Dataset,
    lat: xr.DataArray,
    area: xr.DataArray,
    delp: xr.DataArray,
    predicted_vars: Sequence[str],
    predict_coord: str = PREDICT_COORD,
    target_coord: str = TARGET_COORD,
    derivation_dim: str = DERIVATION_DIM,
    vertical_dim: str = VERTICAL_DIM,
    area_var: str = AREA_VAR,
    delp_var: str = DELP_VAR,
    vertical_profile_mean_dims: Sequence[str] = VERTICAL_PROFILE_MEAN_DIMS,
) -> xr.Dataset:
    """Routine for computing ML prediction metrics (_bias, _mse]) on a dataset of
    variables, assumed to include variables in
    list arg predicted as well as area and delp
    """
    column_integrated_vars = [f"column_integrated_{name}" for name in predicted_vars]
    vertical_bias_vars = [var for var in predicted_vars if var not in ["Q1", "Q2"]]

    derivation_kwargs = {
        "predict_coord": predict_coord,
        "target_coord": target_coord,
        "derivation_dim": derivation_dim,
    }

    area_weights = area / (area.mean())
    delp_weights = delp / delp.mean(vertical_dim)
    ds_regrid_z = _regrid_dataset_zdim(ds, vertical_dim, **derivation_kwargs)

    metric_sets = {
        "scalar_metrics_column_integrated_vars": {
            "ds": ds,
            "dim_tag": "scalar",
            "vars": column_integrated_vars,
            "weights": [area_weights],
            "mean_dim_vars": None,
        },
        "scalar_column_integrated_metrics": {
            "ds": ds,
            "dim_tag": "scalar",
            "vars": predicted_vars,
            "weights": [area_weights, delp_weights],
            "mean_dim_vars": None,
        },
        "pressure_level_metrics": {
            "ds": ds_regrid_z,
            "dim_tag": "pressure_level",
            "vars": predicted_vars,
            "weights": [area_weights],
            "mean_dim_vars": vertical_profile_mean_dims,
        },
        "zonal_avg_pressure_level": {
            "ds": ds_regrid_z,
            "dim_tag": "zonal_avg_pressure_level",
            "vars": vertical_bias_vars,
            "weights": [],
            "mean_dim_vars": ["sample"],
            "metric_funcs": (_bias,),
        },
    }

    metrics = []
    for group, kwargs in metric_sets.items():
        metrics_ = _calc_same_dims_metrics(**kwargs)  # type: ignore
        if "zonal_avg" in group:
            metrics_ = zonal_average_approximate(lat, metrics_).rename(
                {"lat": "lat_interp"}
            )
        metrics.append(metrics_)

    zonal_error = []
    for var in predicted_vars:
        mse_zonal, variance_zonal = _zonal_avg_mse_variance(
            target=ds_regrid_z[var].sel({derivation_dim: target_coord}),
            predict=ds_regrid_z[var].sel({derivation_dim: predict_coord}),
            lat=lat,
            dims=["sample"],
        )
        mse_zonal = mse_zonal.rename(
            f"zonal_avg_pressure_level/mse/{var}/predict_vs_target"
        )
        variance_zonal = variance_zonal.rename(
            f"zonal_avg_pressure_level/mse/{var}/mean_vs_target"
        )
        zonal_error += [
            mse_zonal.rename({"lat": "lat_interp"}),
            variance_zonal.rename({"lat": "lat_interp"}),
        ]
    ds = xr.merge(metrics + zonal_error)
    return ds


def _regrid_dataset_zdim(
    ds: xr.Dataset,
    vertical_dim: str = VERTICAL_DIM,
    delp_var: str = DELP_VAR,
    predict_coord: str = PREDICT_COORD,
    target_coord: str = TARGET_COORD,
    derivation_dim: str = DERIVATION_DIM,
) -> xr.Dataset:
    # have to separate the derivation coordinates before interpolating
    # to regridded pressure
    regridded_datasets = []
    vertical_dim_vars = [var for var in ds.data_vars if vertical_dim in ds[var].dims]
    ds_2d = ds.drop(vertical_dim_vars)
    ds_3d = safe.get_variables(ds, vertical_dim_vars)

    for derivation_coord in [target_coord, predict_coord]:
        ds_regrid = ds_3d.sel({derivation_dim: derivation_coord})
        for var in vertical_dim_vars:
            ds_regrid[var] = interpolate_to_pressure_levels(
                delp=ds[delp_var], field=ds_regrid[var], dim=vertical_dim,
            )
        regridded_datasets.append(ds_regrid)
    return xr.merge([ds_2d, xr.concat(regridded_datasets, dim=derivation_dim)])


def _calc_same_dims_metrics(
    ds: xr.Dataset,
    dim_tag: str,
    vars: Sequence[str],
    weights: Sequence[Union[float, xr.DataArray]] = None,
    mean_dim_vars: Sequence[str] = None,
    predict_coord: str = PREDICT_COORD,
    target_coord: str = TARGET_COORD,
    derivation_dim: str = DERIVATION_DIM,
    metric_funcs: Sequence[Callable] = (),
) -> xr.Dataset:
    """Computes a set of metrics that all have the same dimension,
    ex. mean vertical error profile on pressure levels, or global mean scalar error
    Args:
        ds: test dataset
        dim_tag: describe the dimensions of the metrics returned. e.g. "scalar", "z"
        vars: data variables to calculate metrics on
        weight_vars: data variables to use as weights
        mean_dim_vars: Sequence of dimensions to take each weighted mean over
            for final value. Must match order of weights.
            Defaults to None (mean over all dims).
        target_coord: derivation coord for TARGET_COORD values in metric comparison.
        predict_coord: derivation coord for "prediction".
        derivation_dim: dimension name for derivation (comparision)

    Returns:
        dataset of metrics
    """
    weights = weights or [1.0]
    metric_comparison_coords = [(target_coord, predict_coord), (target_coord, "mean")]

    ds = _insert_means(
        ds, vars, predict_coord, target_coord, derivation_dim, weights, mean_dim_vars
    )
    metrics = xr.Dataset()
    metric_funcs = metric_funcs or (_bias, _mse)
    for var in vars:
        for metric_func in metric_funcs:
            for comparison in metric_comparison_coords:
                metric = _calc_metric(
                    ds,
                    metric_func,
                    var,
                    derivation_dim,
                    *comparison,
                    weights,
                    mean_dim_vars,
                )
                metrics[f"{dim_tag}/{metric.name}"] = metric
    return metrics


def _insert_means(
    ds: xr.Dataset,
    var_names: Sequence[str],
    predict_coord: str = PREDICT_COORD,
    target_coord: str = TARGET_COORD,
    derivation_dim: str = DERIVATION_DIM,
    weights: Sequence[Union[float, xr.DataArray]] = None,
    mean_dims: Sequence[str] = None,
) -> xr.Dataset:
    weights = weights or [1.0]
    for var in var_names:
        da = ds[var].sel({derivation_dim: [target_coord, predict_coord]})
        da_mean = _weighted_average(
            da.sel({derivation_dim: target_coord}), weights, mean_dims
        ).assign_coords({derivation_dim: "mean"})
        da = xr.concat([da, da_mean], dim=derivation_dim)
        ds = ds.drop([var])
        ds = ds.merge(da)
    return ds


def _calc_metric(
    ds: xr.Dataset,
    metric_func: Callable[[MetricFunctionInputs, MetricFunctionInputs], xr.DataArray],
    var: str,
    derivation_dim: str,
    target_coord: str,
    predict_coord: str,
    weights: Sequence[Union[float, xr.DataArray]] = None,
    mean_dims: Sequence[str] = None,
) -> xr.DataArray:
    """helper function to calculate arbitrary metrics given the variable, target, and
    prediction coords

    Args:
        ds: Dataset with variables to calculate metrics on. Variables
            should have a DERIVATION dim with coords denoting the sets for comparison,
            e.g. "prediction", TARGET_COORD
        metric_func: metric calculation function.
            One of {_mse, _biase}.
        var: Variable name to calculate metric for
        target_coord: derivation coord for TARGET_COORD values in metric comparison.
        predict_coord: derivation coord for "prediction".
        derivation_dim: dimension name for derivation (comparision)
        weights: data arrays to weight the metric over before taking mean.
        mean_dims: Sequence of dimensions to take each weighted mean over
            for final value. Must match order of weights. Defaults to None (all)

    Returns:
        xr.DataArray: data array of metric values
    """
    da_target = ds[var].sel({derivation_dim: target_coord})
    da_predict = ds[var].sel({derivation_dim: predict_coord})
    metric = metric_func(da_target, da_predict)

    metric_weighted_average = _weighted_average(
        metric, weights, mean_dims  # type: ignore
    )
    metric_name = (
        f"{metric_func.__name__.strip('_')}/{var}/{predict_coord}_vs_{target_coord}"
    )
    return metric_weighted_average.rename(metric_name)


def _bias(da_target: xr.DataArray, da_pred: xr.DataArray,) -> xr.DataArray:
    return da_pred - da_target


def _mse(
    da_target: xr.DataArray, da_pred: xr.DataArray,
):
    return (da_target - da_pred) ** 2


def _weighted_average(
    data: Union[xr.DataArray, xr.Dataset],
    weights: Sequence[Union[float, xr.DataArray]],
    mean_dims: Sequence[str] = None,
):
    # Differs from diagnostics_utils.weighted_average in that
    # this does multiple weightings (e.g. area + delp) in one go
    # NOTE: this assumes the variables used in weighting have already
    # been transformed into weights!
    data_copy = copy.deepcopy(data)
    for weight in weights:
        data_copy *= weight
    return data_copy.mean(dim=mean_dims, skipna=True)


def _zonal_avg_mse_variance(
    target: xr.DataArray, predict: xr.DataArray, lat, dims: Sequence[str] = None,
):
    """
    Compute the percent of variance explained of the anomaly from the zonal mean
    This is done separately from the _calc_metrics func because it uses the
    zonal_average_approximate function.
    """
    sse = (predict - target) ** 2
    mse_zonal = zonal_average_approximate(lat, sse).mean(dims)

    variance_zonal = (
        zonal_average_approximate(lat, target ** 2)
        - zonal_average_approximate(lat, target) ** 2
    ).mean(dims)

    return mse_zonal, variance_zonal
