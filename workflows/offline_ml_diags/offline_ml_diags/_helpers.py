import fsspec
import json
import numpy as np
import os
import shutil
from typing import Mapping, Sequence, Dict
import yaml
import xarray as xr

import report
from vcm import safe
from vcm.cloud import gsutil
from vcm.catalog import catalog


GRID_INFO_VARS = [
    "eastward_wind_u_coeff",
    "eastward_wind_v_coeff",
    "northward_wind_u_coeff",
    "northward_wind_v_coeff",
    "lat",
    "lon",
    "latb",
    "lonb",
    "land_sea_mask",
    "area",
]
ScalarMetrics = Dict[str, Mapping[str, float]]


def insert_scalar_metrics_r2(
    metrics: ScalarMetrics,
    predicted: Sequence[str],
    mse_coord: str = "mse",
    r2_coord: str = "r2",
    predict_coord: str = "predict",
    target_coord: str = "target",
):
    for var in predicted:
        r2_name = f"scalar/r2/{var}/predict_vs_target"
        mse = metrics[f"scalar/mse/{var}/predict_vs_target"]["mean"]
        variance = metrics[f"scalar/mse/{var}/mean_vs_target"]["mean"]
        # std is across batches
        mse_std = metrics[f"scalar/mse/{var}/predict_vs_target"]["std"]
        variance_std = metrics[f"scalar/mse/{var}/mean_vs_target"]["std"]
        r2 = 1.0 - (mse / variance)
        r2_std = r2 * np.sqrt((mse_std / mse) ** 2 + (variance_std / variance) ** 2)
        metrics[r2_name] = {"mean": r2, "std": r2_std}
    return metrics


def insert_dataset_r2(
    ds: xr.Dataset,
    mse_coord: str = "mse",
    r2_coord: str = "r2",
    predict_coord: str = "predict",
    target_coord: str = "target",
):
    mse_vars = [
        var
        for var in ds.data_vars
        if (
            str(var).endswith(f"{predict_coord}_vs_{target_coord}")
            and mse_coord in str(var)
        )
    ]
    for mse_var in mse_vars:
        name_pieces = str(mse_var).split("-")
        variance = "-".join(name_pieces[:-1] + [f"mean_vs_{target_coord}"])
        r2_var = "-".join([s if s != mse_coord else r2_coord for s in name_pieces])
        ds[r2_var] = 1.0 - ds[mse_var] / ds[variance]
    return ds


def mse_to_rmse(ds: xr.Dataset):
    # replaces MSE variables with RMSE after the weighted avg is calculated
    mse_vars = [var for var in ds.data_vars if "mse" in str(var)]
    for mse_var in mse_vars:
        rmse_var = str(mse_var).replace("mse", "rmse")
        ds[rmse_var] = np.sqrt(ds[mse_var])
    return ds.drop(mse_vars)


def load_grid_info(res: str = "c48"):
    grid = catalog[f"grid/{res}"].read()
    wind_rotation = catalog[f"wind_rotation/{res}"].read()
    land_sea_mask = catalog[f"landseamask/{res}"].read()
    grid_info = xr.merge([grid, wind_rotation, land_sea_mask])
    return safe.get_variables(grid_info, GRID_INFO_VARS)


def write_report(
    output_dir: str,
    title: str,
    sections: Mapping[str, Sequence[str]],
    metadata: report.Metadata = None,
    report_metrics: report.Metrics = None,
    html_header: str = None,
):
    filename = title.replace(" ", "_") + ".html"
    html_report = report.create_html(
        sections,
        title,
        metadata=metadata,
        metrics=report_metrics,
        html_header=html_header,
    )
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(html_report)


def open_diagnostics_outputs(
    data_dir,
    diagnostics_nc_name: str,
    diurnal_nc_name: str,
    transect_nc_name: str,
    metrics_json_name: str,
    config_name: str,
):
    with fsspec.open(os.path.join(data_dir, diagnostics_nc_name), "rb") as f:
        ds_diags = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, diurnal_nc_name), "rb") as f:
        ds_diurnal = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, transect_nc_name), "rb") as f:
        ds_transect = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, metrics_json_name), "r") as f:
        metrics = json.load(f)
    with fsspec.open(os.path.join(data_dir, config_name), "r") as f:
        config = yaml.safe_load(f)
    return ds_diags, ds_diurnal, ds_transect, metrics, config


def copy_outputs(temp_dir, output_dir):
    if output_dir.startswith("gs://"):
        gsutil.copy(temp_dir, output_dir)
    else:
        shutil.copytree(temp_dir, output_dir)


def tidy_title(var: str):
    title = (
        var.strip("pressure_level")
        .strip("zonal_avg_pressure_level")
        .strip("predict_vs_target")
        .strip("-")
        .replace("-", " ")
    )
    return title[0].upper() + title[1:]


def get_metric_string(
    metrics: Mapping[str, Mapping[str, float]],
    metric_name: str,
    var: str,
    predict_coord: str = "predict",
    target_coord: str = "target",
    precision=2,
):
    value = metrics[f"scalar/{metric_name}/{var}/{predict_coord}_vs_{target_coord}"][
        "mean"
    ]
    std = metrics[f"scalar/{metric_name}/{var}/{predict_coord}_vs_{target_coord}"][
        "std"
    ]
    return f"{value:.{precision}f} +/- {std:.{precision}f}"


def column_integrated_metric_names(metrics):
    names = set([key.split("/")[2] for key in metrics.keys()])
    return [name for name in names if "column_integrated" in name]


def units_from_Q_name(var):
    if "q1" in var.lower():
        if "column_integrated" in var:
            return "[W/m^2]"
        else:
            return "[K/s]"
    elif "q2" in var.lower():
        if "column_integrated" in var:
            return "[mm/day]"
        else:
            return "[kg/kg/s]"
    elif "qu" in var.lower() or "qv" in var.lower():
        if "column_integrated" in var:
            return "[Pa]"
        else:
            return "[m/s^2]"
    else:
        return None


def _shorten_coordinate_label(coord: str):
    # shortens and formats labels that get too long to display
    # in multi panel figures
    return (
        coord.replace("_", " ")
        .replace("average", "avg")
        .replace("precipitation", "precip")
        .replace("positive", "> 0")
        .replace("negative", "< 0")
    )


def net_precipitation_provenance_information(
    domain: xr.DataArray, derivation: str
) -> xr.Variable:
    # adds information about which data was used to determine pos/neg precip
    new_domain_coords = []
    for coord in np.asarray(domain):
        if "net_precip" in coord:
            new_domain_coords.append(
                _shorten_coordinate_label(coord) + f" ({derivation})"
            )
        else:
            new_domain_coords.append(_shorten_coordinate_label(coord))
    return xr.Variable(domain.dims, new_domain_coords)
