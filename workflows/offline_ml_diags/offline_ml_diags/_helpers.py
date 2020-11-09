import fsspec
import intake
import json
import os
import shutil
from typing import Mapping, Sequence
import yaml
import xarray as xr
import report
from vcm import safe
from vcm.cloud import gsutil


GRID_INFO_VARS = [
    "eastward_wind_u_coeff",
    "eastward_wind_v_coeff",
    "northward_wind_u_coeff",
    "northward_wind_v_coeff",
    "lat",
    "lon",
    "land_sea_mask",
    "area",
]


def load_grid_info(catalog_path: str = "catalog.yml", res: str = "c48"):
    cat = intake.open_catalog(catalog_path)
    grid = cat[f"grid/{res}"].read()
    wind_rotation = cat[f"wind_rotation/{res}"].read()
    land_sea_mask = cat[f"landseamask/{res}"].read()
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
    metrics_json_name: str,
    config_name: str,
):
    with fsspec.open(os.path.join(data_dir, diagnostics_nc_name), "rb") as f:
        ds_diags = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, diurnal_nc_name), "rb") as f:
        ds_diurnal = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, metrics_json_name), "r") as f:
        metrics = json.load(f)
    with fsspec.open(os.path.join(data_dir, config_name), "r") as f:
        config = yaml.safe_load(f)
    return ds_diags, ds_diurnal, metrics, config


def copy_outputs(temp_dir, output_dir):
    if output_dir.startswith("gs://"):
        gsutil.copy(temp_dir, output_dir)
    else:
        shutil.copytree(temp_dir, output_dir)


def tidy_title(var: str):
    title = (
        var.strip("pressure_level")
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


def add_net_precip_domain_info(ds: xr.Dataset, domain_source: str):
    # adds information about which data was used to determine pos/neg precip
    new_domain_coords = []
    for coord in ds["domain"].values:
        if "net_precip" in coord:
            new_domain_coords.append(
                _shorten_coordinate_label(coord) + f" ({domain_source})"
            )
        else:
            new_domain_coords.append(_shorten_coordinate_label(coord))
    ds.coords["domain"] = new_domain_coords
