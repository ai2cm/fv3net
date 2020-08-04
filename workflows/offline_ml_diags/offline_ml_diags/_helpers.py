import fsspec
import json
import os
import shutil
from typing import Mapping, Union, Sequence
import xarray as xr

from report import create_html
from vcm.cloud import gsutil


def _write_report(
        output_dir: str,
        title: str,
        sections: Mapping[str, Sequence[str]],
        metadata: Mapping[str, Union[str, float, int, bool]] = None,
        metrics: Mapping[str, Mapping[str, Union[str, float]]] = None,
        html_header: str = None,
):
    filename = title.replace(" ", "_") + ".html"
    html_report = create_html(
        sections, title, metadata=metadata, metrics=metrics, html_header=html_header)
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(html_report)


def _open_diagnostics_outputs(
    data_dir,
    diagnostics_nc_name: str,
    diurnal_nc_name: str,
    metrics_json_name: str,
):
    with fsspec.open(os.path.join(data_dir, diagnostics_nc_name), "rb") as f:
        ds_diags = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, diurnal_nc_name), "rb") as f:
        ds_diurnal = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, metrics_json_name), "r") as f:
        metrics = json.load(f)
    return ds_diags, ds_diurnal, metrics


def _copy_outputs(temp_dir, output_dir):
    if output_dir.startswith("gs://"):
        gsutil.copy(temp_dir, output_dir)
    else:
        shutil.copytree(temp_dir, output_dir)


def _tidy_title(var: str):
    title = (var
        .strip("pressure_level")
        .strip("predict_vs_target")
        .strip("-")
        .replace("-", " "))
    return title[0].upper() + title[1:]


def _get_metric_string(
    metrics: Mapping[str, Mapping[str, float]],
    metric_name: str,
    var: str,
    predict_coord: str = "predict",
    target_coord: str = "target",
    precision=2,
):
    value = metrics[f"scalar/{metric_name}/{var}/{predict_coord}_vs_{target_coord}"]["mean"]
    std = metrics[f"scalar/{metric_name}/{var}/{predict_coord}_vs_{target_coord}"]["std"]
    return f"{value:.{precision}f} +/- {std:.{precision}f}"
