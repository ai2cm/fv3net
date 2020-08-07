import fsspec
import json
import os
import shutil
from typing import Any, Mapping, Sequence
import xarray as xr

import report
from vcm.cloud import gsutil


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
    data_dir, diagnostics_nc_name: str, diurnal_nc_name: str, metrics_json_name: str,
):
    with fsspec.open(os.path.join(data_dir, diagnostics_nc_name), "rb") as f:
        ds_diags = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, diurnal_nc_name), "rb") as f:
        ds_diurnal = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, metrics_json_name), "r") as f:
        metrics = json.load(f)
    return ds_diags, ds_diurnal, metrics


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
    else:
        return None


def shield_data_included(
    config: Mapping[str, Any], shield_kwarg: str = "shield_diags_url"
):
    # checks all keys because this arg could be in config.mapping_kwargs top level
    # or in a deeper kwargs dict within that level
    def recursive_items(dictionary):
        # https://stackoverflow.com/questions/39233973/get-all-keys-of-a-nested-dictionary
        for key, value in dictionary.items():
            if type(value) is dict:
                yield (key)
                yield from recursive_items(value)
            else:
                yield (key)

    return shield_kwarg in recursive_items(config)
