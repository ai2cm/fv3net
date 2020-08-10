#!/usr/bin/env python

import json
from typing import Iterable
import os
import xarray as xr
import fsspec
import pandas as pd
from pathlib import Path
import argparse
import holoviews as hv
from report import create_html
from report.holoviews import HVPlot, get_html_header

hv.extension("bokeh")
PUBLIC_GCS_DOMAIN = "https://storage.googleapis.com"


def upload(html: str, url: str, content_type: str = "text/html"):
    """Upload to a local or remote path, setting the content type if remote
    
    Setting the content type is necessary for viewing the uploaded object in a
    the web browser (e.g. it is a webpage or image).
    
    """
    with fsspec.open(url, "w") as f:
        f.write(html)

    if url.startswith("gs"):
        fs = fsspec.filesystem("gs")
        fs.setxattrs(url, content_type=content_type)


class PlotManager:
    """An object for managing lists of plots in an extensible way

    New plotting functions can be registered using the ``register`` method.

    All plotting functions registered by the object will be called in sequence on
    the data passed to `make_plots``.

    We could extend this class in the future to have even more features
    (e.g. parallel plot generation, exception handling, etc)

    """

    def __init__(self):
        self._diags = []

    def register(self, func):
        """Register a given function as a diagnostic

        This can be used to generate a new set of plots to appear the html reports
        """
        self._diags.append(func)
        return func

    def make_plots(self, data) -> Iterable:
        for func in self._diags:
            yield func(data)


def get_variables_with_dims(ds, dims):
    return ds.drop([key for key in ds if set(ds[key].dims) != set(dims)])


def convert_time_index_to_datetime(ds, dim):
    return ds.assign_coords({dim: ds.indexes[dim].to_datetimeindex()})


def detect_rundirs(bucket: str, fs: fsspec.AbstractFileSystem):
    diag_ncs = fs.glob(os.path.join(bucket, "*", "diags.nc"))
    if len(diag_ncs) < 2:
        raise ValueError(
            "Plots require more than 1 diagnostic directory in"
            f" {bucket} for holoviews plots to display correctly."
        )
    return [Path(url).parent.name for url in diag_ncs]


def load_diags(bucket, rundirs):
    metrics = {}
    for rundir in rundirs:
        path = os.path.join(bucket, rundir, "diags.nc")
        with fsspec.open(path, "rb") as f:
            metrics[rundir] = xr.open_dataset(f, engine="h5netcdf").compute()
    return metrics


def _yield_metric_rows(metrics):
    """yield rows to be combined into a dataframe
    """
    for run in metrics:
        for name in metrics[run]:
            yield {
                "run": run,
                "metric": name,
                "value": metrics[run][name]["value"],
                "units": metrics[run][name]["units"],
            }


def _parse_metadata(run: str):
    baseline_s = "-baseline"
    rf_s = "-rf"

    if run.endswith(baseline_s):
        baseline = "Baseline"
        one_step = run[: -len(baseline_s)]
    elif run.endswith(rf_s):
        one_step = run[: -len(rf_s)]
        baseline = "RF"
    else:
        one_step = run
        baseline = "misc"

    return {"run": run, "one_step": one_step, "baseline": baseline}


def load_metrics(bucket, rundirs):
    metrics = {}
    for rundir in rundirs:
        path = os.path.join(bucket, rundir, "metrics.json")
        with fsspec.open(path, "rb") as f:
            metrics[rundir] = json.load(f)

    return metrics


def get_movie_links(bucket, rundirs, fs, domain=PUBLIC_GCS_DOMAIN):
    movie_links = {}
    for rundir in rundirs:
        movie_paths = fs.glob(os.path.join(bucket, rundir, "*.mp4"))
        for gcs_path in movie_paths:
            movie_name = os.path.basename(gcs_path)
            if movie_name not in movie_links:
                movie_links[movie_name] = ""
            public_url = os.path.join(domain, gcs_path)
            movie_links[movie_name] += " " + _html_link(public_url, rundir)
    return movie_links


def _html_link(url, tag):
    return f"<a href='{url}'>{tag}</a>"


def holomap_filter(time_series, varfilter, run_attr_name="run"):
    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["variable", "run"])
    for ds in time_series:
        for varname in ds:
            if varfilter in varname:
                try:
                    v = ds[varname]
                except KeyError:
                    pass
                else:
                    if ds.attrs["baseline"] == "Baseline":
                        style = "solid"
                    else:
                        style = "dashed"

                    run = ds.attrs[run_attr_name]
                    long_name = ds[varname].long_name
                    hmap[(long_name, run)] = hv.Curve(v, label=varfilter).options(
                        line_dash=style, color=p
                    )
    return hmap.opts(norm={"framewise": True}, plot=dict(width=850, height=500))


def time_series_plot(time_series: Iterable[xr.Dataset], varfilter: str) -> HVPlot:
    return HVPlot(
        holomap_filter(time_series, varfilter=varfilter)
        .overlay("run")
        .opts(legend_position="right")
    )


def _parse_diurnal_component_fields(varname: str):

    # diags key format: diurn_comp_<varname>_diurnal_<sfc_type>
    tokens = varname.split("_")
    short_varname = tokens[2]
    surface_type = tokens[-1]

    return short_varname, surface_type


def diurnal_component_plot(
    time_series: Iterable[xr.Dataset],
    run_attr_name="run",
    diurnal_component_name="diurn_comp",
) -> HVPlot:

    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["run", "surface_type", "short_varname"])

    for ds in time_series:
        for varname in ds:
            if diurnal_component_name in varname:
                v = ds[varname]
                short_vname, surface_type = _parse_diurnal_component_fields(varname)
                run = ds.attrs[run_attr_name]
                hmap[(run, surface_type, short_vname)] = hv.Curve(
                    v, label=diurnal_component_name
                ).options(color=p)

    hmap = (
        hmap.opts(norm={"framewise": True}, plot=dict(width=850, height=500),)
        .overlay("short_varname")
        .opts(legend_position="right")
    )

    return HVPlot(hmap)


# Initialize diagnostic managers
# diag_plot_manager will be passed the data from the diags.nc files
diag_plot_manager = PlotManager()
# this will be passed the data from the metrics.json files
metrics_plot_manager = PlotManager()


# Routines for plotting the "diagnostics"
@diag_plot_manager.register
def rms_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return time_series_plot(time_series, varfilter="rms")


@diag_plot_manager.register
def global_avg_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return time_series_plot(time_series, varfilter="global_avg")


@diag_plot_manager.register
def global_avg_physics_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return time_series_plot(time_series, varfilter="global_phys_avg")


@diag_plot_manager.register
def diurnal_cycle_land_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return time_series_plot(time_series, varfilter="diurnal_land")


@diag_plot_manager.register
def diurnal_cycle_sea_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return time_series_plot(time_series, varfilter="diurnal_sea")


@diag_plot_manager.register
def diurnal_cycle_component_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return diurnal_component_plot(time_series)


# Routines for plotting the "metrics"
# New plotting routines can be registered here.
@metrics_plot_manager.register
def rmse_metrics(metrics: pd.DataFrame) -> hv.HoloMap:
    hmap = hv.HoloMap(kdims=["metric"])
    bar_opts = dict(norm=dict(framewise=True), plot=dict(width=600))
    for metric in metrics.metric.unique():
        s = metrics[metrics.metric == metric]
        bars = hv.Bars((s.one_step, s.baseline, s.value), kdims=["one_step", "type"])
        if metric.startswith("rmse"):
            hmap[metric] = bars
    return HVPlot(hmap.opts(**bar_opts))


@metrics_plot_manager.register
def bias_metrics(metrics: pd.DataFrame) -> hv.HoloMap:
    hmap = hv.HoloMap(kdims=["metric"])
    bar_opts = dict(norm=dict(framewise=True), plot=dict(width=600))
    for metric in metrics.metric.unique():
        s = metrics[metrics.metric == metric]
        bars = hv.Bars((s.one_step, s.baseline, s.value), kdims=["one_step", "type"])
        if metric.startswith("drift"):
            hmap[metric] = bars
    return HVPlot(hmap.opts(**bar_opts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()
    bucket = args.input

    # get run information
    fs = fsspec.filesystem("gs")
    rundirs = detect_rundirs(bucket, fs)
    run_table = pd.DataFrame.from_records(_parse_metadata(run) for run in rundirs)
    run_table_lookup = run_table.set_index("run")

    # load diagnostics
    diags = load_diags(bucket, rundirs)
    dims = ["time", "local_time"]  # keep all vars that have only these dimensions
    diagnostics = [
        xr.merge([get_variables_with_dims(ds, [dim]) for dim in dims]).assign_attrs(
            run=key, **run_table_lookup.loc[key]
        )
        for key, ds in diags.items()
    ]
    diagnostics = [convert_time_index_to_datetime(ds, "time") for ds in diagnostics]

    # load metrics
    nested_metrics = load_metrics(bucket, rundirs)
    metric_table = pd.DataFrame.from_records(_yield_metric_rows(nested_metrics))
    metrics = pd.merge(run_table, metric_table, on="run")

    # generate all plots
    sections = {
        "Diagnostics": list(diag_plot_manager.make_plots(diagnostics)),
        "Metrics": list(metrics_plot_manager.make_plots(metrics)),
    }

    # get metadata
    run_urls = {key: ds.attrs["url"] for key, ds in diags.items()}
    movie_links = get_movie_links(bucket, rundirs, fs)

    html = create_html(
        title="Prognostic run report",
        metadata={**run_urls, **movie_links},
        sections=sections,
        html_header=get_html_header(),
    )
    upload(html, args.output, content_type="text/html")


if __name__ == "__main__":
    main()
