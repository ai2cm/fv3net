#!/usr/bin/env python

import json
from typing import Mapping, Iterable, Any
import yaml
import os
import xarray as xr
import fsspec
import pandas as pd
from pathlib import Path
import argparse
import holoviews as hv
from report import create_html, Plot
from bokeh.embed import components

hv.extension("bokeh")

units = {}

_bokeh_html_header = """
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-2.0.2.min.js" integrity="sha384-ufR9RFnRs6lniiaFvtJziE0YeidtAgBRH6ux2oUItHw5WTvE1zuk9uzhUU/FJXDp" crossorigin="anonymous"></script>
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.0.2.min.js" integrity="sha384-8QM/PGWBT+IssZuRcDcjzwIh1mkOmJSoNMmyYDZbCfXJg3Ap1lEvdVgFuSAwhb/J" crossorigin="anonymous"></script>
        <script type="text/javascript" src="https://unpkg.com/@holoviz/panel@^0.9.5/dist/panel.min.js" integrity="sha384-" crossorigin="anonymous"></script>
"""


class PlotManager:
    """An object for managing lists of plots in an extensible way

    New plotting functions can be registered using the ``register`` method.

    All plotting functions registered by the object will be called in sequence on 
    the data passed to `make_plots``.

    """

    def __init__(self):
        self._diags = []

    def register(self, func):
        """Register a given function as a diagnostic 

        This can be used to generate a new set of plots to appear the html reports
        """
        self._diags.append(func)
        return func

    def make_plots(self, data) -> Iterable[Plot]:
        for func in self._diags:
            yield func(data)


class HVPlot(Plot):
    """Renders holoviews plots to HTML for use in the diagnostic reports
    """

    def __init__(self, hvplot):
        self._plot = hvplot

    def render(self) -> str:
        # It took hours to find this combinitation of commands!
        # it was really hard finding a combintation that
        # 1. embedded the data for an entire HoloMap object
        # 2. exported the html as a div which can easily be embedded in the reports.
        r = hv.renderer("bokeh")
        html, _ = r.components(self._plot)
        html = html["text/html"]
        return html


def get_ts(ds):
    return ds.drop([key for key in ds if set(ds[key].dims) != {"time"}])


def convert_time_index_to_datetime(ds, dim):
    return ds.assign_coords({dim: ds.indexes[dim].to_datetimeindex()})


def detect_rundirs(bucket):
    fs = fsspec.filesystem("gs")
    diag_ncs = fs.glob(os.path.join(bucket, "*", "diags.nc"))
    if len(diag_ncs) == 0:
        raise ValueError(f"No diagnostic outputs detected at {bucket}")
    return [Path(url).parent.name for url in diag_ncs]


def load_diags(bucket):
    rundirs = detect_rundirs(bucket)
    metrics = {}
    for rundir in rundirs:
        path = os.path.join(bucket, rundir, "diags.nc")
        with fsspec.open(path, "rb") as f:
            metrics[rundir] = xr.open_dataset(f, engine="h5netcdf").compute()
    return metrics


def flatten(metrics):
    for run in metrics:
        for name in metrics[run]:
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
            units[name] = metrics[run][name]["units"]
            yield one_step, baseline, name, metrics[run][name]["value"]


def load_metrics(bucket):
    rundirs = detect_rundirs(bucket)
    metrics = {}
    for rundir in rundirs:
        path = os.path.join(bucket, rundir, "metrics.json")
        with fsspec.open(path, "rb") as f:
            metrics[rundir] = json.load(f)

    return pd.DataFrame(
        flatten(metrics), columns=["one_step", "baseline", "metric", "value"]
    )


def holomap_filter(time_series, varfilter):
    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["variable", "run"])
    for run, ds in time_series.items():
        for varname in ds:
            if varfilter in varname:
                try:
                    v = ds[varname]
                except:
                    pass
                else:
                    if run.endswith("baseline"):
                        style = "dashed"
                    else:
                        style = "solid"
                    long_name = ds[varname].long_name
                    hmap[(long_name, run)] = hv.Curve(v, label=varfilter).options(
                        line_dash=style, color=p
                    )
    return hmap.opts(norm={"framewise": True}, plot=dict(width=700, height=500))


# Initialize diagnostic managers
# diag_plot_manager will be passed the data from the diags.nc files
diag_plot_manager = PlotManager()
# this will be passed the data from the metrics.json files
metrics_plot_manager = PlotManager()


# Routines for plotting the "diagnostics"
@diag_plot_manager.register
def rms_plots(time_series: Mapping[str, xr.Dataset]) -> hv.HoloMap:
    return HVPlot(holomap_filter(time_series, varfilter="rms").overlay("run"))


@diag_plot_manager.register
def global_avg_plots(time_series: Mapping[str, xr.Dataset]) -> hv.HoloMap:
    return HVPlot(holomap_filter(time_series, varfilter="global_avg").overlay("run"))


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

    # load data
    diags = load_diags(args.input)
    diagnostics = {
        key: convert_time_index_to_datetime(get_ts(ds), "time")
        for key, ds in diags.items()
    }
    metrics = load_metrics(args.input)

    # generate all plots
    sections = {
        "Diagnostics": list(diag_plot_manager.make_plots(diagnostics)),
        "Metrics": list(metrics_plot_manager.make_plots(metrics)),
    }

    html = create_html(
        title="Prognostic run report", sections=sections, html_header=_bokeh_html_header
    )
    with fsspec.open(args.output, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
