#!/usr/bin/env python

import json
import yaml
import os
import xarray as xr
import fsspec
import pandas as pd
from pathlib import Path
import argparse
import holoviews as hv
from report import create_html, HTMLPlot, Plot

hv.extension("bokeh")

units = {}


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


from bokeh.embed import components

section = []

class HVPlot(HTMLPlot):
    """Renderer for holoviews plot"""
    def __init__(self, hvplot):
        self._plot = hvplot
    
    def render(self):
        # I spent hours trying to find this combination of lines
        r = hv.renderer('bokeh')
        html, _ = r.components(self._plot)
        html = html['text/html']
        return html



parser = argparse.ArgumentParser()
parser.add_argument("input")

args = parser.parse_args()

# BUCKET = os.getenv("INPUT", "gs://vcm-ml-data/experiments-2020-03/prognostic_run_diags")

diags = load_diags(args.input)

time_series = {
    key: convert_time_index_to_datetime(get_ts(ds), "time") for key, ds in diags.items()
}
renderer = hv.renderer("bokeh")

hmap = holomap_filter(time_series, varfilter="rms").overlay("run")
save(hmap)

hmap = holomap_filter(time_series, "global_avg").overlay("run")
save(hmap)

# metrics plots
df = load_metrics(args.input)

bar_opts = dict(norm=dict(framewise=True), plot=dict(width=600))

# collect data into a holoviews object
hmap = hv.HoloMap(kdims=["metric"])
bias = hv.HoloMap(kdims=["metric"])

for metric in df.metric.unique():
    s = df[df.metric == metric]
    bars = hv.Bars((s.one_step, s.baseline, s.value), kdims=["one_step", "type"])

    if metric.startswith("rmse"):
        hmap[metric] = bars
    elif metric.startswith("drift"):
        bias[metric] = bars


save(hmap.opts(**bar_opts))
save(bias.opts(**bar_opts))

sections = {"Diagnostics": section}

header = """
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-2.0.2.min.js" integrity="sha384-ufR9RFnRs6lniiaFvtJziE0YeidtAgBRH6ux2oUItHw5WTvE1zuk9uzhUU/FJXDp" crossorigin="anonymous"></script>
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.0.2.min.js" integrity="sha384-8QM/PGWBT+IssZuRcDcjzwIh1mkOmJSoNMmyYDZbCfXJg3Ap1lEvdVgFuSAwhb/J" crossorigin="anonymous"></script>
        <script type="text/javascript" src="https://unpkg.com/@holoviz/panel@^0.9.5/dist/panel.min.js" integrity="sha384-" crossorigin="anonymous"></script>
"""

html = create_html(title="Prognostic run report", sections=sections, html_header=header)
with open("index.html", "w") as f:
    f.write(html)
