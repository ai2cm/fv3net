#!/usr/bin/env python

from typing import Iterable
import os
import xarray as xr
import fsspec
import pandas as pd
import holoviews as hv

from fv3net.diagnostics.prognostic_run.computed_diagnostics import (
    ComputedDiagnosticsList,
    RunDiagnostics,
)

from report import create_html, Link
from report.holoviews import HVPlot, get_html_header
from .matplotlib import plot_2d_matplotlib, plot_cube_matplotlib

import logging

import warnings

warnings.filterwarnings(
    "ignore", message="Creating an ndarray from ragged nested sequences"
)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

logging.basicConfig(level=logging.INFO)

hv.extension("bokeh")


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


def plot_1d(
    run_diags: RunDiagnostics, varfilter: str, run_attr_name: str = "run",
) -> HVPlot:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    All matching diagnostics must be 1D."""
    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["variable", "run"])
    vars_to_plot = set(v for v in run_diags.variables if varfilter in v)
    for run in run_diags.runs:
        for varname in vars_to_plot:
            v = run_diags.get_variable(run, varname).rename("value")
            style = "solid" if run_diags.is_baseline(run) else "dashed"
            long_name = v.long_name
            hmap[(long_name, run)] = hv.Curve(v, label=varfilter).options(
                line_dash=style, color=p
            )
    return HVPlot(_set_opts_and_overlay(hmap))


def plot_1d_min_max_with_region_bar(
    run_diags: RunDiagnostics,
    varfilter_min: str,
    varfilter_max: str,
    run_attr_name: str = "run",
) -> HVPlot:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    All matching diagnostics must be 1D."""
    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["variable", "region", "run"])

    variables_to_plot = [v for v in run_diags.variables if varfilter_min in v]

    for run in run_diags.runs:
        for min_var in variables_to_plot:
            max_var = min_var.replace(varfilter_min, varfilter_max)
            vmin = run_diags.get_variable(run, min_var).rename("min")
            vmax = run_diags.get_variable(run, max_var).rename("max")
            style = "solid" if run_diags.is_baseline(run) else "dashed"
            long_name = vmin.long_name
            region = min_var.split("_")[-1]
            # Area plot doesn't automatically add correct y label
            ylabel = f'{vmin.attrs["long_name"]} {vmin.attrs["units"]}'
            hmap[(long_name, region, run)] = hv.Area(
                (vmin.time, vmin, vmax), label="Min/max", vdims=["y", "y2"]
            ).options(line_dash=style, color=p, alpha=0.6, ylabel=ylabel)
    return HVPlot(_set_opts_and_overlay(hmap))


def plot_1d_with_region_bar(
    run_diags: RunDiagnostics, varfilter: str, run_attr_name: str = "run"
) -> HVPlot:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    Region will be selectable through a drop-down bar. Region is assumed to be part of
    variable name after last underscore. All matching diagnostics must be 1D."""
    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["variable", "region", "run"])
    vars_to_plot = set(v for v in run_diags.variables if varfilter in v)
    for run in run_diags.runs:
        for varname in vars_to_plot:
            v = run_diags.get_variable(run, varname).rename("value")
            style = "solid" if run_diags.is_baseline(run) else "dashed"
            long_name = v.long_name
            region = varname.split("_")[-1]
            hmap[(long_name, region, run)] = hv.Curve(v, label=varfilter,).options(
                line_dash=style, color=p
            )
    return HVPlot(_set_opts_and_overlay(hmap))


def _set_opts_and_overlay(hmap, overlay="run"):
    return (
        hmap.opts(norm={"framewise": True}, plot=dict(width=850, height=500))
        .overlay(overlay)
        .opts(legend_position="right")
    )


def _parse_diurnal_component_fields(varname: str):

    # diags key format: diurn_component_<varname>_diurnal_<sfc_type>
    tokens = varname.split("_")
    short_varname = tokens[2]
    surface_type = tokens[-1]

    return short_varname, surface_type


def diurnal_component_plot(
    run_diags: RunDiagnostics,
    run_attr_name="run",
    diurnal_component_name="diurn_component",
) -> HVPlot:

    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["run", "surface_type", "short_varname"])

    for run in run_diags.runs:
        for varname in run_diags.variables:
            if diurnal_component_name in varname:
                v = run_diags.get_variable(run, varname).rename("value")
                short_vname, surface_type = _parse_diurnal_component_fields(varname)
                hmap[(run, surface_type, short_vname)] = hv.Curve(
                    v, label=diurnal_component_name
                ).options(color=p)
    return HVPlot(_set_opts_and_overlay(hmap, overlay="short_varname"))


# Initialize diagnostic managers
# following plot managers will be passed the data from the diags.nc files
timeseries_plot_manager = PlotManager()
zonal_mean_plot_manager = PlotManager()
hovmoller_plot_manager = PlotManager()
horizontal_maps_plot_manager = PlotManager()
zonal_pressure_plot_manager = PlotManager()
diurnal_plot_manager = PlotManager()

# this will be passed the data from the metrics.json files
metrics_plot_manager = PlotManager()


# Routines for plotting the "diagnostics"
@timeseries_plot_manager.register
def rms_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d(diagnostics, varfilter="rms_global")


@timeseries_plot_manager.register
def spatial_mean_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d_with_region_bar(diagnostics, varfilter="spatial_mean")


@timeseries_plot_manager.register
def spatial_minmax_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d_min_max_with_region_bar(
        diagnostics, varfilter_min="spatial_min", varfilter_max="spatial_max"
    )


@zonal_mean_plot_manager.register
def zonal_mean_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d(diagnostics, varfilter="zonal_and_time_mean")


@hovmoller_plot_manager.register
def zonal_mean_hovmoller_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_2d_matplotlib(
        diagnostics, "zonal_mean_value", dims=["time", "latitude"], cmap="viridis"
    )


@hovmoller_plot_manager.register
def zonal_mean_hovmoller_bias_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_2d_matplotlib(
        diagnostics, "zonal_mean_bias", dims=["time", "latitude"], cmap="RdBu_r",
    )


@horizontal_maps_plot_manager.register
def time_mean_map_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_cube_matplotlib(diagnostics, "time_mean_value")


@horizontal_maps_plot_manager.register
def time_mean_bias_map_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_cube_matplotlib(diagnostics, "time_mean_bias")


@zonal_pressure_plot_manager.register
def zonal_pressure_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_2d_matplotlib(
        diagnostics,
        "pressure_level_zonal_time_mean",
        dims=["latitude", "pressure"],
        cmap="viridis",
        yincrease=False,
        ylabel="Pressure [Pa]",
    )


@zonal_pressure_plot_manager.register
def zonal_pressure_bias_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_2d_matplotlib(
        diagnostics,
        "pressure_level_zonal_bias",
        dims=["latitude", "pressure"],
        cmap="RdBu_r",
        yincrease=False,
        ylabel="Pressure [Pa]",
    )


@diurnal_plot_manager.register
def diurnal_cycle_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d_with_region_bar(diagnostics, varfilter="diurnal")


@diurnal_plot_manager.register
def diurnal_cycle_component_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return diurnal_component_plot(diagnostics)


# Routines for plotting the "metrics"
# New plotting routines can be registered here.
@metrics_plot_manager.register
def time_mean_bias_metrics(metrics: pd.DataFrame) -> hv.HoloMap:
    return generic_metric_plot(metrics, "time_and_global_mean_bias")


@metrics_plot_manager.register
def rmse_time_mean_metrics(metrics: pd.DataFrame) -> hv.HoloMap:
    return generic_metric_plot(metrics, "rmse_of_time_mean")


@metrics_plot_manager.register
def rmse_3day_metrics(metrics: pd.DataFrame) -> hv.HoloMap:
    return generic_metric_plot(metrics, "rmse_3day")


@metrics_plot_manager.register
def drift_metrics(metrics: pd.DataFrame) -> hv.HoloMap:
    return generic_metric_plot(metrics, "drift")


def generic_metric_plot(metrics: pd.DataFrame, name: str) -> hv.HoloMap:
    hmap = hv.HoloMap(kdims=["metric"])
    bar_opts = dict(norm=dict(framewise=True), plot=dict(width=600))
    metrics_contains_name = False
    for metric in metrics.metric.unique():
        if metric.startswith(name):
            metrics_contains_name = True
            s = metrics[metrics.metric == metric]
            bars = hv.Bars((s.run, s.value), hv.Dimension("Run"), s.units.iloc[0])
            hmap[metric] = bars
    if metrics_contains_name:
        return HVPlot(hmap.opts(**bar_opts))


navigation = [
    Link("Home", "index.html"),
    Link("Latitude versus time hovmoller", "hovmoller.html"),
    Link("Time-mean maps", "maps.html"),
    Link("Time-mean zonal-pressure profiles", "zonal_pressure.html"),
]


def render_index(metadata, diagnostics, metrics, movie_links):
    sections_index = {
        "Links": navigation,
        "Timeseries": list(timeseries_plot_manager.make_plots(diagnostics)),
        "Zonal mean": list(zonal_mean_plot_manager.make_plots(diagnostics)),
        "Diurnal cycle": list(diurnal_plot_manager.make_plots(diagnostics)),
    }

    if not metrics.empty:
        sections_index["Metrics"] = list(metrics_plot_manager.make_plots(metrics))
    return create_html(
        title="Prognostic run report",
        metadata={**metadata, **render_links(movie_links)},
        sections=sections_index,
        html_header=get_html_header(),
    )


def render_hovmollers(metadata, diagnostics):
    sections_hovmoller = {
        "Links": navigation,
        "Zonal mean value and bias": list(
            hovmoller_plot_manager.make_plots(diagnostics)
        ),
    }
    return create_html(
        title="Latitude versus time hovmoller plots",
        metadata=metadata,
        sections=sections_hovmoller,
        html_header=get_html_header(),
    )


def render_horizontal_maps(metadata, diagnostics):
    sections = {
        "Links": navigation,
        "Time mean value and bias": list(
            horizontal_maps_plot_manager.make_plots(diagnostics)
        ),
    }
    return create_html(
        title="Time mean maps",
        metadata=metadata,
        sections=sections,
        html_header=get_html_header(),
    )


def render_zonal_pressures(metadata, diagnostics):
    sections_zonal_pressure = {
        "Links": navigation,
        "Zonal mean values at pressure levels": list(
            zonal_pressure_plot_manager.make_plots(diagnostics)
        ),
    }
    return create_html(
        title="Pressure versus latitude plots",
        metadata=metadata,
        sections=sections_zonal_pressure,
        html_header=get_html_header(),
    )


def _html_link(url, tag):
    return f"<a href='{url}'>{tag}</a>"


def render_links(link_dict):
    """Render links to html

    Args:
        link_dict: dict where keys are names, and values are lists (url,
            text_to_display). For example::

                {"column_moistening.mp4": [(url_to_qv, "specific humidity"), ...]}
    """
    return {
        key: " ".join([_html_link(url, tag) for (url, tag) in links])
        for key, links in link_dict.items()
    }


def register_parser(subparsers):
    parser = subparsers.add_parser("report", help="Generate a static html report.")
    parser.add_argument("input", help="Directory containing multiple run diagnostics.")
    parser.add_argument("output", help="Location to save report html files.")
    parser.set_defaults(func=main)


def main(args):

    computed_diagnostics = ComputedDiagnosticsList(url=args.input)
    metrics = computed_diagnostics.load_metrics()
    movie_links = computed_diagnostics.find_movie_links()
    metadata, diagnostics = computed_diagnostics.load_diagnostics()

    pages = {
        "index.html": render_index(metadata, diagnostics, metrics, movie_links),
        "hovmoller.html": render_hovmollers(metadata, diagnostics),
        "maps.html": render_horizontal_maps(metadata, diagnostics),
        "zonal_pressure.html": render_zonal_pressures(metadata, diagnostics),
    }

    for filename, html in pages.items():
        upload(html, os.path.join(args.output, filename))


if __name__ == "__main__":
    main()
