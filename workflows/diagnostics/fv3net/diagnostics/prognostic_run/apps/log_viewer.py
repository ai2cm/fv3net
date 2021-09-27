import sys
from fv3net.diagnostics.prognostic_run.load_run_data import (
    open_segmented_stats,
    open_segmented_logs,
)
import streamlit as st
import vcm.fv3.logs
import pandas as pd
import plotly.express as px
import subprocess


open_segmented_stats = st.cache(open_segmented_stats)
open_segmented_logs = st.cache(open_segmented_logs)

# https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
wong_palette = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


def plotly_chart(fig):
    """Apply common themes to a plotly chart and return as streamlit"""
    return st.plotly_chart(fig.update_layout(font=dict(size=10)))


def line_chart(df, url):
    df = pd.melt(df.reset_index(), id_vars=["index"])
    fig = px.line(df, x="index", y="value", color="variable", title=url)
    return plotly_chart(fig)


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "log-viewer",
        help="Webapp for plotting time series information from the "
        "standard output of segmented FV3 runs.",
    )
    parser.set_defaults(func=run_streamlit)


def view_stats(stats: pd.DataFrame, url: str):
    stats_no_time = set(stats.columns) - {"time"}
    st.header("Statistics.txt")
    variable = st.radio("variable", sorted(stats_no_time))
    return plotly_chart(px.line(stats, x="time", y=variable, title=url))


def view_logs(log: vcm.fv3.logs.FV3Log, url: str):
    df = pd.DataFrame({species: log.totals[species] for species in log.totals})
    df.index = log.dates
    tracers = [
        "total cloud water",
        "total cloud ice",
        "total snow",
        "total graupel",
        "total rain water",
    ]
    line_chart(df[tracers], url)
    line_chart(df["total water vapor"], url)
    line_chart(df[["total surface pressure", "mean dry surface pressure"]], url)


def run_streamlit(args):
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", __file__])


def main():
    px.defaults.color_discrete_sequence = wong_palette
    st.title("Prognostic Run Log Viewer")

    url = st.text_input(
        "enter url",
        "gs://vcm-ml-experiments/default/2021-05-19/n2f-3km-timescale-3hr-tuned-mp-v1/fv3gfs_run",  # noqa: E501
    )
    log = open_segmented_logs(url)
    view_logs(log, url)

    stats = open_segmented_stats(url)
    view_stats(stats, url)


if __name__ == "__main__":
    main()
