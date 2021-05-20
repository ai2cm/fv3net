import sys
import streamlit as st
import vcm.fv3.logs
from vcm.cloud.fsspec import get_fs
from toolz.curried import map, compose, reduce
import pandas as pd
import plotly.express as px
import subprocess


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


def concat_logs(a, b):
    return vcm.fv3.logs.FV3Log(
        dates=a.dates + b.dates,
        totals={key: a.totals[key] + b.totals[key] for key in b.totals},
        ranges=None,
    )


def open_segmented_logs(url):
    fs = get_fs(url)
    logfiles = sorted(fs.glob(f"{url}/**/logs.txt"))
    parsed = reduce(
        concat_logs, map(compose(vcm.fv3.logs.loads, bytes.decode, fs.cat), logfiles)
    )

    df = pd.DataFrame({species: parsed.totals[species] for species in parsed.totals})
    df.index = parsed.dates
    return df


def line_chart(df, url):
    df = pd.melt(df.reset_index(), id_vars=["index"])
    fig = px.line(df, x="index", y="value", color="variable", title=url).update_layout(
        font=dict(size=10)
    )
    return st.plotly_chart(fig)


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "log-viewer",
        help="Webapp for plotting time series information from the "
        "standard output of segmented FV3 runs.",
    )
    parser.set_defaults(func=run_streamlit)


def run_streamlit(args):
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", __file__])


def main():
    px.defaults.color_discrete_sequence = wong_palette
    st.title("Prognostic Run Log Viewer")

    url = st.text_input(
        "enter url",
        "gs://vcm-ml-experiments/default/2021-05-19/n2f-3km-timescale-3hr-tuned-mp-v1/fv3gfs_run",  # noqa: E501
    )
    df = open_segmented_logs(url)
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


if __name__ == "__main__":
    main()
