# flake8: noqa
"""Configurations settings
"""
import matplotlib.pyplot as plt
from cycler import cycler


# Matplotlib styles
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
plt.rcParams["axes.prop_cycle"] = cycler("color", wong_palette)


# ERROR SCALE
ERROR_SCALE_OUTPUT_DIR = "figs/error-scale"
ERROR_SCALE_MODELS = [
    "gs://vcm-ml-experiments/microphysics-emulation/2022-01-18/rnn-predict-gscond-3f77ec",
    "gs://vcm-ml-experiments/microphysics-emulation/2022-01-27/rnn-gscond-cloudtdep-cbfc4a",
    "gs://vcm-ml-experiments/microphysics-emulation/2022-01-27/rnn-gscond-alltdep-c9af46",
]
