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
