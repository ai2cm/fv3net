import matplotlib.pyplot as plt
from cycler import cycler


# https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
WONG_PALLETE = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


def use_colorblind_friendly_style():
    plt.rcParams["axes.prop_cycle"] = cycler("color", WONG_PALLETE)
