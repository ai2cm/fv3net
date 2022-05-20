import matplotlib.pyplot as plt
from cycler import cycler


# adapted from https://davidmathlogic.com/colorblind
wong_palette = [
    "#56B4E9",
    "#E69F00",
    "#009E73",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#F0E442",  # put yellow last, remove black
]


def use_colorblind_friendly_style():
    plt.rcParams["axes.prop_cycle"] = cycler("color", wong_palette)
