import matplotlib.pyplot as plt
from matplotlib import dates

import vcm.catalog
import vcm
import common
import os


os.makedirs("figs", exist_ok=True)


@common.memoize_xarray_out
def get_global_avg(group, field):
    _, data_3d = common.open_group(group)
    # data is already on pressure levels
    return vcm.weighted_average(data_3d[field], data_3d.area).load().to_dataset()


def figure_global():
    def label(ax):
        ax.set_ylabel("pressure (mb)")
        ax.set_xlabel("July 2016")
        locator = dates.AutoDateLocator()
        formatter = dates.DateFormatter("%d")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    def avg(x):
        return x

    M = 30

    group = "precpd-diff-only-rnn-combined-cloudlimit-v5-online"
    cloud_avg = get_global_avg(group, "cloud_water_mixing_ratio")

    group = "precpd-diff-only-rnn-combined-cloudlimit-v5-offline"
    cloud_avg_offline = get_global_avg(group, "cloud_water_mixing_ratio")

    bias = cloud_avg - cloud_avg_offline

    plt.figure(figsize=(common.WIDTH / 2, common.WIDTH / 2 / 1.61))
    z = avg(cloud_avg_offline.cloud_water_mixing_ratio)
    z["pressure"] = z["pressure"] / 100
    z *= 1e6
    z.plot(
        y="pressure",
        cmap=plt.get_cmap("Blues"),
        vmax=M,
        vmin=0,
        yincrease=False,
        add_labels=False,
        rasterized=True,
    )
    plt.title("a) Truth", loc="left")
    label(plt.gca())
    common.savefig("global-a")

    plt.figure(figsize=(common.WIDTH / 2, common.WIDTH / 2 / 1.61))
    z = avg(bias.cloud_water_mixing_ratio)
    z["pressure"] = z["pressure"] / 100
    z = z * 1e6
    z.plot(
        y="pressure",
        yincrease=False,
        vmax=10,
        vmin=-10,
        cmap=plt.get_cmap("RdBu"),
        add_labels=False,
        rasterized=True,
    )
    plt.title("b) Bias", loc="left")
    label(plt.gca())
    common.savefig("global-b")


figure_global()
