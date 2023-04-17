import xarray as xr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    c48_real_all: xr.Dataset = (
        xr.open_zarr(
            "gs://vcm-ml-experiments/mcgibbon/2023-03-29/coarse-combined.zarr"
        ).rename({"grid_xt": "x", "grid_yt": "y"})
    )
    # # plot the c48 histograms of PRATEsfc for each year, labelled in the legend
    # # by year
    # plt.figure()
    # for i in range(0, len(c48_real_all.time) - 365*8, 365*8):
    #     print(f"plotting year {i // (365*8)}")
    #     c48_real: xr.Dataset = c48_real_all.isel(time=slice(i, i+365*8))
    #     plt.hist(
    #         c48_real["PRATEsfc"].values.flatten(),
    #         histtype="step",
    #         bins=100,
    #         label=i // (365*8),
    #         alpha=0.5,
    #         density=True
    #     )
    #     plt.yscale("log")
    # plt.xlabel("PRATEsfc")
    # plt.ylabel("count")
    # plt.legend()
    # plt.savefig("c48_histograms.png")
    C384_I_TRAIN_END = 14600
    plt.figure()
    c48_real: xr.Dataset = c48_real_all.isel(time=slice(0, C384_I_TRAIN_END))
    plt.hist(
        c48_real["PRATEsfc"].values.flatten(),
        histtype="step",
        bins=100,
        label="train",
        alpha=0.5,
        density=True,
    )
    c48_real = c48_real_all.isel(time=slice(C384_I_TRAIN_END, None))
    plt.hist(
        c48_real["PRATEsfc"].values.flatten(),
        histtype="step",
        bins=100,
        label="val",
        alpha=0.5,
        density=True,
    )
    plt.yscale("log")
    plt.xlabel("PRATEsfc")
    plt.ylabel("count")
    plt.legend()
    plt.savefig("c48_histograms.png")
