from typing import Tuple
import xarray as xr
import matplotlib.pyplot as plt
import fv3fit
import os
from fv3fit.pytorch import DEVICE

# for "march" data and for multi-climate "march" data
C48_I_TRAIN_END = 14600
C384_I_TRAIN_END = 14600

TO_MM_DAY = 86400 / 0.997


def predict_dataset(
    c48_real: xr.Dataset, c384_real: xr.Dataset, cyclegan: fv3fit.pytorch.CycleGAN
) -> Tuple[xr.Dataset, xr.Dataset]:
    c48_real = c48_real[cyclegan.state_variables]
    c384_real = c384_real[cyclegan.state_variables]
    c384_list = []
    c48_list = []
    for i in range(len(c384_real.perturbation)):
        c384_list.append(cyclegan.predict(c48_real.isel(perturbation=i)))
        c48_list.append(cyclegan.predict(c384_real.isel(perturbation=i), reverse=True))
    c384_gen = xr.concat(c384_list, dim="perturbation")
    c48_gen = xr.concat(c48_list, dim="perturbation")
    return c48_gen, c384_gen


class Aggregator:
    def __init__(self):
        self._data_list = []

    def add(
        self,
        c48_real: xr.Dataset,
        c384_real: xr.Dataset,
        c48_gen: xr.Dataset,
        c384_gen: xr.Dataset,
    ):
        c48_mean = c48_real.mean(dim=["tile", "x", "y"])
        c384_mean = c384_real.mean(dim=["tile", "x", "y"])
        c48_gen_mean = c48_gen.mean(dim=["tile", "x", "y"])
        c384_gen_mean = c384_gen.mean(dim=["tile", "x", "y"])
        c48 = xr.concat([c48_mean, c48_gen_mean], dim="source").assign_coords(
            source=["real", "gen"]
        )
        c384 = xr.concat([c384_mean, c384_gen_mean], dim="source").assign_coords(
            source=["real", "gen"]
        )
        data = xr.concat([c48, c384], dim="grid").assign_coords(grid=["C48", "C384"])
        self._data_list.append(data)

    def get_dataset(self) -> xr.Dataset:
        ds = xr.concat(self._data_list, dim="time")
        return ds


def plot(ds: xr.Dataset):
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    ax[0].plot(
        ds.PRATEsfc.sel(source="real", grid="C48").values.flatten() * TO_MM_DAY,
        ds.PRATEsfc.sel(source="gen", grid="C384").values.flatten() * TO_MM_DAY,
        ".",
        alpha=0.05,
    )
    ax[0].set_xlabel("C48 real")
    ax[0].set_ylabel("C384 ML")
    ax[0].set_title("Global Precipitation (mm/day)")
    ax[1].plot(
        ds.PRATEsfc.sel(source="real", grid="C384").values.flatten() * TO_MM_DAY,
        ds.PRATEsfc.sel(source="gen", grid="C48").values.flatten() * TO_MM_DAY,
        ".",
        alpha=0.05,
    )
    ax[1].set_xlabel("C384 real")
    ax[1].set_ylabel("C48 ML")
    ax[1].set_title("Global Precipitation (mm/day)")
    plt.tight_layout()
    fig.savefig(f"global-precip.png", dpi=100)
    plt.close("all")


if __name__ == "__main__":
    fv3fit.set_random_seed(0)
    CHECKPOINT_PATH = "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/"
    EVALUATE_ON_TRAIN = False
    for BASE_NAME, label, EPOCH in [
        # multi-climate new-data models
        ("20230329-221949-9d8e8abc", "prec-lr-1e-4-decay-0.63096-full", 16),
        # ("20230330-174749-899f5c19", "prec-lr-1e-5-decay-0.63096-full", 23),
        # ("20230424-183552-125621da", "prec-lr-1e-4-decay-0.63096-full-no-geo-features", 16),  # noqa: E501
        # ("20230424-191937-253268fd", "prec-lr-1e-4-decay-0.63096-full-no-identity-loss", 16),  # noqa: E501
        # ("20230427-160655-b8b010ce", "prec-lr-1e-4-decay-0.63096-1-year", 16),
    ]:
        label = label + f"-e{EPOCH:02d}"
        fv3fit.set_random_seed(0)
        print(f"Loading {BASE_NAME} epoch {EPOCH}")
        cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load(
            # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230130-231729-82b939d9-epoch_075/"  # precip-only  # noqa: E501
            # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230202-233100-c5d574a4-epoch_045/"  # precip-only, properly normalized  # noqa: E501
            # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230208-183103-cdda934c-epoch_017/"  # precip-only, properly normalized, +45 epochs  # noqa: E501
            # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230217-220447-a693c405-epoch_040/"  # lr=2e-6, xyz features  # noqa: E501
            CHECKPOINT_PATH
            + BASE_NAME
            + f"-epoch_{EPOCH:03d}/"
        ).to(DEVICE)
        VARNAME = "PRATEsfc"
        initial_time = xr.open_zarr(
            "gs://vcm-ml-experiments/mcgibbon/2023-03-29/fine-combined.zarr"
        )["time"][0].values

        if EVALUATE_ON_TRAIN:
            BASE_NAME = "train-" + BASE_NAME
            label = "train-" + label

        PROCESSED_FILENAME = f"./processed/processed-global-{BASE_NAME}-e{EPOCH}.nc"
        # label = "subset_5-" + label

        if not os.path.exists(PROCESSED_FILENAME):
            print(f"Calculating processed data for {PROCESSED_FILENAME}")
            c384_real_all: xr.Dataset = (
                xr.open_zarr(
                    "gs://vcm-ml-experiments/mcgibbon/2023-03-29/fine-combined.zarr"
                ).rename({"grid_xt": "x", "grid_yt": "y"})
            )
            c48_real_all: xr.Dataset = (
                xr.open_zarr(
                    "gs://vcm-ml-experiments/mcgibbon/2023-03-29/coarse-combined.zarr"
                ).rename({"grid_xt": "x", "grid_yt": "y"})
            )
            if not EVALUATE_ON_TRAIN:
                c48_real_in = c48_real_all.isel(
                    time=slice(C48_I_TRAIN_END, None)
                ).transpose(..., "x", "y")
                c384_real_in = c384_real_all.isel(
                    time=slice(C384_I_TRAIN_END, None)
                ).transpose(..., "x", "y")
            else:
                c48_real_in = c48_real_all.isel(
                    time=slice(None, C48_I_TRAIN_END)
                ).transpose(..., "x", "y")
                c384_real_in = c384_real_all.isel(
                    time=slice(None, C384_I_TRAIN_END)
                ).transpose(..., "x", "y")
            aggregator = Aggregator()
            nt_bin = 5 * 8
            nt_final = len(c48_real_in.time) // nt_bin * nt_bin
            for i_time in range(0, nt_final, nt_bin):
                print(f"Predicting {i_time} / {len(c48_real_in.time)}")
                c48_real = c48_real_in.isel(time=slice(i_time, i_time + nt_bin))
                c384_real = c384_real_in.isel(time=slice(i_time, i_time + nt_bin))
                c48_gen, c384_gen = predict_dataset(c48_real, c384_real, cyclegan,)
                aggregator.add(c48_real, c384_real, c48_gen, c384_gen)
                ds = aggregator.get_dataset()
                ds = ds.assign_coords(perturbation=c48_real_in.perturbation)
                plot(ds)
            ds = aggregator.get_dataset()
            ds = ds.assign_coords(perturbation=c48_real_in.perturbation)
            ds.to_netcdf(PROCESSED_FILENAME)
        else:
            print(f"Loading processed data from {PROCESSED_FILENAME}")
            ds = xr.open_dataset(PROCESSED_FILENAME)
        plot(ds)
