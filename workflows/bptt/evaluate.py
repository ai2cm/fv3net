import matplotlib.pyplot as plt
import argparse
import numpy as np
import fv3fit
import vcm
import xarray as xr


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arrays_dir", type=str, help="directory containing TrainingArrays data"
    )
    parser.add_argument(
        "model_dir", type=str, help="directory containing trained model"
    )
    return parser


def get_vmin_vmax(*arrays):
    vmin = min(np.min(a) for a in arrays)
    vmax = max(np.max(a) for a in arrays)
    return vmin, vmax


if __name__ == "__main__":
    timestep_seconds = 3 * 60 * 60
    parser = get_parser()
    args = parser.parse_args()

    model = fv3fit.load(args.model_dir)

    fs = vcm.get_fs(args.arrays_dir)
    filename = sorted(fs.listdir(args.arrays_dir, detail=False))[0]
    print(filename)
    with open(filename, "rb") as f:
        ds = xr.open_dataset(filename)#.isel(sample=slice(0, 64))
        ds.load()

    state_out = model.integrate_stepwise(ds)

    def plot_single(predicted, reference, label, ax):
        vmin, vmax = get_vmin_vmax(predicted, reference)
        im = ax[0].pcolormesh(predicted.T, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title(f"predicted {label}")
        im = ax[1].pcolormesh(reference.T, vmin=vmin, vmax=vmax)
        ax[1].set_title(f"reference {label}")
        plt.colorbar(im, ax=ax[1])

    seconds_in_day = 60 * 60 * 24

    print("stds")
    print(np.std(state_out["air_temperature"] - state_out["air_temperature_reference"]))
    print(np.mean(np.var(state_out["air_temperature_reference"], axis=(0, 1))) ** 0.5)
    print(
        np.std(
            state_out["specific_humidity"] - state_out["specific_humidity_reference"]
        )
    )
    print(np.mean(np.var(state_out["specific_humidity_reference"], axis=(0, 1))) ** 0.5)

    lat = ds["lat"].isel(time=0).values
    lon = ds["lon"].isel(time=0).values

    antarctica_idx = np.argwhere(
        (15 + 180. < lon < 60. + 180.) &
        (-82 < lat < -75)
    )

    for i in antarctica_idx[:4]:
        fig, ax = plt.subplots(4, 2, figsize=(12, 8))
        print(ax.shape)
        plot_single(
            state_out["dQ1"][i, :, :].values * seconds_in_day,
            state_out["air_temperature_tendency_due_to_nudging"][i, :, :].values
            * seconds_in_day,
            "air_temperature (K/day)",
            ax[:2, 0],
        )
        plot_single(
            state_out["air_temperature"][i, :, :].values,
            state_out["air_temperature_reference"][i, :, :].values,
            "air_temperature (K)",
            ax[2:, 0],
        )
        plot_single(
            state_out["dQ2"][i, :, :].values * seconds_in_day,
            state_out["specific_humidity_tendency_due_to_nudging"][i, :, :].values
            * seconds_in_day,
            "specific_humidity (kg/kg/day)",
            ax[:2, 1],
        )
        plot_single(
            state_out["specific_humidity"][i, :, :].values,
            state_out["specific_humidity_reference"][i, :, :].values,
            "specific_humidity (kg/kg)",
            ax[2:, 1],
        )
        lat = ds["lat"][i, 0].values.item() * 180.0 / np.pi
        lon = ds["lon"][i, 0].values.item() * 180.0 / np.pi
        plt.suptitle(f"lat: {lat}, lon: {lon}")
        for a in ax.flatten():
            a.set_ylim(a.get_ylim()[::-1])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
