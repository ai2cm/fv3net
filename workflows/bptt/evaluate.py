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
        ds = xr.open_dataset(filename)
        # lat = ds["lat"].isel(time=0).values
        # lon = ds["lon"].isel(time=0).values
        # antarctica_idx = np.argwhere(
        #     np.logical_and(
        #         (195. * np.pi / 180. < lon) & (lon < 240. * np.pi / 180.),
        #         (-82 * np.pi / 180. < lat) & (lat < -75 * np.pi / 180.)
        #     )
        # )
        # print(f"{len(antarctica_idx)} antarctica samples found")
        # assert len(antarctica_idx) > 0
        # ds = ds.isel(sample=list(antarctica_idx.flatten()))
        # ds = ds.isel(sample=slice(0, 64))
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

    def get_r2(predict, reference):
        ref_variance = np.var(reference.values, axis=(0, 1))
        mse = np.var(predict - reference, axis=0)
        return (ref_variance[None, :] - mse) / ref_variance[None, :]

    def get_std(predict, reference):
        return np.std(predict - reference, axis=0)

    def plot_2d(ax, data_func, label: str, vmin=None, vmax=None, cmap="viridis"):
        data = {}
        for name in ("specific_humidity", "air_temperature"):
            data[name] = data_func(
                state_out[name], reference=state_out[f"{name}_reference"]
            )
        data["air_temperature_tendency"] = data_func(
            state_out["dQ1"], state_out["air_temperature_tendency_due_to_nudging"]
        )
        data["specific_humidity_tendency"] = data_func(
            state_out["dQ2"], state_out["specific_humidity_tendency_due_to_nudging"]
        )
        for i, name in enumerate(sorted(list(data.keys()))):
            if isinstance(vmin, tuple):
                vmin_i = vmin[i]
            else:
                vmin_i = vmin
            if isinstance(vmax, tuple):
                vmax_i = vmax[i]
            else:
                vmax_i = vmax
            im = ax[i].pcolormesh(data[name].T, vmin=vmin_i, vmax=vmax_i, cmap=cmap)
            plt.colorbar(im, ax=ax[i])
            for a in ax.flatten():
                a.set_ylim((79, 0))
            ax[i].set_title(f"{name} {label}")

    fig, ax = plt.subplots(4, 1, figsize=(8, 8))
    plot_2d(ax, get_r2, "R2", vmin=-1, vmax=1, cmap="RdBu")
    plt.tight_layout()
    fig, ax = plt.subplots(4, 1, figsize=(8, 8))
    plot_2d(ax, get_std, "std err")
    plt.tight_layout()

    # plt.show()

    def get_r2(predict, reference):
        ref_variance = np.var(reference, axis=0)
        mse = np.var(predict - reference, axis=0)
        return np.mean((ref_variance - mse) / ref_variance, axis=-1)

    r2 = {}
    for name in ("specific_humidity", "air_temperature"):
        r2[name] = get_r2(state_out[name], reference=state_out[f"{name}_reference"])
    r2["air_temperature_tendency"] = get_r2(
        state_out["dQ1"], state_out["air_temperature_tendency_due_to_nudging"]
    )
    r2["specific_humidity_tendency"] = get_r2(
        state_out["dQ2"], state_out["specific_humidity_tendency_due_to_nudging"]
    )

    fig, ax = plt.subplots(4, 1, figsize=(8, 8))
    for i, name in enumerate(sorted(list(r2.keys()))):
        ax[i].plot(r2[name])
        ax[i].set_ylim(-1, 1)
        ax[i].set_xlim(0, None)
        ax[i].set_title(f"{name} R2")
    plt.tight_layout()
    # plt.show()

    for i in range(0):
        fig, ax = plt.subplots(4, 2, figsize=(12, 8))
        plot_single(
            state_out["dQ1"][i, :, :].values * seconds_in_day,
            state_out["air_temperature_tendency_due_to_nudging"][i, :, :].values
            * seconds_in_day,
            "tendency of air_temperature (K/day)",
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
            "tendency of specific_humidity (kg/kg/day)",
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
