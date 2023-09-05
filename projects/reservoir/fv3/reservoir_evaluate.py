# takes prediction and calculates evaluation scores
# todo apply land sea mask
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score
import argparse
import glob
import os
import matplotlib.pyplot as plt
import csv

"""example call:
python reservoir_evaluate.py
--data_path /home/paulah/data/era5/fv3-halo-0-masked/val
--input_path /home/paulah/fv3net-offline-reservoirs/hybrid-full-sub-24-halo-4-masked
--n_synchronize 200"""

DELTA_T = 604800  # 7 days in seconds


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", help="path to data directory, should contain subfolders /tile-x"
    )
    parser.add_argument("--input_path", help="path to save predictions")
    parser.add_argument("--n_synchronize", type=int, default=100)
    return parser.parse_args()


def main(args):
    # load data
    dataset = xr.concat(
        [
            xr.open_dataset(
                glob.glob(os.path.join(args.data_path, f"tile-{r}/*.nc"))[0]
            )
            for r in range(6)
        ],
        dim="tile",
    )

    # load single step and rollout predictions
    single_step_prediction = xr.open_dataset(
        args.input_path + "/single_step_prediction.nc"
    )
    rollout_prediction = xr.open_dataset(args.input_path + "/rollout.nc")

    # calculate tendencies need exactly the length of target tendency
    single_step_prediction_tendency = (
        dataset.sst.shift(time=-1) - single_step_prediction.sst
    ) / DELTA_T
    # for rollout tendency use rollout prediction to calculate
    rollout_prediction_tendency = (
        rollout_prediction.sst.shift(time=-1) - rollout_prediction.sst
    ) / DELTA_T
    # calculate scores function on
    # full prediction - single step
    single_step_scores = calculate_scores(
        dataset.sst.data[:, 1:, ...],
        single_step_prediction.sst.data[:-1, ...],
        dataset.mask_field[:, 1:, ...],
        "single_step",
        "swap_target",
    )
    # full prediction - rollout
    rollout_scores_full = calculate_scores(
        dataset.sst.data[:, args.n_synchronize + 1 :, ...],
        rollout_prediction.sst.data[:-1, ...],
        dataset.mask_field[:, args.n_synchronize + 1 :, ...],
        "rollout",
        "swap_target",
    )
    # tendency of prediction
    single_step_scores_tend = calculate_scores(
        dataset.sst_tendency[:, :-1, ...].data * DELTA_T,
        single_step_prediction_tendency.data[:, :-1, ...] * DELTA_T,
        dataset.mask_field[:, :-1, ...],
        "single_step_tend",
        "swap_both",
    )
    # tendency of rollout
    rollout_scores_tend = calculate_scores(
        dataset.sst_tendency.data[:, args.n_synchronize : -1, ...] * DELTA_T,
        rollout_prediction_tendency.data[:-1, ...] * DELTA_T,
        dataset.mask_field[:, args.n_synchronize : -1, ...],
        "rollout_tend",
        "swap_target",
    )
    # combine single_step_scores, rollout_scores_full,
    # single_step_scores_tend, rollout_scores_tend into one dictionary
    combined_scores = {
        **single_step_scores,
        **rollout_scores_full,
        **single_step_scores_tend,
        **rollout_scores_tend,
    }
    # save scores as csv
    w = csv.writer(open(args.input_path + "/scores.csv", "w"))
    # loop over dictionary keys and values
    for key, val in combined_scores.items():
        # write every key and value to file
        w.writerow([key, val])


def calculate_scores(target, pred, mask, case, swap=None):
    # dimensions of target and pred are (time, tile, y, x)
    # make sure to mask out land
    # todo: apply land mask
    print(case)
    print(swap)  #
    print(target.shape)
    print(pred.shape)
    if swap == "swap_target":
        target = np.swapaxes(target, 0, 1)
    elif swap == "swap_both":
        target = np.swapaxes(target, 0, 1)
        pred = np.swapaxes(pred, 0, 1)
    scores = {}
    squared_difference_masked = np.ma.array((target - pred) ** 2, mask=mask)
    difference_masked = np.ma.array(target - pred, mask=mask)
    absolute_difference_masked = np.ma.array(np.abs(target - pred), mask=mask)
    scores[case + "_mse"] = np.mean(squared_difference_masked)
    scores[case + "_rmse"] = np.sqrt(scores[case + "_mse"])
    scores[case + "_mae"] = np.mean(absolute_difference_masked)
    scores[case + "_mean_bias"] = np.mean(difference_masked)
    # calculate r2 score over timeseries, ignoring nans
    r2_scores = []
    for tile in range(6):
        area_r2_scores = r2_score(
            target[:, tile, :, :].reshape(-1, target.shape[-1] ** 2)[
                ~np.isnan(target[:, tile, :, :].reshape(-1, target.shape[-1] ** 2))
            ],
            pred[:, tile, :, :].reshape(-1, target.shape[-1] ** 2)[
                ~np.isnan(target[:, tile, :, :].reshape(-1, target.shape[-1] ** 2))
            ],
            multioutput="uniform_average",
        )
        r2_scores.append(area_r2_scores)
    # mean over tiles
    scores[case + "_r2"] = np.mean(r2_scores)

    # add scores per tile
    scores[case + "_mse_tile"] = np.mean(squared_difference_masked, axis=(0, 2, 3))
    scores[case + "_rmse_tile"] = np.sqrt(scores[case + "_mse_tile"])
    scores[case + "_mae_tile"] = np.mean(absolute_difference_masked, axis=(0, 2, 3))
    scores[case + "_mean_bias_tile"] = np.mean(difference_masked, axis=(0, 2, 3))
    scores[case + "_r2_tile"] = np.array(r2_scores)
    return scores


def create_plots(true, pred, mask):
    pass


def create_timeseries_mean_plots(true, pred, mask, dir):
    # make plot with 7 subplots, one global plot and 6 tiles
    fig, axs = plt.subplots(7, 1, figsize=(10, 10))
    # plot global
    axs[0].plot(true.mean(axis=(1, 2, 3)), label="true")
    # add title
    axs[0].set_title("Global")
    # plot tiles
    for tile in range(6):
        axs[tile + 1].plot(true[:, tile, :, :].mean(axis=(1, 2)), label="true")
        axs[tile + 1].plot(pred[:, tile, :, :].mean(axis=(1, 2)), label="pred")
        # add title
        axs[tile + 1].set_title("Tile " + str(tile))
    # save plot
    plt.savefig(dir + "/mean_timeseries.png")


def create_error_timeseries_mean_plots(error, mask, dir):
    # smae plot as create_timeseries_mean_plots but with error
    # make plot with 7 subplots, one global plot and 6 tiles
    fig, axs = plt.subplots(7, 1, figsize=(10, 10))
    # plot global error
    axs[0].plot(error.mean(axis=(1, 2, 3)), label="error")
    # add title
    axs[0].set_title("Global")
    # plot tiles erros
    for tile in range(6):
        axs[tile + 1].plot(error[:, tile, :, :].mean(axis=(1, 2)), label="error")
        # add title
        axs[tile + 1].set_title("Tile " + str(tile))
    # save plot
    plt.savefig(dir + "/mean_error_timeseries.png")


def create_random_timeseries_plots(true, pred, mask, dir):
    # make plot with 2 x 6 sublplots, two for each tile
    fig, axs = plt.subplots(6, 2, figsize=(10, 10))
    # plot tiles
    for tile in range(6):
        # choose a random location
        y = np.random.randint(0, true.shape[-2])
        x = np.random.randint(0, true.shape[-1])
        # plot timeseires at random location
        axs[tile, 0].plot(true[:, tile, x, y], label="true")
        axs[tile, 0].plot(pred[:, tile, x, y], label="pred")

        # add title that shows tile number and location
        axs[tile, 0].set_title("Tile " + str(tile) + " at location " + str((x, y)))

        # plot random timeseries
        axs[tile, 1].plot(true[:, tile, x, y], label="true")
        axs[tile, 1].plot(pred[:, tile, x, y], label="pred")
        # add title
        axs[tile, 1].set_title("Tile " + str(tile) + " at location " + str((x, y)))

    # save plot
    plt.savefig(dir + "/random_timeseries.png")


def create_spatial_plots(true, pred, mask, dir):
    # make plot with 3 x 2 subplots, one column for true one column for pred
    pass


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
