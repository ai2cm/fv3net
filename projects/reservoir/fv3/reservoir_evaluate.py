# takes prediction and calculates evaluation scores
# todo apply land sea mask
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score
import argparse
import glob
import os
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
        "swap",
    )
    # full prediction - rollout
    rollout_scores_full = calculate_scores(
        dataset.sst.data[:, args.n_synchronize + 1 :, ...],
        rollout_prediction.sst.data[:-1, ...],
        dataset.mask_field[:, args.n_synchronize + 1 :, ...],
        "rollout",
        "swap",
    )
    # tendency of prediction
    single_step_scores_tend = calculate_scores(
        dataset.sst_tendency[:, :-1, ...].data * DELTA_T,
        single_step_prediction_tendency.data[:, :-1, ...] * DELTA_T,
        dataset.mask_field[:, :-1, ...],
        "single_step_tend",
    )
    # tendency of rollout
    rollout_scores_tend = calculate_scores(
        dataset.sst_tendency.data[:, args.n_synchronize : -1, ...] * DELTA_T,
        rollout_prediction_tendency.data[:-1, ...] * DELTA_T,
        dataset.mask_field[:, args.n_synchronize : -1, ...],
        "rollout_tend",
        "swap",
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
    # make sure to mask out land
    # todo: apply land mask
    if swap == "swap":
        target = np.swapaxes(target, 0, 1)
    scores = {}
    squared_difference_masked = np.ma.array((target - pred) ** 2, mask=mask)
    difference_masked = np.ma.array(target - pred, mask=mask)
    absolute_difference_masked = np.ma.array(np.abs(target - pred), mask=mask)
    scores[case + "_mse"] = np.mean(squared_difference_masked)
    scores[case + "_rmse"] = np.sqrt(scores[case + "_mse"])
    scores[case + "_mae"] = np.mean(absolute_difference_masked)
    scores[case + "_mean_bias"] = np.mean(difference_masked)
    scores[case + "_r2"] = r2_score(
        target.flatten()[~np.isnan(target.flatten())],
        pred.flatten()[~np.isnan(target.flatten())],
    )
    return scores


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
