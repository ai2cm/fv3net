# takes a trained model and performs single step and rollout prediction
# todo: handle hybrid case
# ie child stepper class
# todo: handle tendency case
# ie adapt rollout
# calculate full prediction
import fv3fit
from fv3fit.reservoir import ReservoirDatasetAdapter, HybridReservoirDatasetAdapter
from tqdm import tqdm
from fv3fit._shared.halos import append_halos
import numpy as np
from typing import Mapping
import xarray as xr
from argparse import ArgumentParser
import os
import glob

HYBRID_VARIABLES = [
    "t2m_at_next_timestep",
    "u10_at_next_timestep",
    "v10_at_next_timestep",
]

DELTA_T = 604800  # 7 days in seconds


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model_path", help="path trained model directory")
    parser.add_argument(
        "--data_path",
        help="path to input data directory, should contain subfolders /tile-x",
    )
    parser.add_argument("--output_path", help="path to save predictions")
    parser.add_argument(
        "--tendency_or_full",
        default="full",
        help="str whether we trained on the tendency or the full sst",
    )
    parser.add_argument(
        "--hybrid_or_sst_only",
        default="sst_only",
        help="str whether we are in the hybrid or sst-only case",
    )
    parser.add_argument("--n_synchronize", type=int, default=100)
    return parser.parse_args()


"""example call: python reservoir_inference.py
--model_path /home/paulah/fv3net-offline-reservoirs/sst-only-full-sub-24-halo-4-masked
--data_path /home/paulah/data/era5/fv3-halo-0-masked/val/
--output_path
/home/paulah/fv3net-offline-reservoirs/sst-only-full-sub-24-halo-4-masked
--n_synchronize 200"""


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

    n_synchronize = args.n_synchronize
    # load models
    rank_models = {r: fv3fit.load(args.model_path + f"-tile-{r}") for r in range(6)}

    # initialize stepper
    if args.hybrid_or_sst_only == "hybrid":
        stepper = GlobalHybridReservoirStepper(rank_models)
    else:
        stepper = GlobalReservoirStepper(rank_models)

    # single step prediction
    stepper.reset_states()
    single_steps = []

    for t in tqdm(range(len(dataset.time))):
        stepper.increment_reservoir_states(dataset.isel(time=t))
        if args.tendency_or_full == "tendency":
            prediction = stepper.predict_global_state_tendency(dataset.isel(time=t))
        else:
            prediction = stepper.predict_global_state(dataset.isel(time=t))
        single_steps.append(prediction)
    single_step_prediction = xr.concat(single_steps, dim="time")

    # rollout
    stepper.reset_states()
    for t in tqdm(range(n_synchronize)):
        stepper.increment_reservoir_states(dataset.isel(time=t))
    rollout_steps = []

    for t in tqdm(range(n_synchronize, len(dataset.time))):
        if args.tendency_or_full == "tendency":
            prediction = stepper.predict_global_state_tendency(dataset.isel(time=t))
        else:
            prediction = stepper.predict_global_state(dataset.isel(time=t))
        rollout_steps.append(prediction)
        stepper.increment_reservoir_states(prediction)
    rollout = xr.concat(rollout_steps, dim="time")

    # save predictions
    os.system("mkdir -p " + args.output_path)
    single_step_prediction.to_netcdf(args.output_path + "/single_step_prediction.nc")
    rollout.to_netcdf(args.output_path + "/rollout.nc")


class GlobalReservoirStepper:
    _TOTAL_RANKS = 6

    def __init__(self, model_adapters: Mapping[int, ReservoirDatasetAdapter]):
        self._validate_models(model_adapters)
        self.model_adapters = model_adapters
        self.global_state = None
        self.n_halo = self.model_adapters[0].model.rank_divider.overlap

    def _validate_models(self, model_adapters):
        if len(model_adapters) != 6 and model_adapters.keys() != set(range(6)):
            raise ValueError("GlobalReservoir can only be used for the 6 rank case.")
        overlaps = [
            model_adapter.model.rank_divider.overlap
            for model_adapter in model_adapters.values()
        ]
        if len(np.unique(overlaps)) != 1:
            raise ValueError("All models must have the same number of overlap cells.")

    def increment_reservoir_states(self, input_global_state: xr.Dataset):
        state_append_halos = append_halos(input_global_state, n_halo=self.n_halo)
        for rank in range(self._TOTAL_RANKS):
            model_adapter = self.model_adapters[rank]
            rank_input = state_append_halos.sel(tile=rank)[
                model_adapter.model.input_variables
            ]
            model_adapter.increment_state(rank_input)
        return state_append_halos

    def predict_global_state(self, hybrid_input: xr.Dataset):
        rank_predictions = []
        # dummy input xr.Dataset is a required arg for Predictor class but is not
        # used by the non-hybrid reservoir's predict method
        dummy_inputs = xr.Dataset({})
        for rank in range(self._TOTAL_RANKS):
            rank_predictions.append(self.model_adapters[rank].predict(dummy_inputs))
        return xr.concat(rank_predictions, dim="tile")

    def predict_global_state_tendency(self, hybrid_input: xr.Dataset):
        rank_predictions = []
        dummy_inputs = xr.Dataset({})

        for rank in range(self._TOTAL_RANKS):
            true_sst = hybrid_input.sst.sel(tile=rank)
            full_prediction = self.model_adapters[rank].predict(dummy_inputs)
            full_prediction.sst.data = true_sst + DELTA_T * full_prediction.sst.data
            rank_predictions.append(full_prediction)
        return xr.concat(rank_predictions, dim="tile")

    def reset_states(self):
        for rank in range(self._TOTAL_RANKS):
            self.model_adapters[rank].model.reset_state()


class GlobalHybridReservoirStepper(GlobalReservoirStepper):
    def __init__(self, model_adapters: Mapping[int, HybridReservoirDatasetAdapter]):
        super().__init__(model_adapters)

    def predict_global_state(self, hybrid_input: xr.Dataset):
        rank_predictions = []
        # hybrid_input gives the additional input variables for the reservoir
        for rank in range(self._TOTAL_RANKS):
            rank_predictions.append(
                self.model_adapters[rank].predict(hybrid_input.sel(tile=rank))
            )
        return xr.concat(rank_predictions, dim="tile")

    def predict_global_state_tendency(self, hybrid_input: xr.Dataset):
        rank_predictions = []
        # hybrid_input gives the additional input variables for the reservoir
        for rank in range(self._TOTAL_RANKS):
            true_sst = hybrid_input.sst.sel(tile=rank)
            full_prediction = self.model_adapters[rank].predict(
                hybrid_input.sel(tile=rank)
            )
            full_prediction.sst.data = true_sst + DELTA_T * full_prediction.sst.data
            rank_predictions.append(full_prediction)
        return xr.concat(rank_predictions, dim="tile")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
