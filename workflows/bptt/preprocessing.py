from typing import Iterable, Hashable, Tuple
import numpy as np
import xarray as xr
import tensorflow as tf
import numpy as np
import os
import vcm
import dataclasses
import argparse

COARSE_OUTPUT_URL = "/Volumes/OWC Envoy Pro EX/gs/vcm-ml-experiments/2021-01-22-nudge-to-fine-3hr-averages"
# COARSE_OUTPUT_URL = "/Volumes/OWC Envoy Pro EX/gs/vcm-ml-experiments/2021-01-22-nudge-to-fine-3hr/merged.zarr"

INPUT_NAMES = ["surface_geopotential", "cos_zenith_angle", "land_sea_mask"]

# notebook using the output data:
# https://github.com/VulcanClimateModeling/explore/blob/master/noahb/2021-01-26-average-nudging-ml.ipynb
# url = "gs://vcm-ml-experiments/2021-01-22-nudge-to-fine-3hr-averages"
# restarts_path: gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts
# mapper = open_nudge(url)
# diagnostics configuration:
# https://github.com/VulcanClimateModeling/vcm-workflow-control/blob/experiments/2021-01-21-nudge-to-fine-3hr-averages-working/examples/nudge-to-fine-run/baseline-config.yaml
# remember tendencies are stored in units per second, so multiply by timestep
# maybe have timestep as input for the ML model
# total_sky_downward_longwave_flux_at_surface, total_sky_downward_shortwave_flux_at_surface, total_sky_upward_longwave_flux_at_surface, total_sky_upward_shortwave_flux_at_surface


def open_zarr(url: str) -> xr.Dataset:
    fs = vcm.get_fs(url)
    map_ = fs.get_mapper(url)
    return xr.open_zarr(map_, consolidated=True)


def get_random_indices(n: int, i_max: int):
    batch_indices = np.arange(i_max)
    np.random.shuffle(batch_indices)
    return batch_indices[:n]


def get_dQ_arrays(base_value: xr.DataArray, nudging_tendency: xr.DataArray, timestep_seconds: float, n_total_samples: int, nt: int, nz: int, sample_idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # T and Q1 used here only as example, this is used equivalently for e.g. q, dQ2
    if base_value.shape != nudging_tendency.shape:
        raise ValueError(
            "received different shapes for base_value and nudging_tendency: "
            f"{base_value.shape} and {nudging_tendency.shape}"
        )
    T = base_value.values.reshape([n_total_samples, nt, nz])[sample_idx, :, :]
    dQ1_total = np.diff(T, axis=1) / timestep_seconds
    dQ1_nudging = nudging_tendency.values.reshape([n_total_samples, nt, nz])[sample_idx, :, :]
    dQ1_base = np.zeros_like(T)
    # before operating ML at each timestep, add the physics tendency from the last timestep
    # this is "parallel" in that the ML and the physics operate on the same state
    dQ1_base[:, 1:, :] = dQ1_total - dQ1_nudging[:, 1:, :]
    return T[:, :, :], dQ1_base, dQ1_nudging


def get_packed_array(
    dataset: xr.Dataset,
    names: Iterable[Hashable],
    n_total_samples=None,
    nt=None,
    nz=None,
    sample_idx=slice(None, None),
) -> Tuple[np.ndarray, Tuple[int]]:
    """
    Transform dataset into packed numpy array.

    Requires the dataset either be a "sample" dataset with dimensions [sample]
    and [sample, feature] for each variable, or a "timeseries" dataset with
    dimensions [tile, x, y, time, z] and [tile, x, y, time].

    Args:
        dataset: data to transform
        names: variable names to put in packed array, in order
        n_total_samples: n_tile * nx * ny, required if dataset is a timeseries dataset
        nt: length of time dimension, required if dataset is a timeseries dataset
        nz: length of vertical dimension, required if dataset is a timeseries dataset
    
    Returns:
        packed: packed numpy array
        features: number of features corresponding to each of the input names
    """
    array_list = []
    features_list = []
    for name in names:
        if len(dataset[name].dims) == 1:  # scalar sample array
            array_list.append(dataset[name][sample_idx, None])
        elif len(dataset[name].dims) == 2:  # vertically-resolved sample array
            array_list.append(dataset[name][sample_idx, :])
        if len(dataset[name].dims) == 4:  # scalar:
            _ensure_no_nones(n_total_samples, nt, nz)
            array_list.append(
                dataset[name].values.reshape([n_total_samples, nt])[sample_idx, :, None]
            )
            features_list.append(1)
        elif len(dataset[name].dims) == 5:  # assume z dimension is atmospheric vertical
            _ensure_no_nones(n_total_samples, nt, nz)
            array_list.append(
                dataset[name].values.reshape([n_total_samples, nt, nz])[sample_idx, :, :]
            )
            features_list.append(nz)
        else:
            raise NotImplementedError(
                "received array with unexpected number of dimensions: "
                f"{dataset[name].dims}"
            )
    return np.concatenate(array_list, axis=-1), tuple(features_list)


def _ensure_no_nones(n_total_samples, nt, nz):
    if n_total_samples is None:
        raise ValueError(
            "n_total_samples is required for a timeseries dataset, got None"
        )
    elif nt is None:
        raise ValueError("nt is required for a timeseries dataset, got None")
    elif nz is None:
        raise ValueError("nz is required for a timeseries dataset, got None")


def open_nudge(input_names) -> Iterable["TrainingArrays"]:
    # hard-coded prognostic names are air_temperature and specific_humidity
    # this function is hard-coded for this dataset source
    # assumptions like the nudging tendencies being available
    # every 3h while state is every 1.5h are hard-coded
    url = COARSE_OUTPUT_URL
    assert not url[-1] == "/"
    timestep_seconds = 3 * 60 * 60
    window_seconds = 7 * 24 * 60 * 60
    between_window_seconds = (2 * 24 + 9) * 60 * 60
    nt_window = int(window_seconds / timestep_seconds)
    nt_between_window = int(between_window_seconds / timestep_seconds)
    subsample_fraction = 0.125  # fraction of data to keep, randomly selected

    nudge_url = f"{url}/nudging_tendencies.zarr"
    state_url = f"{url}/state_after_timestep.zarr"

    # put time and z-dimensions last for sampling
    dim_order = ["tile", "x", "x_interface", "y", "y_interface", "time", "z", "z_soil"]
    nudge = open_zarr(nudge_url).transpose(*dim_order[:-1])  # no soil dimension
    # state is available on twice the timestep as nudging. On the second timestep,
    # it corresponds to the state after the first data point of nudging has been applied
    state = open_zarr(state_url)
    state = state.isel(
        time=range(1, len(state["time"]), 2)
    ).transpose(
        *dim_order
    ).rename_vars(
        {"longitude": "lon", "latitude": "lat"}
    )
    state = vcm.DerivedMapping(state).dataset(
        list(state.data_vars.keys()) + ["cos_zenith_angle"]
    )
    n_total_samples = np.product(state["air_temperature"].shape[:3])
    n_samples = int(subsample_fraction * n_total_samples)
    nt = state["air_temperature"].shape[3]
    nz = state["air_temperature"].shape[4]

    for i_start in range(0, nt - nt_window, nt_between_window):
        state_window = state.isel(time=slice(i_start, i_start+nt_window))
        nudge_window = nudge.isel(time=slice(i_start, i_start+nt_window))

        sample_idx = get_random_indices(n_samples, n_total_samples)

        Q1_base, dQ1_given, dQ1_nudge = get_dQ_arrays(
            state_window["air_temperature"],
            nudge_window["air_temperature_tendency_due_to_nudging"],
            timestep_seconds,
            n_total_samples,
            nt_window,
            nz,
            sample_idx,
        )

        Q2_base, dQ2_given, dQ2_nudge = get_dQ_arrays(
            state_window["specific_humidity"],
            nudge_window["specific_humidity_tendency_due_to_nudging"],
            timestep_seconds,
            n_total_samples,
            nt_window,
            nz,
            sample_idx,
        )

        X_in, input_features = get_packed_array(
            state_window, input_names, n_total_samples, nt_window, nz, sample_idx
        )
        lat = state_window["lat"].values.reshape([n_total_samples, nt_window])[sample_idx, 0]
        lon = state_window["lat"].values.reshape([n_total_samples, nt_window])[sample_idx, 0]

        yield TrainingArrays(
            inputs_baseline=X_in,
            given_tendency=np.concatenate([dQ1_given, dQ2_given], axis=2),
            nudging_tendency=np.concatenate([dQ1_nudge, dQ2_nudge], axis=2),
            prognostic_baseline=np.concatenate([Q1_base, Q2_base], axis=2),
            # temporary work-around for not having reference data in run, use baseline as reference data
            prognostic_reference=np.concatenate([Q1_base, Q2_base], axis=2),
            latitude=lat,
            longitude=lon,
            input_names=input_names,
            input_features=input_features,
        )


class TrainingArrays:
    """Normalized numpy arrays to provide directly to Keras model routines.
    Dimensionality of each array is [sample, time, feature/height]."""

    inputs_baseline: np.ndarray
    """Non-prognostic inputs for the machine learning model."""

    given_tendency: np.ndarray
    """The base model tendency added on each time step, in normalized units.

    The tendency at the first time step acts as an initial condition, since
    the machine learning model prognostic state is initialized at zero.

    Note the base tendency differs from the time difference of the prognostic_baseline
    state, in that it does not include the part of the tendency we'd like to
    predict with ML (e.g. nudging or disabled physics).
    
    Given as an input to the machine learning model.
    """
    nudging_tendency: np.ndarray
    """The nudging tendency of the coarse model.
    
    Used mainly as a diagnostic tool for visualizing tendencies,
    can be derived from given_tendency and prognostic_baseline.
    """
    prognostic_baseline: np.ndarray
    """The prognostic state of the base model.
    
    Used mainly as a diagnostic tool for comparing model states,
    can be derived from y_base_tendency."""
    prognostic_reference: np.ndarray
    """the prognostic state of the reference model used as a target"""


    def __init__(
        self,
        inputs_baseline: np.ndarray,
        given_tendency: np.ndarray,
        nudging_tendency: np.ndarray,
        prognostic_baseline: np.ndarray,
        prognostic_reference: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        input_names: Tuple[str],
        input_features: Tuple[str],
    ):
        """
        Args:
        """
        if prognostic_baseline.shape != prognostic_reference.shape:
            raise ValueError(
                "expected y_base and y_reference to have the same shape, "
                f"but received shapes {prognostic_baseline.shape} and {prognostic_reference.shape}"
            )
        self.inputs_baseline = inputs_baseline
        self.given_tendency = given_tendency
        self.nudging_tendency = nudging_tendency
        self.prognostic_baseline = prognostic_baseline
        self.prognostic_reference = prognostic_reference
        self.latitude = latitude
        self.longitude = longitude
        self.input_names = input_names
        self.input_features = input_features

    def dump(self, f):
        np.savez(
            f,
            inputs_baseline = self.inputs_baseline,
            given_tendency = self.given_tendency,
            nudging_tendency = self.nudging_tendency,
            prognostic_baseline = self.prognostic_baseline,
            prognostic_reference = self.prognostic_reference,
            input_names = self.input_names,
            input_features = self.input_features,
            latitude=self.latitude,
            longitude=self.longitude
        )

    @classmethod
    def load(cls, f):
        data = np.load(f, allow_pickle=False)
        return cls(
            data["inputs_baseline"],
            data["given_tendency"],
            data["nudging_tendency"],
            data["prognostic_baseline"],
            data["prognostic_reference"],
            data["latitude"],
            data["longitude"],
            tuple(str(name) for name in data["input_names"]),
            tuple(int(n) for n in data["input_features"])
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_data_path", type=str, help="directory to output array data"
    )
    # parser.add_argument(
    #     "config_file", type=str, help="Local path for training configuration yaml file",
    # )
    # parser.add_argument(
    #     "output_data_path", type=str, help="Location to save config and trained model."
    # )
    # parser.add_argument(
    #     "--timesteps-file",
    #     type=str,
    #     default=None,
    #     help="json file containing a list of timesteps in YYYYMMDD.HHMMSS format",
    # )
    # parser.add_argument(
    #     "--validation-timesteps-file",
    #     type=str,
    #     default=None,
    #     help="Json file containing a list of validation timesteps in "
    #     "YYYYMMDD.HHMMSS format. Only relevant for keras training",
    # )
    # parser.add_argument(
    #     "--local-download-path",
    #     type=str,
    #     help="Optional path for downloading data before training. If not provided, "
    #     "will read from remote every epoch. Local download greatly speeds training.",
    # )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    fs = vcm.get_fs(args.output_data_path)
    fs.makedirs(args.output_data_path, exist_ok=True)
    for i, arrays in enumerate(open_nudge(INPUT_NAMES)):
        arrays_filename = os.path.join(args.output_data_path, f"arrays_{i:05d}")
        with fs.open(arrays_filename, "wb") as f:
            arrays.dump(f)
