import argparse
import fsspec
import numpy as np
import os
from tempfile import NamedTemporaryFile
from typing import Union, Optional, Sequence, cast
import xarray as xr
import vcm
import yaml

import fv3fit
from fv3fit.reservoir.validation import validation_prediction
from fv3fit.reservoir import (
    ReservoirComputingModel,
    HybridReservoirComputingModel,
    HybridReservoirDatasetAdapter,
    ReservoirDatasetAdapter,
)

import logging

logger = logging.getLogger(__name__)

ReservoirModel = Union[ReservoirComputingModel, HybridReservoirComputingModel]
ReservoirAdapter = Union[HybridReservoirDatasetAdapter, ReservoirDatasetAdapter]


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("reservoir_model_path", type=str, help="Input Zarr path")
    parser.add_argument("output_path", type=str, help="Directory to save outputs to")
    parser.add_argument(
        "validation_config_path", type=str, help="Path to validation data config"
    )
    parser.add_argument(
        "n_synchronize",
        type=int,
        help=(
            "Number of timesteps from start of validation to use in reservoir "
            "synchronization (not used in prediction)."
        ),
    )
    parser.add_argument(
        "--n-validation-batches",
        type=int,
        default=None,
        help="Number of batch data netcdfs to use for validation. Defaults to use all.",
    )
    return parser


def _load_batches(path, variables, nfiles):
    ts_loader = fv3fit.data.NCDirLoader(
        url=path,
        nfiles=nfiles,
        dim_order=["time", "x", "y", "z"],
        varying_first_dim=True,
        sort_files=True,
        shuffle=False,
    )
    tfdata = ts_loader.open_tfdataset(
        variable_names=variables, local_download_path=None
    )
    return tfdata


def _get_variables_to_load(model: ReservoirModel):
    variables = list(model.input_variables)
    if isinstance(model, HybridReservoirComputingModel):
        return variables + list(model.hybrid_variables)
    else:
        return variables


def _get_predictions_over_batch(
    model: ReservoirModel,
    states_with_overlap_time_series: Sequence[np.ndarray],
    hybrid_inputs_time_series: Optional[Sequence[np.ndarray]] = None,
):
    prediction_time_series = []
    n_timesteps = states_with_overlap_time_series[0].shape[0]
    for t in range(n_timesteps):
        state = [
            variable_time_series[t]
            for variable_time_series in states_with_overlap_time_series
        ]
        model.increment_state(state)
        predict_kwargs = {}
        if hybrid_inputs_time_series is not None:
            predict_kwargs["hybrid_input"] = [
                variable_time_series
                for variable_time_series in hybrid_inputs_time_series[t]
            ]
        prediction = model.predict(**predict_kwargs)
        prediction_time_series.append(prediction)
    return prediction_time_series


def _time_mean_dataset(variables, arr, label):
    ds = xr.Dataset()
    time_mean_error = np.mean(arr, axis=0)
    for v, var in enumerate(variables):
        ds[f"{label}_{var}"] = xr.DataArray(time_mean_error[v], dims=["x", "y", "z"])
    return ds


def _get_states_without_overlap(
    states_with_overlap_time_series: Sequence[np.ndarray], overlap: int
):
    states_without_overlap_time_series = []
    for var_time_series in states_with_overlap_time_series:
        # dims in array var_time_series are (t, x, y, z)
        states_without_overlap_time_series.append(
            var_time_series[:, overlap:-overlap, overlap:-overlap, :]
        )
    # dims (t, var, x, y, z)
    return np.stack(states_without_overlap_time_series, axis=1)


def main(args):
    adapter = cast(ReservoirAdapter, fv3fit.load(args.reservoir_model_path))
    model: ReservoirModel = adapter.model
    with fsspec.open(args.validation_config_path, "r") as f:
        val_data_config = yaml.safe_load(f)
    val_batches = _load_batches(
        path=val_data_config["url"],
        variables=_get_variables_to_load(model),
        nfiles=val_data_config.get("nfiles", None),
    )

    ds = validation_prediction(model, val_batches, args.n_synchronize,)

    output_file = os.path.join(args.output_path, "offline_diags.nc")

    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        vcm.get_fs(args.output_path).put(tmpfile.name, output_file)

    logger.info(f"Saved netcdf output to {output_file}")

    # TODO: a following PR will calculate condensed metrics and time averages of the
    # predicted errors


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    main(args)
