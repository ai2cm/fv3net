import argparse
import fsspec
import numpy as np
from typing import Union, Optional, Sequence
import yaml

import fv3fit
from fv3fit.reservoir.utils import get_ordered_X
from fv3fit.reservoir import ReservoirComputingModel, HybridReservoirComputingModel


ReservoirModel = Union[ReservoirComputingModel, HybridReservoirComputingModel]


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("reservoir_model_path", type=str, help="Input Zarr path")
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


def get_predictions_over_batch(
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
                variable_time_series[t]
                for variable_time_series in hybrid_inputs_time_series
            ]
        prediction = model.predict(**predict_kwargs)
        prediction_time_series.append(prediction)
    return prediction_time_series


def _get_states_without_overlap(
    states_with_overlap_time_series: Sequence[np.ndarray], overlap: int
):
    states_without_overlap_time_series = []
    for var_time_series in states_with_overlap_time_series:
        # dims in array var_time_series are (t, x, y, z)
        states_without_overlap_time_series.append(
            var_time_series[:, overlap:-overlap, overlap:-overlap, :]
        )
    return np.stack(states_without_overlap_time_series, axis=1)


def main(args):
    model: ReservoirModel = fv3fit.load(args.reservoir_model_path)
    with fsspec.open(args.validation_config_path, "r") as f:
        val_data_config = yaml.safe_load(f)
    val_batches = _load_batches(
        path=val_data_config["url"],
        variables=_get_variables_to_load(model),
        nfiles=val_data_config["nfiles"],
    )
    # Initialize hidden state
    model.reset_state()

    one_step_prediction_time_series = []
    target_time_series = []
    for batch_data in val_batches:
        states_with_overlap_time_series = get_ordered_X(
            batch_data, model.input_variables
        )

        if isinstance(model, HybridReservoirComputingModel):
            hybrid_inputs_time_series = get_ordered_X(
                batch_data, model.hybrid_variables
            )
        else:
            hybrid_inputs_time_series = None
        batch_predictions = get_predictions_over_batch(
            model, states_with_overlap_time_series, hybrid_inputs_time_series
        )

        one_step_prediction_time_series += batch_predictions
        target_time_series.append(
            _get_states_without_overlap(
                states_with_overlap_time_series, overlap=model.rank_divider.overlap
            )
        )

    target_time_series = np.concatenate(target_time_series, axis=0)[
        args.n_synchronize :
    ]
    one_step_prediction_time_series = np.array(one_step_prediction_time_series)[
        args.n_synchronize :
    ]
    error = one_step_prediction_time_series - target_time_series
    del error
    # TODO: a following PR will calculate condensed metrics and time averages of the
    # predicted errors


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    main(args)
