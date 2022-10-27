import argparse
import copy
import dacite
import dataclasses
import fsspec
import logging
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle

import yaml
from fv3fit.reservoir import (
    ReservoirComputingReadout,
    Reservoir,
    HybridReservoirComputingModel,
)
from fv3fit.reservoir.config import ReservoirTrainingConfig
from ks import KuramotoSivashinskyConfig, ImperfectKSModel


logger = logging.getLogger(__name__)


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "ks_config", type=str, help=("Config file for KS equation parameters")
    )
    parser.add_argument("train_config", type=str, help=("Config file for training"))
    parser.add_argument(
        "output_path", type=str, help="Output location for saving model."
    )
    parser.add_argument(
        "trial_config",
        type=str,
        help="config file of field name and values to trial over",
    )
    parser.add_argument(
        "epsilon",
        type=float,
        help="epsilon error for KS integration to generate 'imperfect' KS model",
    )
    return parser


def add_input_noise(arr, stddev):
    return arr + np.random.normal(loc=0, scale=stddev, size=arr.shape)


def transform_inputs_to_reservoir_states(X, reservoir):
    reservoir_states = [
        reservoir.state,
    ]
    for x in X:
        reservoir.increment_state(x)
        reservoir_states.append(reservoir.state)
    # last hidden state has no corresponding target output state,
    # so it is not used in training
    return np.array(reservoir_states[:-1])


def add_imperfect_prediction_to_train_data(X, imperfect_prediction_states):
    return np.hstack(X, imperfect_prediction_states)


def get_imperfect_ks_model(ks_config, epsilon, steps_per_rc_step):
    imperfect_ks_config = copy.copy(ks_config)
    imperfect_ks_config.error_eps = epsilon
    imperfect_ks_config.time_downsampling_factor = (
        steps_per_rc_step * ks_config.time_downsampling_factor
    )
    return ImperfectKSModel(imperfect_ks_config)


def create_imperfect_prediction_train_data(
    imperfect_model, ts_truth,
):
    imperfect_predictions = [
        ts_truth[0],
    ]
    for i in range(len(ts_truth) - 1):
        input = ts_truth[i]
        imperfect_predictions.append(imperfect_model.predict(input))
    return imperfect_predictions


def train_hybrid(ks_config, train_config, epsilon, imperfect_model_steps_per_rc_step):
    training_ts = ks_config.generate(
        n_steps=train_config.n_samples + train_config.n_burn, seed=train_config.seed
    )
    imperfect_model = get_imperfect_ks_model(
        ks_config, epsilon, imperfect_model_steps_per_rc_step
    )
    imperfect_training_ts = create_imperfect_prediction_train_data(
        imperfect_model, training_ts,
    )

    training_ts = add_input_noise(training_ts, stddev=train_config.input_noise)
    training_ts_burnin, training_ts_keep = (
        training_ts[: train_config.n_burn],
        training_ts[train_config.n_burn :],
    )
    reservoir = Reservoir(train_config.reservoir_hyperparameters)

    reservoir.synchronize(training_ts_burnin)
    training_reservoir_states = transform_inputs_to_reservoir_states(
        X=training_ts_keep, reservoir=reservoir
    )

    X_train = np.hstack(
        [training_reservoir_states, imperfect_training_ts[train_config.n_burn :]]
    )
    y_train = training_ts_keep
    X_train, y_train = shuffle(X_train, y_train, random_state=train_config.seed)

    linear_regressor = Ridge(
        **train_config.readout_hyperparameters.linear_regressor_kwargs
    )
    readout = ReservoirComputingReadout(
        linear_regressor, train_config.readout_hyperparameters.square_half_hidden_state,
    )
    readout.fit(X_train, y_train)

    predictor = HybridReservoirComputingModel(
        reservoir=reservoir, readout=readout, imperfect_model=imperfect_model
    )
    return predictor


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    with open(args.ks_config, "r") as f:
        ks_config = dacite.from_dict(KuramotoSivashinskyConfig, yaml.safe_load(f))
    with open(args.train_config, "r") as f:
        train_config_dict = yaml.safe_load(f)
        train_config = ReservoirTrainingConfig.from_dict(train_config_dict)

    predictor = train_hybrid(ks_config, train_config, args.epsilon)

    predictor.dump(args.output_path)
    train_config.dump(args.output_path)
    with fsspec.open(os.path.join(args.output_path, "ks_config.yaml"), "w") as f:
        yaml.dump(dataclasses.asdict(ks_config), f, indent=4)
