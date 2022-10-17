import argparse
import dacite
import logging
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle

import yaml
from fv3fit.reservoir import (
    ReservoirComputingReadout,
    Reservoir,
    ReservoirComputingModel,
)
from fv3fit.reservoir.config import ReservoirTrainingConfig
from ks import KSConfig


logger = logging.getLogger(__name__)


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "ks_config", type=str, help=("Config file for KS equation parameters")
    )
    parser.add_argument("train_config", type=str, help=("Config file for training"))
    return parser


def add_input_noise(arr, stddev):
    return np.random.normal(loc=0, scale=stddev, size=arr.shape)


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


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    with open(args.ks_config, "r") as f:
        ks_config = dacite.from_dict(KSConfig, yaml.safe_load(f))
    with open(args.train_config, "r") as f:
        train_config_dict = yaml.safe_load(f)
        train_config = ReservoirTrainingConfig.from_dict(train_config_dict)

    training_ts = ks_config.generate(
        n_steps=train_config.n_samples + train_config.n_burn, seed=train_config.seed
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

    X_train = training_reservoir_states
    y_train = training_ts_keep
    X_train, y_train = shuffle(X_train, y_train, random_state=train_config.seed)

    linear_regressor = Ridge(
        **train_config.readout_hyperparameters.linear_regressor_kwargs
    )
    readout = ReservoirComputingReadout(
        linear_regressor, train_config.readout_hyperparameters.square_half_hidden_state,
    )
    readout.fit(X_train, y_train)

    predictor = ReservoirComputingModel(reservoir=reservoir, readout=readout,)

    predictor.dump("gs://vcm-ml-scratch/annak/2022-10-13/rc_predictor")
