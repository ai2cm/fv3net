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
from fv3fit.reservoir.one_dim import ReservoirTrainingConfig
from ks import KuramotoSivashinskyConfig, get_time_downsampling_factor


logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "ks_config", type=str, help=("Config file for KS equation parameters")
    )
    parser.add_argument("train_config", type=str, help=("Config file for training"))
    parser.add_argument(
        "output_path", type=str, help="Output location for saving model."
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


def generate_training_time_series(
    ks_config, reservoir_timestep, n_samples, seed, input_noise
):
    # downsample in time to the reservoir timestep
    time_downsampling_factor = get_time_downsampling_factor(
        reservoir_timestep, ks_config.timestep
    )
    training_ts = ks_config.generate_from_seed(
        n_steps=time_downsampling_factor * n_samples, seed=seed,
    )

    training_ts = training_ts[:: int(time_downsampling_factor), :]
    training_ts = add_input_noise(training_ts, stddev=input_noise)
    return training_ts


def train(ks_config, train_config):
    training_ts = generate_training_time_series(
        ks_config=ks_config,
        reservoir_timestep=train_config.timestep,
        n_samples=train_config.n_burn + train_config.n_samples,
        seed=train_config.seed,
        input_noise=train_config.input_noise,
    )
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

    return ReservoirComputingModel(reservoir=reservoir, readout=readout,)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    with open(args.ks_config, "r") as f:
        ks_config = dacite.from_dict(KuramotoSivashinskyConfig, yaml.safe_load(f))
    with open(args.train_config, "r") as f:
        train_config_dict = yaml.safe_load(f)
        train_config = ReservoirTrainingConfig.from_dict(train_config_dict)

    predictor = train(ks_config, train_config)
    predictor.dump(args.output_path)
