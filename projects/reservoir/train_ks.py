import argparse
import dacite
import logging
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle

import yaml
from fv3fit.reservoir.reservoir import Reservoir
from fv3fit.reservoir.predictor import ReservoirPredictor
from fv3fit.reservoir.transform import InputNoise
from fv3fit.reservoir.config import ReservoirTrainingConfig
from ks import KSConfig


logger = logging.getLogger(__name__)


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "ks_config", type=str, help=("Config file for KS equation parameters")
    )
    parser.add_argument("train_config", type=str, help=("Config file for training"))

    parser.add_argument(
        "--ks-seed",
        type=int,
        default=0,
        help=("Optional random seed for generating KS initial condition"),
    )
    return parser


def transform_inputs_to_reservoir_states(X, reservoir, input_noise: InputNoise):
    reservoir_states = []
    X_noised = X + input_noise.generate()
    for x in X_noised:
        reservoir.increment_state(x)
        reservoir_states.append(reservoir.state)
    return np.array(reservoir_states)


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    with open(args.ks_config, "r") as f:
        ks_config = dacite.from_dict(KSConfig, yaml.safe_load(f))
    with open(args.train_config, "r") as f:
        train_config_dict = yaml.safe_load(f)
        train_config = ReservoirTrainingConfig.from_dict(train_config_dict)

    training_ts = ks_config.generate(
        n_steps=train_config.n_samples, seed=train_config.seed
    )

    reservoir = Reservoir(train_config.reservoir_hyperparameters)
    input_noise = InputNoise(
        size=ks_config.N, stddev=train_config.input_noise, seed=train_config.seed
    )

    training_ts_burnin, training_ts_keep = (
        training_ts[: train_config.n_burn],
        training_ts[train_config.n_burn :],
    )
    reservoir.synchronize(training_ts_burnin, input_noise=input_noise)
    training_reservoir_states = transform_inputs_to_reservoir_states(
        X=training_ts_keep, reservoir=reservoir, input_noise=input_noise
    )

    X_train = training_reservoir_states[:1]
    y_train = training_ts_keep[1:]
    X_train, y_train = shuffle(X_train, y_train, random_state=train_config.seed)

    linear_regressor = Ridge(
        alpha=train_config.readout_config.l2, solver=train_config.readout_config.solver
    )
    linear_regressor.fit(X_train, y_train)

    predictor = ReservoirPredictor(
        reservoir=reservoir,
        linreg=linear_regressor,
        square_half_hidden_state=train_config.square_half_hidden_state,
    )
