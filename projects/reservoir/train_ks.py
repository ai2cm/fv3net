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
        reservoir_states.append(reservoir.state)
        reservoir.increment_state(x)
    return np.array(reservoir_states)


def _initialize_reservoir(reservoir_config):
    return Reservoir(reservoir_config)


def _train_predictor(subdomain, reservoir, train_config):
    n_select = (
        None
        if not train_config.n_samples
        else train_config.n_burn + train_config.n_samples
    )
    noise = InputNoise(reservoir.hyperparameters.input_dim, train_config.noise)
    X = subdomain.overlapping[:-1]
    y = subdomain.nonoverlapping[1:]

    X_res = transform_inputs_to_reservoir_states(
        X=X[:n_select], reservoir=reservoir, input_noise=noise
    )

    reg = Ridge(alpha=train_config.l2)
    X_, y_ = shuffle(
        X_res[train_config.n_burn : n_select], y[train_config.n_burn : n_select]
    )
    reg.fit(X_, y_)

    return ReservoirPredictor(reservoir, reg)


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
