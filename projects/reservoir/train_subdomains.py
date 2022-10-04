import argparse
import dacite
from joblib import Parallel, delayed

import logging
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
from typing import Sequence

import yaml
from fv3fit.reservoir.reservoir import (
    TrainConfig,
    ReservoirHyperparameters,
    Reservoir,
    InputNoise,
    ReservoirPredictor,
    transform_inputs_to_reservoir_states,
)
from fv3fit.reservoir.domain import PeriodicDomain
from ks import KSConfig


logger = logging.getLogger(__name__)


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "ks_config", type=str, help=("Config file for KS equation parameters")
    )
    parser.add_argument("train_config", type=str, help=("Config file for training"))
    parser.add_argument(
        "reservoir_config", type=str, help=("Config file for reservoir parameters")
    )
    return parser


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


def train_subdomain_predictors(
    domain: PeriodicDomain, reservoirs: Sequence[Reservoir], train_config: TrainConfig
):
    subdomains = [domain[i] for i in range(domain.n_subdomains)]

    subdomain_predictors = Parallel(n_jobs=train_config.n_jobs, verbose=True)(
        delayed(_train_predictor)(subdomain, res, train_config)
        for res, subdomain in zip(reservoirs, subdomains)
    )

    return subdomain_predictors


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    with open(args.ks_config, "r") as f:
        ks_config = dacite.from_dict(KSConfig, yaml.safe_load(f))
    with open(args.train_config, "r") as f:
        train_config = dacite.from_dict(TrainConfig, yaml.safe_load(f))

    training_ts = ks_config.generate(
        n_steps=train_config.n_samples, seed=train_config.seed
    )
    training_domain = PeriodicDomain(
        data=training_ts,
        output_size=train_config.subdomain_output_size,
        overlap=train_config.subdomain_overlap,
        subdomain_axis=1,
    )
    with open(args.reservoir_config, "r") as f:
        res_config_dict_ = yaml.safe_load(f)
        reservoir_configs = []
        for i in range(training_domain.n_subdomains):
            res_config_dict_.update({"seed": train_config.seed + i})
            reservoir_configs.append(
                dacite.from_dict(ReservoirHyperparameters, res_config_dict_)
            )

    subdomain_reservoirs = Parallel(n_jobs=train_config.n_jobs, verbose=True)(
        delayed(_initialize_reservoir)(res_config) for res_config in reservoir_configs
    )
    subdomain_predictors = train_subdomain_predictors(
        domain=training_domain,
        reservoirs=subdomain_reservoirs,
        train_config=train_config,
    )
    print(subdomain_predictors)
