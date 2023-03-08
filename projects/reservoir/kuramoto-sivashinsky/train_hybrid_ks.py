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
)
from fv3fit.reservoir.one_dim import (
    Reservoir1DTrainingConfig,
    HybridReservoirComputingModel,
)
from ks import KuramotoSivashinskyConfig, ImperfectKSModel
from train_ks import (
    transform_inputs_to_reservoir_states,
    get_parser,
    generate_training_time_series,
)

logger = logging.getLogger(__name__)


def add_imperfect_prediction_to_train_data(X, imperfect_prediction_states):
    return np.hstack(X, imperfect_prediction_states)


def get_imperfect_ks_model(hybrid_imperfect_model_config, reservoir_timestep):
    imperfect_ks_config = KuramotoSivashinskyConfig(**hybrid_imperfect_model_config)
    return ImperfectKSModel(
        config=imperfect_ks_config, reservoir_timestep=reservoir_timestep,
    )


def generate_imperfect_prediction_time_series(imperfect_model, ts_truth,) -> np.ndarray:
    """ Initialize imperfect model at the target state at time t, and save
    its predictions at the end of each reservoir timestep. First dimension is
    time, second is spatial.
    """
    imperfect_predictions = [
        ts_truth[0],
    ]
    for i in range(len(ts_truth) - 1):
        input = ts_truth[i]
        # the imperfect KS model handles the downsampling in time to the
        # reservoir timestep in its predict method
        imperfect_predictions.append(imperfect_model.predict(input))
    return np.array(imperfect_predictions)


def train_hybrid(ks_config, train_config):
    training_ts = generate_training_time_series(
        ks_config=ks_config,
        reservoir_timestep=train_config.timestep,
        n_samples=train_config.n_burn + train_config.n_samples,
        seed=train_config.seed,
        input_noise=train_config.input_noise,
    )
    imperfect_model = get_imperfect_ks_model(
        train_config.hybrid_imperfect_model_config,
        reservoir_timestep=train_config.timestep,
    )
    imperfect_training_ts = generate_imperfect_prediction_time_series(
        imperfect_model, training_ts,
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

    predictor = HybridReservoirComputingModel(reservoir=reservoir, readout=readout,)
    return predictor


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    with open(args.ks_config, "r") as f:
        ks_config = dacite.from_dict(KuramotoSivashinskyConfig, yaml.safe_load(f))
    with open(args.train_config, "r") as f:
        train_config_dict = yaml.safe_load(f)
        train_config = Reservoir1DTrainingConfig.from_dict(train_config_dict)

    predictor = train_hybrid(ks_config, train_config,)

    predictor.dump(args.output_path)
    train_config.dump(args.output_path)
    with fsspec.open(os.path.join(args.output_path, "ks_config.yaml"), "w") as f:
        yaml.dump(dataclasses.asdict(ks_config), f, indent=4)
