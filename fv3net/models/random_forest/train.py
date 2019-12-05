import logging
import pprint

import numpy as np
import yaml
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import fire
from fv3net.machine_learning.sklearn.wrapper import SklearnWrapper
from vcm.convenience import open_dataset

logging.basicConfig(level=logging.INFO)

default_kwargs = dict(
    input_variables=("qv", "temp", "lhflx", "shflx"),
    output_variables=("q1", "q2"),
    sample_dimension="sample",
    training_data_tag="1degTrain",
    latitude_range=slice(-80, 80),
    n_jobs=4,
    min_samples_leaf=10,
    n_estimators=50,
    n_train=5000,
    n_test=10000,
    seed=1,
)


def read_yaml(path):
    with open(path) as f:
        return yaml.load(f)


def train(
    model_path,
    input_variables=("qv", "temp", "lhflx", "shflx"),
    output_variables=("q1", "q2"),
    sample_dimension="sample",
    training_data_tag="1degTrain",
    latitude_range=slice(-80, 80),
    n_jobs=4,
    min_samples_leaf=10,
    n_estimators=50,
    n_train=5000,
    n_test=10000,
    seed=1,
):
    """Train a random forest parameterization"""

    np.random.seed(seed)
    ds = open_dataset(training_data_tag)
    ds = ds.sel(lat=latitude_range).dropna("time")

    flat = ds.stack(sample=["time", "lat", "lon"]).load()
    n = len(flat.sample)
    ind = np.random.choice(n, n)
    shuffled = flat.isel(sample=ind)

    train = shuffled.isel(sample=slice(0, n_train))
    # test = shuffled.isel(sample=slice(n_train, n_train + n_test))

    sklearn_model = TransformedTargetRegressor(
        RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=n_jobs, min_samples_leaf=min_samples_leaf
        ),
        StandardScaler(),
    )
    model = SklearnWrapper(sklearn_model)

    logging.info("Fitting model:")
    logging.info(model)
    model.fit(list(input_variables), list(output_variables), sample_dimension, train)

    logging.info("Saving to " + model_path)
    joblib.dump(model, model_path)


def main(output_path: str, options=None):
    """Train a random forest parameterization

    Args:
        output_path: path to save trained model
        options: path to yaml file with options
    """
    opts = default_kwargs.copy()
    if options is not None:
        opts.update(read_yaml(options))
    logging.info("Training with these options: \n%s" % pprint.pformat(opts))

    return train(output_path, **opts)


if __name__ == "__main__":
    fire.Fire(main)
