import argparse
from dataclasses import dataclass
from typing import List
import xarray as xr
import yaml

from ..dataset_handler import BatchGenerator
from .wrapper import SklearnWrapper


@dataclass
class ModelTrainingConfig:
    model_type: str
    train_data_path: str
    hyperparameters: dict
    random_seed: int
    num_batches: int
    batch_size: int
    train_frac: float
    test_frac: float
    input_variables: List[str]
    output_variables: List[str]


def _load_model_training_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")
    config = ModelTrainingConfig(**config_dict)
    return config


def _load_training_data(train_config):
    ds_full = xr.open_zarr(train_config.train_data_path)
    ds_batches = BatchGenerator(
        ds_full,
        train_config.batch_size,
        train_config.train_frac,
        train_config.test_frac,
        train_config.num_batches
    )
    return ds_batches


def _get_regressor(train_config):
    model_type = train_config.model_type.replace(' ', '').replace('_', '')
    if 'rf' in model_type or 'randomforest' in model_type:
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(**train_config.model_params, n_jobs=-1)
    elif 'gbt' in model_type or 'boostedtrees' in model_type:
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        regressor = MultiOutputRegressor(
            XGBRegressor(**train_config.model_params), n_jobs=-1)
    else:
        raise ValueError(f"Model type {train_config.model_type} not implemented. "
                         "Options are random forest (contains keywords 'rf' "
                         "or 'random forest') or gradient boosted trees "
                         "(contains keywords 'gbt' or 'boosted trees').")
    return regressor


def train_model(batched_data, train_config):
    regressor = _get_regressor(train_config)
    model = SklearnWrapper(regressor)
    for i, batch in enumerate(batched_data):
        if i > 0:
            model.add_new_batch_estimators()
        model.fit(
            input_vars=train_config.input_variables,
            output_vars=train_config.output_variables,
            sample_dim='sample',
            data=batched_data
        )
    return model


