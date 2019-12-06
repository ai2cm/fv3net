import argparse
from dataclasses import dataclass
import joblib
import time
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
    num_batches: int
    batch_size: int
    train_frac: float
    test_frac: float
    input_variables: List[str]
    output_variables: List[str]
    gcs_project: str='vcm-ml'


def load_model_training_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")
    config = ModelTrainingConfig(**config_dict)
    return config


def load_training_data(train_config):
    if train_config.train_data_path[:5]=="gs://":
        import gcsfs
        fs = gcsfs.GCSFileSystem(project=train_config.gcs_project)
        train_data_path = fs.get_mapper(train_config.train_data_path)
    else:
        train_data_path = train_config.train_data_path
    ds_full = xr.open_zarr(train_data_path)
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
        regressor = RandomForestRegressor(**train_config.hyperparameters, n_jobs=-1)
    elif 'gbt' in model_type or 'boostedtrees' in model_type:
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        regressor = MultiOutputRegressor(
            XGBRegressor(**train_config.hyperparameters), n_jobs=-1)
    else:
        raise ValueError(f"Model type {train_config.model_type} not implemented. "
                         "Options are random forest (contains keywords 'rf' "
                         "or 'random forest') or gradient boosted trees "
                         "(contains keywords 'gbt' or 'boosted trees').")
    return regressor


def train_model(batched_data, train_config):
    regressor = _get_regressor(train_config)
    model = SklearnWrapper(regressor)
    for i, batch in enumerate(batched_data.generate_train_batches()):
        if i > 0:
            model.add_new_batch_estimators()
        t0 = time.time()
        print(f"Fitting batch {i}/{batched_data.num_batches}")
        model.fit(
            input_vars=train_config.input_variables,
            output_vars=train_config.output_variables,
            sample_dim='sample',
            data=batch
        )
        print(f"Batch {i} done fitting, took {int(time.time()-t0)} s.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-config-path",
        type=str,
        required=True,
        help="Path for training configuration yaml file"
    )
    parser.add_argument(
        "--model-output-path",
        type=str,
        required=True,
        help="Path for writing trained model"
    )
    args = parser.parse_args()
    train_config = load_model_training_config(args.train_config_file)
    batched_data = load_training_data(train_config)
    model = train_model(batched_data, train_config)
    joblib.dump(model, args.model_output_path)

