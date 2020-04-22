import fsspec
import joblib
import logging
import os
import yaml

from dataclasses import dataclass
from typing import List

from fv3net.regression.dataset_handler import BatchGenerator
from fv3net.regression.sklearn.wrapper import SklearnWrapper, RegressorEnsemble
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("ml_training.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


@dataclass
class ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info

    """

    model_type: str
    gcs_data_dir: str
    hyperparameters: dict
    num_batches: int
    files_per_batch: int
    input_variables: List[str]
    output_variables: List[str]
    gcs_project: str = "vcm-ml"
    random_seed: int = 0
    mask_to_surface_type: str = "none"
    coord_z_center: str = "z"
    init_time_dim: str = "initial_time"

    def __post_init__(self):
        # set default random_state for sklearn model if not specified
        if "random_state" not in self.hyperparameters:
            self.hyperparameters["random_state"] = 0

    def validate_number_train_batches(self, batch_generator):
        """ Since number of training files specified may be larger than
        the actual number available, this adds an attribute num_batches_used
        that keeps information about the actual number of training batches
        used.

        Args:
            batch_generator (BatchGenerator)
        """
        self.num_batches_used = batch_generator.num_batches


def load_model_training_config(config_path, gcs_data_dir):
    """

    Args:
        config_path: location of .yaml that contains config for model training

    Returns:
        ModelTrainingConfig object
    """
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")
    config_dict["gcs_data_dir"] = gcs_data_dir
    config = ModelTrainingConfig(**config_dict)
    return config


def load_data_generator(train_config) -> BatchGenerator:
    """

    Args:
        train_config: ModelTrainingConfig object

    Returns:
        iterator that generates xr datasets for training batches
    """
    data_vars = train_config.input_variables + train_config.output_variables
    ds_batches = BatchGenerator(
        data_vars,
        train_config.gcs_data_dir,
        train_config.files_per_batch,
        train_config.num_batches,
        random_seed=train_config.random_seed,
        mask_to_surface_type=train_config.mask_to_surface_type,
    )
    return ds_batches


def _get_regressor(train_config):
    """Reads the model type from the model training config and initializes a regressor
    with hyperparameters from config.

    Args:
        train_config: ModelTrainingConfig object

    Returns:
        regressor (varies depending on model type)
    """
    model_type = train_config.model_type.replace(" ", "").replace("_", "")
    if "rf" in model_type or "randomforest" in model_type:
        from sklearn.ensemble import RandomForestRegressor

        regressor = RandomForestRegressor(**train_config.hyperparameters, n_jobs=-1)
    else:
        raise ValueError(
            f"Model type {train_config.model_type} not implemented. "
            "Options are "
            " 1) random forest (contains keywords 'rf' "
            "or 'random forest') "
        )
    return regressor


def train_model(batched_data: BatchGenerator, train_config):
    """

    Args:
        batched_data: iterator that yields training batch datasets
        train_config: ModelTrainingConfig object
        targets_for_normalization: array of sample output data used to save norm and std
            dev to the StandardScaler transformer

    Returns:
        trained sklearn model wrapper object
    """
    base_regressor = _get_regressor(train_config)
    target_transformer = StandardScaler()
    transform_regressor = TransformedTargetRegressor(base_regressor, target_transformer)
    batch_regressor = RegressorEnsemble(transform_regressor)
    model_wrapper = SklearnWrapper(batch_regressor)

    # count number of timesteps used for training within train_model
    # in case any batches fail
    train_config.validate_number_train_batches(batched_data)
    training_urls_to_attempt = batched_data.train_file_batches
    training_urls_used = []

    for i, batch in enumerate(
        batched_data.generate_batches(
            train_config.coord_z_center, train_config.init_time_dim
        )
    ):
        logger.info(f"Fitting batch {i}/{batched_data.num_batches}")
        model_wrapper.fit(
            input_vars=train_config.input_variables,
            output_vars=train_config.output_variables,
            sample_dim="sample",
            data=batch,
        )
        logger.info(f"Batch {i} done fitting.")
        training_urls_used += training_urls_to_attempt[i]

    return model_wrapper, training_urls_used


def save_model(output_url, model, model_filename):
    """Save model to {output_url}/{model_filename} using joblib.dump"""
    fs, _, _ = fsspec.get_fs_token_paths(output_url)
    fs.makedirs(output_url, exist_ok=True)
    model_url = os.path.join(output_url, model_filename)
    with fs.open(model_url, "wb") as f:
        joblib.dump(model, f)
