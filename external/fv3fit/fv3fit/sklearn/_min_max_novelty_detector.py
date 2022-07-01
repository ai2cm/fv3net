from fv3fit._shared import stacking
from .. import _shared
from .._shared import (
    match_prediction_to_input_coords,
    pack,
    pack_tfdataset,
    Predictor,
    register_training_function,
    SAMPLE_DIM_NAME,
    stack_non_vertical,
)
from .._shared.config import MinMaxNoveltyDetectorHyperparameters, PackerConfig

import dataclasses
import dacite
import fsspec
from fv3fit import tfdataset
from fv3fit._shared.packer import PackingInfo, clip_sample, unpack
from fv3fit.tfdataset import apply_to_mapping, ensure_nd
from fv3fit.typing import Batch
import io
import joblib
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from typing import Optional, Iterable, Sequence
from vcm import safe
import xarray as xr
import yaml

@register_training_function("min_max_novelty_detector", MinMaxNoveltyDetectorHyperparameters)
def train_min_max_novelty_detector(
    hyperparameters: MinMaxNoveltyDetectorHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: tf.data.Dataset
):
    train_batches = train_batches.map(
        tfdataset.apply_to_mapping(tfdataset.float64_to_float32)
    )
    
    model = MinMaxNoveltyDetector(hyperparameters)
    model.fit(train_batches)
    return model

@_shared.io.register("minmax")
class MinMaxNoveltyDetector(Predictor):

    _PICKLE_NAME = "minmax.pkl"
    _METADATA_NAME = "metadata.bin"

    _NOVELTY_OUTPUT_VAR = "is_novelty"

    dims = ["x", "y", "tile", "time"]

    def __init__(
        self,
        hyperparameters: MinMaxNoveltyDetectorHyperparameters
    ):
        output_variables = [self._NOVELTY_OUTPUT_VAR]
        super().__init__(hyperparameters.input_variables, output_variables)
        self.packer_config = hyperparameters.packer_config

    def fit(self, batches: tf.data.Dataset):
        logger = logging.getLogger("MinMaxNoveltyDetector")
        logger.info(f"Fitting min-max novelty detector.")

        batches = batches.map(apply_to_mapping(ensure_nd(2)))
        batches_clipped = batches.map(clip_sample(self.packer_config.clip))
        x_dataset, self.input_features_ = pack_tfdataset(
            batches_clipped, [str(item) for item in self.input_variables]
        )
        x_dataset = x_dataset.unbatch()

        total_samples = sum(1 for _ in iter(x_dataset))
        X: Batch = next(iter(x_dataset.batch(total_samples)))

        self.scaler = MinMaxScaler()
        self.scaler.fit(X)

        logger.info(f"Min-max novelty detector done fitting.")

    def predict(self, data: xr.Dataset) -> xr.Dataset:  
        stack_dims = [dim for dim in data.dims if dim not in stacking.Z_DIM_NAMES]  
        stacked_data = data.stack({SAMPLE_DIM_NAME: stack_dims}).transpose(SAMPLE_DIM_NAME, ...)

        X, _ = pack(stacked_data[self.input_variables], [SAMPLE_DIM_NAME], self.packer_config)
        scaled_X = self.scaler.transform(X)
        larger_than_max = np.where(scaled_X.max(axis=1) > 1, 1, 0)
        smaller_than_min = np.where(scaled_X.min(axis=1) < 0, 1, 0)
        stacked_is_novelty = larger_than_max + smaller_than_min

        new_coords = {k: v for (k, v) in stacked_data.coords.items() if k not in stacking.Z_DIM_NAMES}
        stacked_is_novelty = xr.DataArray(stacked_is_novelty, dims=[SAMPLE_DIM_NAME], coords=new_coords)
        is_novelty = stacked_is_novelty\
            .to_dataset(name=self._NOVELTY_OUTPUT_VAR)\
            .unstack(SAMPLE_DIM_NAME)

        return match_prediction_to_input_coords(data, is_novelty)

    def dump(self, path: str) -> None:
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)

        mapper = fs.get_mapper(path)

        f = io.BytesIO()
        joblib.dump({"scaler": self.scaler}, f)
        mapper[self._PICKLE_NAME] = f.getvalue()

        metadata = {
            "input_variables": self.input_variables,
            "packer_config": dataclasses.asdict(self.packer_config),
            "input_features_": dataclasses.asdict(self.input_features_),
        }
        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")

    @classmethod
    def load(cls, path: str) -> "Predictor":
        mapper = fsspec.get_mapper(path)
        components = joblib.load(io.BytesIO(mapper[cls._PICKLE_NAME]))
        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])

        hyperparameters = MinMaxNoveltyDetectorHyperparameters(metadata["input_variables"])
        obj = cls(hyperparameters)
        obj.scaler = components["scaler"]
        obj.input_features_ = dacite.from_dict(PackingInfo, data=metadata["input_features_"])
        obj.packer_config = dacite.from_dict(PackerConfig, data=metadata.get("packer_config", {}))

        return obj        

