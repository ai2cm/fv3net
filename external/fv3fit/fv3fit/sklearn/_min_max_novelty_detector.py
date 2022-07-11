from fv3fit._shared import stacking
from fv3fit._shared.novelty_detector import NoveltyDetector
from .. import _shared
from .._shared import (
    match_prediction_to_input_coords,
    pack,
    pack_tfdataset,
    Predictor,
    register_training_function,
    SAMPLE_DIM_NAME,
)
from .._shared.config import MinMaxNoveltyDetectorHyperparameters, PackerConfig

import dataclasses
import dacite
import fsspec
from fv3fit import tfdataset
from fv3fit._shared.packer import PackingInfo, clip_sample
from fv3fit.tfdataset import apply_to_mapping, ensure_nd
from fv3fit.typing import Batch
import io
import joblib
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import xarray as xr
import yaml


@register_training_function(
    "min_max_novelty_detector", MinMaxNoveltyDetectorHyperparameters
)
def train_min_max_novelty_detector(
    hyperparameters: MinMaxNoveltyDetectorHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: tf.data.Dataset,
):
    train_batches = train_batches.map(
        tfdataset.apply_to_mapping(tfdataset.float64_to_float32)
    )

    return MinMaxNoveltyDetector.fit(train_batches, hyperparameters)


@_shared.io.register("minmax")
class MinMaxNoveltyDetector(NoveltyDetector):

    _PICKLE_NAME = "minmax.pkl"
    _METADATA_NAME = "metadata.bin"

    scaler: MinMaxScaler
    input_features_: PackingInfo
    is_trained: bool = False

    def __init__(self, hyperparameters: MinMaxNoveltyDetectorHyperparameters):
        super().__init__(hyperparameters.input_variables)
        self.packer_config = hyperparameters.packer_config

    @classmethod
    def fit(
        self,
        batches: tf.data.Dataset,
        hyperparameters: MinMaxNoveltyDetectorHyperparameters,
    ) -> "MinMaxNoveltyDetector":
        logger = logging.getLogger("MinMaxNoveltyDetector")
        logger.info(f"Fitting min-max novelty detector.")

        model = MinMaxNoveltyDetector(hyperparameters)

        batches = batches.map(apply_to_mapping(ensure_nd(2)))
        batches_clipped = batches.map(clip_sample(model.packer_config.clip))
        x_dataset, model.input_features_ = pack_tfdataset(
            batches_clipped, [str(item) for item in model.input_variables]
        )
        x_dataset = x_dataset.unbatch()

        total_samples = sum(1 for _ in iter(x_dataset))
        X: Batch = next(iter(x_dataset.batch(total_samples)))

        model.scaler = MinMaxScaler()
        model.scaler.fit(X)

        logger.info(f"Min-max novelty detector done fitting.")
        model.is_trained = True
        return model

    def predict(self, data: xr.Dataset) -> xr.Dataset:
        """
        For each coordinate c, computes a score with the following expression:
        score(c) = max(0, (c - c_max) / c_max) + max(0, (c_min - c) / c_min).
        A total scores over a column is taken by taking the maximum score.
        If the score is greater than 0, then at least one coordinate is out of range.
        """
        assert self.is_trained

        scaled_X, coords = self.predict_pre_aggregation(data)

        scores_larger_than_max = np.maximum(scaled_X.max(axis=1) - 1, 0)
        scores_smaller_than_min = np.maximum(-1 * scaled_X.min(axis=1), 0)
        stacked_scores = scores_larger_than_max + scores_smaller_than_min

        new_coords = {
            k: v for (k, v) in coords.items() if k not in stacking.Z_DIM_NAMES
        }
        stacked_scores = xr.DataArray(
            stacked_scores, dims=[SAMPLE_DIM_NAME], coords=new_coords
        )
        score_dataset = stacked_scores.to_dataset(name=self._SCORE_OUTPUT_VAR).unstack(
            SAMPLE_DIM_NAME
        )

        return match_prediction_to_input_coords(data, score_dataset)

    def predict_pre_aggregation(self, data: xr.Dataset):
        stack_dims = [dim for dim in data.dims if dim not in stacking.Z_DIM_NAMES]
        stacked_data = data.stack({SAMPLE_DIM_NAME: stack_dims})
        stacked_data = stacked_data.transpose(SAMPLE_DIM_NAME, ...)

        X, _ = pack(
            stacked_data[self.input_variables], [SAMPLE_DIM_NAME], self.packer_config
        )
        stacked_data.coords
        return self.scaler.transform(X), stacked_data.coords

    def dump(self, path: str) -> None:
        assert self.is_trained

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

        hyperparameters = MinMaxNoveltyDetectorHyperparameters(
            metadata["input_variables"]
        )
        obj = cls(hyperparameters)
        obj.scaler = components["scaler"]
        obj.input_features_ = dacite.from_dict(
            PackingInfo, data=metadata["input_features_"]
        )
        obj.packer_config = dacite.from_dict(
            PackerConfig, data=metadata.get("packer_config", {})
        )
        obj.is_trained = True

        return obj
