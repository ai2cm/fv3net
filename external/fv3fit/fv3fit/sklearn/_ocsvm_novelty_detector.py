import dataclasses
import io
import logging
import time
]import dacite
import fsspec
import joblib
import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline
import yaml
from fv3fit import _shared, tfdataset
from fv3fit._shared import stacking, SAMPLE_DIM_NAME
from fv3fit._shared.config import (
    OCSVMNoveltyDetectorHyperparameters,
    PackerConfig,
    register_training_function,
)
from fv3fit._shared.novelty_detector import NoveltyDetector
from fv3fit._shared.packer import PackingInfo, clip_sample, pack, pack_tfdataset
from fv3fit.tfdataset import apply_to_mapping, ensure_nd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import tensorflow as tf
import xarray as xr

from fv3fit._shared.predictor import Predictor
from fv3fit.typing import Batch


@register_training_function(
    "ocsvm_novelty_detector", OCSVMNoveltyDetectorHyperparameters
)
def train_ocsvm_novelty_detector(
    hyperparameters: OCSVMNoveltyDetectorHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: tf.data.Dataset,
):
    train_batches = train_batches.map(
        tfdataset.apply_to_mapping(tfdataset.float64_to_float32)
    )

    return OCSVMNoveltyDetector.fit(train_batches, hyperparameters)


@_shared.io.register("ocsvm")
class OCSVMNoveltyDetector(NoveltyDetector):

    _PICKLE_NAME = "ocsvm.pkl"
    _METADATA_NAME = "metadata.bin"
    logger = logging.getLogger("OCSVMNoveltyDetector")

    input_features_: PackingInfo
    is_trained: bool = False
    maximum_training_score: float
    pipeline: Pipeline

    def __init__(self, hyperparameters: OCSVMNoveltyDetectorHyperparameters):
        super().__init__(hyperparameters.input_variables)
        self.packer_config = hyperparameters.packer_config
        self.gamma = hyperparameters.gamma
        self.nu = hyperparameters.nu
        self.max_iter = hyperparameters.max_iter

    @classmethod
    def fit(
        self,
        batches: tf.data.Dataset,
        hyperparameters: OCSVMNoveltyDetectorHyperparameters,
    ) -> "OCSVMNoveltyDetector":
        model = OCSVMNoveltyDetector(hyperparameters)

        start_time = time.time()
        model.logger.info(f"Fitting OCSVM novelty detector.")

        batches = batches.map(apply_to_mapping(ensure_nd(2)))
        batches_clipped = batches.map(clip_sample(model.packer_config.clip))
        x_dataset, model.input_features_ = pack_tfdataset(
            batches_clipped, [str(item) for item in model.input_variables]
        )
        x_dataset = x_dataset.unbatch()

        total_samples = sum(1 for _ in iter(x_dataset))
        X: Batch = next(iter(x_dataset.batch(total_samples)))

        scaler = StandardScaler()
        ocsvm = OneClassSVM(
            kernel="rbf", gamma=model.gamma, nu=model.nu, max_iter=model.max_iter
        )
        model.pipeline = make_pipeline(scaler, ocsvm)
        model.pipeline.fit(X)

        seconds = time.time() - start_time
        model.logger.info(f"OCSVM novelty detector done fitting in {seconds}s.")

        model._compute_max_score_train(X)
        model.is_trained = True
        return model

    def _compute_max_score_train(self, X):
        self.logger.info(f"Computing maximum scores on training data.")
        start_time = time.time()

        # We negate the scores to ensure that larger scores correspond to a higher
        # likelihood of being an outlier
        scores = -1 * self.pipeline.score_samples(X)
        max_score = np.max(scores)
        self.maximum_training_score = float(max_score)

        seconds = time.time() - start_time
        self.logger.info(f"Computed largest score of {max_score} in {seconds}s.")

    def _get_default_cutoff(self):
        return self.maximum_training_score

    def predict(self, data: xr.Dataset) -> xr.Dataset:
        assert self.is_trained

        start_time = time.time()
        self.logger.info(f"Predicting with OCSVM novelty detector.")

        stack_dims = [dim for dim in data.dims if dim not in stacking.Z_DIM_NAMES]
        stacked_data = data.stack({SAMPLE_DIM_NAME: stack_dims})
        stacked_data = stacked_data.transpose(SAMPLE_DIM_NAME, ...)

        X, _ = pack(
            stacked_data[self.input_variables], [SAMPLE_DIM_NAME], self.packer_config
        )
        stacked_scores = -1 * self.pipeline.score_samples(X)

        new_coords = {
            k: v
            for (k, v) in stacked_data.coords.items()
            if k not in stacking.Z_DIM_NAMES
        }
        stacked_scores = xr.DataArray(
            stacked_scores, dims=[SAMPLE_DIM_NAME], coords=new_coords
        )
        score_dataset = stacked_scores.to_dataset(name=self._SCORE_OUTPUT_VAR).unstack(
            SAMPLE_DIM_NAME
        )

        seconds = time.time() - start_time
        self.logger.info(
            f"Finished predicting with OCSVM novelty detector in {seconds}s."
        )

        return stacking.match_prediction_to_input_coords(data, score_dataset)

    def dump(self, path: str) -> None:
        assert self.is_trained

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)

        mapper = fs.get_mapper(path)

        f = io.BytesIO()
        joblib.dump({"pipeline": self.pipeline}, f)
        mapper[self._PICKLE_NAME] = f.getvalue()

        metadata = {
            "input_variables": self.input_variables,
            "packer_config": dataclasses.asdict(self.packer_config),
            "input_features_": dataclasses.asdict(self.input_features_),
            "gamma": self.gamma,
            "nu": self.nu,
            "max_iter": self.max_iter,
            "maximum_training_score": self.maximum_training_score,
        }
        print(metadata)
        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")

    @classmethod
    def load(cls, path: str) -> "Predictor":
        mapper = fsspec.get_mapper(path)
        components = joblib.load(io.BytesIO(mapper[cls._PICKLE_NAME]))
        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])

        hyperparameters = OCSVMNoveltyDetectorHyperparameters(
            metadata["input_variables"]
        )
        obj = cls(hyperparameters)
        obj.pipeline = components["pipeline"]
        obj.input_features_ = dacite.from_dict(
            PackingInfo, data=metadata["input_features_"]
        )
        obj.packer_config = dacite.from_dict(
            PackerConfig, data=metadata.get("packer_config", {})
        )
        obj.gamma = metadata["gamma"]
        obj.nu = metadata["nu"]
        obj.max_iter = metadata["max_iter"]
        obj.maximum_training_score = metadata["maximum_training_score"]
        obj.is_trained = True

        return obj
