import dataclasses
from typing import Hashable, Mapping
import logging
import io
import dacite
from fv3fit._shared.packer import PackingInfo, clip_sample
import numpy as np
from copy import copy
import xarray as xr
import fsspec
import joblib
from .._shared import (
    pack,
    pack_tfdataset,
    unpack,
    Predictor,
    register_training_function,
    PackerConfig,
)

from .._shared.input_sensitivity import InputSensitivity, RandomForestInputSensitivity
from .._shared.config import RandomForestHyperparameters
from .. import _shared
from .._shared import (
    scaler,
    stack_non_vertical,
    match_prediction_to_input_coords,
    SAMPLE_DIM_NAME,
)
import sklearn.base
import sklearn.ensemble
import tensorflow as tf
from fv3fit.typing import Batch
from fv3fit.tfdataset import apply_to_mapping, ensure_nd

from typing import Optional, Iterable, Sequence
import yaml
from vcm import safe


@register_training_function("sklearn_random_forest", RandomForestHyperparameters)
def train_random_forest(
    hyperparameters: RandomForestHyperparameters,
    train_batches: Sequence[xr.Dataset],
    validation_batches: Sequence[xr.Dataset],
):
    """
    Args:
        hyperparameters: configuration for training
        train_batches: batched data for training, must be stacked
            with at most one non-sample dimension
        validation_batches: ignored in this function
    """
    model = RandomForest(
        hyperparameters.input_variables,
        hyperparameters.output_variables,
        hyperparameters,
    )
    # TODO: make use of validation_batches to report validation loss
    model.fit(train_batches)
    return model


@_shared.io.register("sklearn")
class RandomForest(Predictor):
    def __init__(
        self,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        hyperparameters: RandomForestHyperparameters,
    ):
        super().__init__(input_variables, output_variables)
        batch_regressor = _RegressorEnsemble(
            sklearn.ensemble.RandomForestRegressor(
                # n_jobs != 1 is non-reproducible,
                # None uses joblib which is reproducible
                n_jobs=None,
                random_state=hyperparameters.random_state,
                n_estimators=hyperparameters.n_estimators,
                max_depth=hyperparameters.max_depth,
                min_samples_split=hyperparameters.min_samples_split,
                min_samples_leaf=hyperparameters.min_samples_leaf,
                max_features=hyperparameters.max_features,
            ),
            # pass n_jobs along here to use joblib
            n_jobs=hyperparameters.n_jobs,
        )
        self._model_wrapper = SklearnWrapper(
            input_variables,
            output_variables,
            model=batch_regressor,
            scaler_type=hyperparameters.scaler_type,
            scaler_kwargs=hyperparameters.scaler_kwargs,
            packer_config=hyperparameters.packer_config,
        )
        self.input_variables = self._model_wrapper.input_variables
        self.output_variables = self._model_wrapper.output_variables

    def fit(self, batches: Sequence[xr.Dataset]):
        return self._model_wrapper.fit(batches)

    def predict(self, features):
        return self._model_wrapper.predict(features)

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        self._model_wrapper.dump(path)

    @classmethod
    def load(cls, path: str) -> "SklearnWrapper":
        return SklearnWrapper.load(path)


class _RegressorEnsemble:
    """
    Ensemble of RandomForestRegressor objects that are each trained on a separate
    batch of data.
    """

    def __init__(
        self,
        base_regressor,
        n_jobs,
        regressors: Sequence[sklearn.base.BaseEstimator] = None,
    ) -> None:
        self.base_regressor = base_regressor
        self.regressors = regressors or []
        self.n_jobs = n_jobs

    @property
    def n_estimators(self):
        return len(self.regressors)

    def fit(self, features, outputs):
        """ Adds a base regressor fit on features to the ensemble

        Args:
            features: numpy array of features
            outputs: numpy array of targets

        Returns:

        """
        new_regressor = copy(self.base_regressor)
        # each regressor needs different randomness
        if hasattr(new_regressor, "random_state"):
            new_regressor.random_state += len(self.regressors)
        # loky is the process-based backend
        with joblib.parallel_backend("loky", n_jobs=self.n_jobs):
            new_regressor.fit(features, outputs)
        self.regressors.append(new_regressor)

    def predict(self, features):
        """
        Args:
            features: 2D numpy array of features to predict on

        Returns:
            2D numpy array of predictions with N rows corresponding to N input samples.
            Each row is the average ensemble prediction for that sample.
        """
        predictions = np.array(
            [regressor.predict(features) for regressor in self.regressors]
        )
        return np.mean(predictions, axis=0)

    def dumps(self) -> bytes:
        batch_regressor_components = {
            "regressors": self.regressors,
            "base_regressor": self.base_regressor,
            "n_jobs": self.n_jobs,
        }
        f = io.BytesIO()
        joblib.dump(batch_regressor_components, f)
        return f.getvalue()

    @classmethod
    def loads(cls, b: bytes) -> "_RegressorEnsemble":
        f = io.BytesIO(b)
        batch_regressor_components = joblib.load(f)
        regressors: Sequence[sklearn.base.BaseEstimator] = batch_regressor_components[
            "regressors"
        ]
        base_regressor = batch_regressor_components["base_regressor"]
        obj = cls(
            base_regressor=base_regressor,
            regressors=regressors,
            n_jobs=batch_regressor_components.get("n_jobs", 1),
        )
        return obj


def _get_iterator_size(iterator):
    size = 0
    for _ in iterator:
        size += 1
    return size


class SklearnWrapper(Predictor):
    """Wrap a SkLearn model for use with xarray

    """

    _PICKLE_NAME = "sklearn.pkl"
    _SCALER_NAME = "scaler.bin"
    _METADATA_NAME = "metadata.bin"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        model: _RegressorEnsemble,
        scaler_type: str = "standard",
        scaler_kwargs: Optional[Mapping] = None,
        packer_config: PackerConfig = PackerConfig({}),
    ) -> None:
        """
        Initialize the wrapper

        Args:
            input_variables: list of input variables
            output_variables: list of output variables
            model: a scikit learn regression model
        """
        super().__init__(input_variables, output_variables)
        if scaler_type != "standard":
            raise NotImplementedError("only 'standard' scaler_type is implemented")
        self.model = model
        self.scaler_type = scaler_type
        self.scaler_kwargs = scaler_kwargs or {}
        self.target_scaler: Optional[scaler.NormalizeTransform] = None
        self.packer_config = packer_config
        for name in self.packer_config.clip:
            if name in self.output_variables:
                raise NotImplementedError("Clipping for ML outputs is not implemented.")

    def __repr__(self):
        return "SklearnWrapper(\n%s)" % repr(self.model)

    def _init_target_scaler(self, target_data: np.ndarray):
        target_scaler = scaler.StandardScaler()
        target_scaler.fit(target_data)
        return target_scaler

    def fit(self, batches: tf.data.Dataset):
        logger = logging.getLogger("SklearnWrapper")
        logger.info(f"Fitting random forest")
        batches = batches.map(apply_to_mapping(ensure_nd(1)))

        batches_clipped = batches.map(clip_sample(self.packer_config.clip))
        x_dataset, _ = pack_tfdataset(
            batches_clipped, [str(item) for item in self.input_variables]
        )
        y_dataset, self.output_features_ = pack_tfdataset(
            batches, [str(item for item in self.output_variables)]
        )

        # put all data in one batch, implement batching later
        batch_size = _get_iterator_size(iter(x_dataset))
        X: Batch = next(iter(x_dataset.batch(batch_size)))
        y: Batch = next(iter(y_dataset.batch(batch_size)))
        if self.target_scaler is None:
            self.target_scaler = self._init_target_scaler(y)

        y = self.target_scaler.normalize(y)
        self.model.fit(X, y)
        logger.info(f"Random forest done fitting.")

    def _predict_on_stacked_data(self, stacked_data):
        X, _ = pack(
            stacked_data[self.input_variables], [SAMPLE_DIM_NAME], self.packer_config
        )
        y = self.model.predict(X)
        if self.target_scaler is not None:
            y = self.target_scaler.denormalize(y)
        else:
            raise ValueError("Target scaler not present.")
        return unpack(y, [SAMPLE_DIM_NAME], packing_info=self.output_features_)

    def predict(self, data):
        stacked_data = stack_non_vertical(
            safe.get_variables(data, self.input_variables)
        )

        stacked_output = self._predict_on_stacked_data(stacked_data)
        unstacked_output = stacked_output.assign_coords(
            {SAMPLE_DIM_NAME: stacked_data[SAMPLE_DIM_NAME]}
        ).unstack(SAMPLE_DIM_NAME)

        return match_prediction_to_input_coords(data, unstacked_output)

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]

        fs.makedirs(path, exist_ok=True)

        mapper = fs.get_mapper(path)
        mapper[self._PICKLE_NAME] = self.model.dumps()
        if self.target_scaler is not None:
            mapper[self._SCALER_NAME] = scaler.dumps(self.target_scaler).encode("UTF-8")

        metadata = {
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
            "output_features": dataclasses.asdict(self.output_features_),
            "packer_config": dataclasses.asdict(self.packer_config),
        }

        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")

    @classmethod
    def load(cls, path: str) -> "SklearnWrapper":
        """Load a model from a remote path"""
        mapper = fsspec.get_mapper(path)
        model = _RegressorEnsemble.loads(mapper[cls._PICKLE_NAME])

        scaler_str = mapper.get(cls._SCALER_NAME, b"")
        scaler_obj: Optional[scaler.NormalizeTransform]
        if scaler_str:
            scaler_obj = scaler.loads(scaler_str)
        else:
            scaler_obj = None

        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])
        input_variables = metadata["input_variables"]
        output_variables = metadata["output_variables"]
        output_features_dict = metadata["output_features"]
        packer_config_dict = metadata.get("packer_config", {})
        output_features_ = dacite.from_dict(PackingInfo, data=output_features_dict)
        packer_config = dacite.from_dict(PackerConfig, packer_config_dict)

        obj = cls(input_variables, output_variables, model, packer_config=packer_config)
        obj.target_scaler = scaler_obj
        obj.output_features_ = output_features_

        return obj

    @property
    def feature_importances(self) -> Sequence[float]:
        importances = []
        for member in self.model.regressors:
            importances.append(member.feature_importances_)
        return importances

    @property
    def mean_importances(self) -> np.ndarray:
        return np.array(self.feature_importances).mean(axis=0)

    @property
    def std_importances(self) -> np.ndarray:
        return np.array(self.feature_importances).std(axis=0)

    def input_sensitivity(self, stacked_sample: xr.Dataset) -> InputSensitivity:
        _, input_multiindex = pack(
            stacked_sample[self.input_variables], ["sample"], self.packer_config
        )
        feature_importances = {}
        for (name, feature_index), mean_importance, std_importance in zip(
            input_multiindex, self.mean_importances, self.std_importances
        ):
            if name not in feature_importances:
                feature_importances[name] = {
                    "indices": [feature_index],
                    "mean_importances": [mean_importance],
                    "std_importances": [std_importance],
                }
            else:
                feature_importances[name]["indices"].append(feature_index)
                feature_importances[name]["mean_importances"].append(mean_importance)
                feature_importances[name]["std_importances"].append(std_importance)

        formatted_feature_importances = {
            name: RandomForestInputSensitivity(
                indices=info["indices"],
                mean_importances=info["mean_importances"],
                std_importances=info["std_importances"],
            )
            for name, info in feature_importances.items()
        }
        return InputSensitivity(rf_feature_importances=formatted_feature_importances)
