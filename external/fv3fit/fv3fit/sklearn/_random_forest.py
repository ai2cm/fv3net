from typing import Mapping
import logging
import io
import numpy as np
from copy import copy
import xarray as xr
import fsspec
import joblib
from .._shared import (
    pack,
    unpack,
    Predictor,
    get_scaler,
    register_training_function,
    multiindex_to_tuple,
    tuple_to_multiindex,
)
from .._shared.config import RandomForestHyperparameters
from .. import _shared
from .._shared import (
    scaler,
    StackedBatches,
    stack_non_vertical,
    match_prediction_to_input_coords,
    SAMPLE_DIM_NAME,
)
import sklearn.base
import sklearn.ensemble

from typing import Optional, Iterable, Sequence
import yaml
from vcm import safe


def _parse_metadata_backward_compatible(metadata: dict) -> tuple:
    # first two cases here for backward compatibility (https://github.com/ai2cm/fv3net/issues/1403) # noqa: E501
    if isinstance(metadata, list) and len(metadata) == 3:
        (input_variables, output_variables, output_features_tuple,) = metadata
    elif isinstance(metadata, list) and len(metadata) == 4:
        (input_variables, output_variables, output_features_tuple,) = metadata[1:]
    else:
        input_variables = metadata["input_variables"]
        output_variables = metadata["output_variables"]
        output_features_tuple = metadata["output_features"]
    return input_variables, output_variables, output_features_tuple


@register_training_function("sklearn_random_forest", RandomForestHyperparameters)
def train_random_forest(
    hyperparameters: RandomForestHyperparameters,
    train_batches: Sequence[xr.Dataset],
    validation_batches: Sequence[xr.Dataset],
):
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


class SklearnWrapper(Predictor):
    """Wrap a SkLearn model for use with xarray

    """

    _PICKLE_NAME = "sklearn.pkl"
    _SCALER_NAME = "scaler.bin"
    _METADATA_NAME = "metadata.bin"

    def __init__(
        self,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        model: _RegressorEnsemble,
        scaler_type: str = "standard",
        scaler_kwargs: Optional[Mapping] = None,
    ) -> None:
        """
        Initialize the wrapper

        Args:
            input_variables: list of input variables
            output_variables: list of output variables
            model: a scikit learn regression model
        """
        super().__init__(input_variables, output_variables)
        self.model = model
        self.scaler_type = scaler_type
        self.scaler_kwargs = scaler_kwargs or {}
        self.target_scaler: Optional[scaler.NormalizeTransform] = None
        self._input_variables = input_variables
        self._output_variables = output_variables

    def __repr__(self):
        return "SklearnWrapper(\n%s)" % repr(self.model)

    def _fit_batch(self, data: xr.Dataset):
        x, _ = pack(
            data[self.input_variables], SAMPLE_DIM_NAME  # type: ignore
        )
        y, self.output_features_ = pack(
            data[self.output_variables], SAMPLE_DIM_NAME  # type: ignore
        )
        # https://github.com/pydata/xarray/pull/4144

        if self.target_scaler is None:
            self.target_scaler = self._init_target_scaler(data)

        y = self.target_scaler.normalize(y)
        self.model.fit(x, y)

    def _init_target_scaler(self, batch):
        return get_scaler(
            self.scaler_type,
            self.scaler_kwargs,
            batch,
            self._output_variables,
            SAMPLE_DIM_NAME,
        )

    def fit(self, batches: Sequence[xr.Dataset]):
        logger = logging.getLogger("SklearnWrapper")
        random_state = np.random.RandomState(np.random.get_state()[1][0])
        stacked_batches = StackedBatches(batches, random_state)
        for i, batch in enumerate(stacked_batches):
            logger.info(f"Fitting batch {i+1}/{len(batches)}")
            self._fit_batch(batch)
            logger.info(f"Batch {i+1} done fitting.")

    def _predict_on_stacked_data(self, stacked_data):
        X, _ = pack(stacked_data[self.input_variables], SAMPLE_DIM_NAME)
        y = self.model.predict(X)
        if self.target_scaler is not None:
            y = self.target_scaler.denormalize(y)
        else:
            raise ValueError("Target scaler not present.")
        return unpack(y, SAMPLE_DIM_NAME, self.output_features_)

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
            "output_features": multiindex_to_tuple(self.output_features_),
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
        (
            input_variables,
            output_variables,
            output_features_tuple,
        ) = _parse_metadata_backward_compatible(metadata)
        output_features_ = tuple_to_multiindex(output_features_tuple)

        obj = cls(input_variables, output_variables, model)
        obj.target_scaler = scaler_obj
        obj.output_features_ = output_features_

        return obj
