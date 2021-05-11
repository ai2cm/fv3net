from typing import Mapping
import logging
import io
from copy import copy
import numpy as np
import xarray as xr
import pandas as pd
import fsspec
import joblib
from .._shared import pack, unpack, Estimator, get_scaler
from .. import _shared
from .._shared import scaler
import sklearn.base

from typing import Optional, Iterable, Sequence, List, Tuple
import yaml


logger = logging.getLogger("SklearnWrapper")


def _multiindex_to_tuple(index: pd.MultiIndex) -> tuple:
    return list(index.names), list(index.to_list())


def _tuple_to_multiindex(d: tuple) -> pd.MultiIndex:
    names, list_ = d
    return pd.MultiIndex.from_tuples(list_, names=names)


class RegressorEnsemble:
    """Ensemble of regressors that are incrementally trained in batches

    """

    def __init__(
        self, base_regressor, regressors: Sequence[sklearn.base.BaseEstimator] = None,
    ) -> None:
        self.base_regressor = base_regressor
        self.regressors = regressors or []

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
        }
        f = io.BytesIO()
        joblib.dump(batch_regressor_components, f)
        return f.getvalue()

    @classmethod
    def loads(cls, b: bytes) -> "RegressorEnsemble":
        f = io.BytesIO(b)
        batch_regressor_components = joblib.load(f)
        regressors: Sequence[sklearn.base.BaseEstimator] = batch_regressor_components[
            "regressors"
        ]
        base_regressor = batch_regressor_components["base_regressor"]
        obj = cls(base_regressor=base_regressor, regressors=regressors)
        return obj


def _is_feature(x):
    name, i = x
    if name == "specific_humidity":
        return i >= 30
    elif name == "air_temperature":
        return i >= 20
    elif name == "northward_wind":
        return i >= 10
    else:
        return True


def select(idx: List[Tuple[str, int]], data: np.ndarray, strategy):
    if strategy == "all":
        return data
    elif strategy == "lower":
        return data[:, [_is_feature(i) for i in idx]]
    else:
        raise NotImplementedError(f"{strategy} not implemented")


@_shared.io.register("sklearn")
class SklearnWrapper(Estimator):
    """Wrap a SkLearn model for use with xarray

    """

    _PICKLE_NAME = "sklearn.pkl"
    _SCALER_NAME = "scaler.bin"
    _METADATA_NAME = "metadata.bin"
    _SELECT_LEVELS = "select_levels"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        model: RegressorEnsemble,
        parallel_backend: str = "threading",
        scaler_type: str = "standard",
        scaler_kwargs: Optional[Mapping] = None,
        target_scaler: Optional[scaler.NormalizeTransform] = None,
        select_levels="all",
        n_jobs: int = 1,
    ) -> None:
        """
        Initialize the wrapper

        Args:
            sample_dim_name: dimension over which samples are taken
            input_variables: list of input variables
            output_variables: list of output variables
            model: a scikit learn regression model
            select_levels: "all" or "lower"
        """
        self._sample_dim_name = sample_dim_name
        self._input_variables = input_variables
        self._output_variables = output_variables
        self.model = model

        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.scaler_type = scaler_type
        self.scaler_kwargs = scaler_kwargs or {}
        self.target_scaler: Optional[scaler.NormalizeTransform] = target_scaler
        self.select_levels = select_levels
        logger.info(f"Initialized {self}")

    def __repr__(self):
        return f"SklearnWrapper({vars(self)})"

    def _pack_inputs(self, data):
        x, index = pack(data[self.input_variables], self.sample_dim_name)
        return select(list(index), x, self.select_levels)

    def _fit_batch(self, data: xr.Dataset):
        x = self._pack_inputs(data)
        # TODO the sample_dim can change so best to use feature dim to flatten
        y, self.output_features_ = pack(
            data[self.output_variables], self.sample_dim_name
        )

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
            self._sample_dim_name,
        )

    def fit(self, batches: Sequence[xr.Dataset]):
        for i, batch in enumerate(batches):
            logger.info(f"Fitting batch {i}/{len(batches)}")
            self._fit_batch(batch)
            logger.info(f"Batch {i} done fitting.")

    def predict(self, data):
        x = self._pack_inputs(data)
        with joblib.parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            y = self.model.predict(x)

            if self.target_scaler is not None:
                y = self.target_scaler.denormalize(y)
            else:
                raise ValueError("Target scaler not present.")

        ds = unpack(y, self.sample_dim_name, self.output_features_)
        return ds.assign_coords({self.sample_dim_name: data[self.sample_dim_name]})

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

        metadata = [
            self.sample_dim_name,
            self.input_variables,
            self.output_variables,
            _multiindex_to_tuple(self.output_features_),
        ]

        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")
        mapper[self._SELECT_LEVELS] = self.select_levels.encode("UTF-8")

    @classmethod
    def load(cls, path: str) -> "SklearnWrapper":
        """Load a model from a remote path"""
        mapper = fsspec.get_mapper(path)
        model = RegressorEnsemble.loads(mapper[cls._PICKLE_NAME])

        scaler_str = mapper.get(cls._SCALER_NAME, b"")
        scaler_obj: Optional[scaler.NormalizeTransform]
        if scaler_str:
            scaler_obj = scaler.loads(scaler_str)
        else:
            scaler_obj = None
        (
            sample_dim_name,
            input_variables,
            output_variables,
            output_features_dict_,
        ) = yaml.safe_load(mapper[cls._METADATA_NAME])

        output_features_ = _tuple_to_multiindex(output_features_dict_)

        obj = cls(
            sample_dim_name,
            input_variables,
            output_variables,
            model,
            target_scaler=scaler_obj,
            select_levels=mapper.get(cls._SELECT_LEVELS, b"all").decode(),
        )
        obj.output_features_ = output_features_

        return obj

    # these are here for backward compatibility with pre-unified API attribute names
    @property
    def input_variables(self):
        if hasattr(self, "_input_variables"):
            return self._input_variables
        elif hasattr(self, "input_vars_"):
            return self.input_vars_
        else:
            raise ValueError("Wrapped model version without input variables attribute.")

    @property
    def output_variables(self):
        if hasattr(self, "_input_variables"):
            return self._output_variables
        elif hasattr(self, "input_vars_"):
            return self.output_vars_
        else:
            raise ValueError(
                "Wrapped model version without output variables attribute."
            )

    @property
    def sample_dim_name(self):
        return getattr(self, "_sample_dim_name", "sample")
