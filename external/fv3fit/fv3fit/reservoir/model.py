import abc
import dacite
import dataclasses
import fsspec
import io
import joblib
import numpy as np
from typing import Sequence
import yaml

from .readout import ReservoirComputingReadout
from .reservoir import Reservoir
from .config import ReservoirHyperparameters, SubdomainConfig
from .domain import PeriodicDomain, Subdomain


class ImperfectModel(abc.ABC):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Predict one reservoir computing step ahead.
        If the imperfect model takes shorter timesteps than the reservoir model,
        this should return the imperfect model's prediction at the next reservoir step.
        """
        pass


class ReservoirComputingModel:
    _READOUT_NAME = "readout.pkl"
    _METADATA_NAME = "metadata.bin"

    def __init__(
        self, reservoir: Reservoir, readout: ReservoirComputingReadout,
    ):
        self.reservoir = reservoir
        self.readout = readout

    def predict(self):
        prediction = self.readout.predict(self.reservoir.state).reshape(-1)
        self.reservoir.increment_state(prediction)
        return prediction

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)

        mapper[self._READOUT_NAME] = self.readout.dumps()
        metadata = {
            "reservoir_hyperparameters": dataclasses.asdict(
                self.reservoir.hyperparameters
            )
        }
        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingModel":
        """Load a model from a remote path"""
        mapper = fsspec.get_mapper(path)

        f = io.BytesIO(mapper[cls._READOUT_NAME])
        readout_components = joblib.load(f)
        readout = ReservoirComputingReadout(**readout_components)
        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])

        reservoir_hyperparameters = dacite.from_dict(
            ReservoirHyperparameters, metadata["reservoir_hyperparameters"]
        )

        return cls(reservoir=Reservoir(reservoir_hyperparameters), readout=readout,)


class HybridReservoirComputingModel:
    _READOUT_NAME = "readout.pkl"
    _METADATA_NAME = "metadata.bin"

    def __init__(
        self, reservoir: Reservoir, readout: ReservoirComputingReadout,
    ):
        self.reservoir = reservoir
        self.readout = readout

    def predict(self, input_state, imperfect_model):
        imperfect_prediction = imperfect_model.predict(input_state)
        readout_input = np.hstack([self.reservoir.state, imperfect_prediction])
        rc_prediction = self.readout.predict(readout_input).reshape(-1)
        self.reservoir.increment_state(rc_prediction)
        return rc_prediction

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)

        mapper[self._READOUT_NAME] = self.readout.dumps()
        metadata = {
            "reservoir_hyperparameters": dataclasses.asdict(
                self.reservoir.hyperparameters
            )
        }
        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")

    @classmethod
    def load(cls, path):
        mapper = fsspec.get_mapper(path)

        f_readout = io.BytesIO(mapper[cls._READOUT_NAME])
        readout_components = joblib.load(f_readout)
        readout = ReservoirComputingReadout(**readout_components)
        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])

        reservoir_hyperparameters = dacite.from_dict(
            ReservoirHyperparameters, metadata["reservoir_hyperparameters"]
        )

        return cls(reservoir=Reservoir(reservoir_hyperparameters), readout=readout,)


class DomainPredictor:
    def __init__(
        self,
        subdomain_predictors: Sequence[ReservoirComputingModel],
        subdomain_config: SubdomainConfig,
        n_jobs: int = -1,
    ):
        self.subdomain_predictors = subdomain_predictors
        self.subdomain_size = subdomain_config.size
        self.subdomain_overlap = subdomain_config.overlap
        self.n_jobs = n_jobs

    def _update_subdomain_reservoir_state(
        self,
        subdomain_predictor: ReservoirComputingModel,
        subdomain_prediction: Subdomain,
    ):
        subdomain_predictor.reservoir.increment_state(subdomain_prediction.overlapping)

    def _synchronize_subdomain(self, subdomain_predictor, subdomain_data):
        subdomain_reservoir = subdomain_predictor.reservoir
        subdomain_reservoir.synchronize(subdomain_data.overlapping)

    def synchronize(self, data):
        data_domain = PeriodicDomain(
            data=data,
            subdomain_size=self.subdomain_size,
            subdomain_overlap=self.subdomain_overlap,
            subdomain_axis=1,
        )
        joblib.Parallel(n_jobs=self.n_jobs, verbose=1, backend="threading")(
            joblib.delayed(self._synchronize_subdomain)(
                subdomain_predictor, subdomain_data
            )
            for subdomain_predictor, subdomain_data in zip(
                self.subdomain_predictors, data_domain
            )
        )

    def dump(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError


class ReservoirOnlyDomainPredictor(DomainPredictor):
    def _predict_on_subdomain(self, subdomain_predictor):
        return subdomain_predictor.readout.predict(
            subdomain_predictor.reservoir.state
        ).reshape(-1)

    def predict(self):
        subdomain_predictions = joblib.Parallel(n_jobs=self.n_jobs, verbose=0)(
            joblib.delayed(self._predict_on_subdomain)(subdomain_predictor)
            for subdomain_predictor in self.subdomain_predictors
        )
        prediction = np.concatenate(subdomain_predictions)
        subdomain_predictions_domain = PeriodicDomain(
            data=prediction,
            subdomain_size=self.subdomain_size,
            subdomain_overlap=self.subdomain_overlap,
            subdomain_axis=0,
        )

        # increment reservoir states after the subdomain predictions
        #  are combined so thatthe input includes overlaps between subdomains
        joblib.Parallel(n_jobs=self.n_jobs, verbose=0, backend="threading")(
            joblib.delayed(self._update_subdomain_reservoir_state)(
                subdomain_predictor, subdomain_prediction
            )
            for subdomain_predictor, subdomain_prediction in zip(
                self.subdomain_predictors, subdomain_predictions_domain
            )
        )
        return prediction


class HybridDomainPredictor(DomainPredictor):
    def _predict_on_subdomain(
        self, subdomain_predictor, imperfect_prediction_subdomain
    ):
        readout_input = np.hstack(
            [
                subdomain_predictor.reservoir.state,
                imperfect_prediction_subdomain.overlapping,
            ]
        )
        return subdomain_predictor.readout.predict(readout_input).reshape(-1)

    def predict(self, input_state, imperfect_model):
        imperfect_prediction = imperfect_model.predict(input_state)
        imperfect_prediction_domain = PeriodicDomain(
            data=imperfect_prediction,
            subdomain_size=self.subdomain_size,
            subdomain_overlap=self.subdomain_overlap,
            subdomain_axis=0,
        )
        subdomain_predictions = joblib.Parallel(n_jobs=self.n_jobs, verbose=0)(
            joblib.delayed(self._predict_on_subdomain)(
                subdomain_predictor, imperfect_prediction_subdomain
            )
            for subdomain_predictor, imperfect_prediction_subdomain in zip(
                self.subdomain_predictors, imperfect_prediction_domain
            )
        )
        prediction = np.concatenate(subdomain_predictions)
        subdomain_predictions_domain = PeriodicDomain(
            data=prediction,
            subdomain_size=self.subdomain_size,
            subdomain_overlap=self.subdomain_overlap,
            subdomain_axis=0,
        )

        # increment reservoir states after the subdomain predictions
        # are combined so that the input includes overlaps between subdomains
        joblib.Parallel(n_jobs=self.n_jobs, verbose=0, backend="threading")(
            joblib.delayed(self._update_subdomain_reservoir_state)(
                subdomain_predictor, subdomain_prediction
            )
            for subdomain_predictor, subdomain_prediction in zip(
                self.subdomain_predictors, subdomain_predictions_domain
            )
        )
        return prediction
