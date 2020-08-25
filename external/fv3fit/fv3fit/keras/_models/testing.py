from typing import Iterable, Sequence, Optional
import logging
import os
import xarray as xr
import numpy as np
from .models import Model, _XyArraySequence
from ._filesystem import get_dir, put_dir
from ..._shared import ArrayPacker

logger = logging.getLogger(__file__)


class DummyModel(Model):
    """
    A dummy keras model for testing, whose `fit` method learns only the input and
    output variable array dimensions in an xarray dataset and ignores their contents,
    and which simply returns zeros for all output variable features
    """

    # these should only be used in the dump/load routines for this class
    _X_PACKER_FILENAME = "X_packer.json"
    _Y_PACKER_FILENAME = "y_packer.json"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
    ):
        """Initialize the DummyModel
        Args:
            sample_dim_name: name of the sample dimension in datasets used as
                inputs and outputs.
            input_variables: names of input variables
            output_variables: names of output variables
        """
        super().__init__(sample_dim_name, input_variables, output_variables)
        self.X_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=input_variables
        )
        self.y_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=output_variables
        )

    def fit(
        self,
        batches: Sequence[xr.Dataset],
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> None:
        # this is all we need to do to learn n output feature
        _, _ = _XyArraySequence(self.X_packer, self.y_packer, batches)[0]

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        if not self.y_packer._n_features:
            raise RuntimeError("must call fit() for dummy model to be available")
        feature_index = X[self.sample_dim_name]
        ds_pred = self.y_packer.to_dataset(
            self.predict_array(self.X_packer.to_array(X))
        )
        return ds_pred.assign_coords({self.sample_dim_name: feature_index})

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], self.y_packer._total_features))

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            with open(os.path.join(path, self._X_PACKER_FILENAME), "w") as f:
                self.X_packer.dump(f)
            with open(os.path.join(path, self._Y_PACKER_FILENAME), "w") as f:
                self.y_packer.dump(f)

    @classmethod
    def load(cls, path: str) -> Model:
        with get_dir(path) as path:
            with open(os.path.join(path, cls._X_PACKER_FILENAME), "r") as f:
                X_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._Y_PACKER_FILENAME), "r") as f:
                y_packer = ArrayPacker.load(f)
            obj = cls(
                X_packer.sample_dim_name, X_packer.pack_names, y_packer.pack_names
            )
            obj.X_packer = X_packer
            obj.y_packer = y_packer
            return obj
