from fv3fit._shared import (
    Predictor,
    io,
    match_prediction_to_input_coords,
    SAMPLE_DIM_NAME,
    stack,
)
from fv3fit.keras.adapters import (
    ensure_dict_output,
    rename_dict_input,
    rename_dict_output,
    ensure_tuple_model,
)
import tensorflow as tf
from typing import Any, Dict, Hashable, Iterable, Sequence, Mapping
import xarray as xr
import os
from ...._shared import get_dir, put_dir, InputSensitivity
import yaml
import numpy as np
from .halos import append_halos, append_halos_using_mpi
from fv3fit.keras.jacobian import compute_jacobians, nondimensionalize_jacobians


@io.register("all-keras")
class PureKerasModel(Predictor):
    """Model which uses Keras for packing and normalization.

    Assumes wrapped keras model accepts [sample, feature] arrays.
    """

    _MODEL_FILENAME = "model.tf"
    _CONFIG_FILENAME = "config.yaml"
    custom_objects: Dict[str, Any] = {}

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        model: tf.keras.Model,
        unstacked_dims: Sequence[str],
        n_halo: int = 0,
    ):
        """Initialize the predictor

        Args:
            input_variables: names of input variables
            output_variables: names of output variables
            model: keras model to wrap
            unstacked_dims: non-sample dimensions of model output
            n_halo: number of halo points required in input data
        """
        super().__init__(input_variables, output_variables)
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.model = model
        self._n_halo = n_halo
        self._unstacked_dims = unstacked_dims

    @classmethod
    def load(cls, path: str) -> "PureKerasModel":
        """Load a serialized model from a directory."""
        with get_dir(path) as path:
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            model = tf.keras.models.load_model(
                model_filename, custom_objects=cls.custom_objects
            )
            with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            obj = cls(
                config["input_variables"],
                config["output_variables"],
                model,
                unstacked_dims=config.get("unstacked_dims", None),
                n_halo=config.get("n_halo", 0),
            )
            return obj

    def _array_prediction_to_dataset(
        self, names, outputs, stacked_coords
    ) -> xr.Dataset:
        ds = xr.Dataset()
        for name, output in zip(names, outputs):
            dims = [SAMPLE_DIM_NAME] + list(self._unstacked_dims)
            scalar_singleton_dim = (
                len(output.shape) == len(dims) and output.shape[-1] == 1
            )
            if scalar_singleton_dim:  # remove singleton dimension
                output = output[..., 0]
                dims = dims[:-1]
            da = xr.DataArray(
                data=output, dims=dims, coords={SAMPLE_DIM_NAME: stacked_coords},
            ).unstack(SAMPLE_DIM_NAME)
            dim_order = [dim for dim in self._unstacked_dims if dim in da.dims]
            ds[name] = da.transpose(*dim_order, ...)
        return ds

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        if self._n_halo > 0:
            if "tile" not in X.dims:
                try:
                    X = append_halos_using_mpi(ds=X, n_halo=self._n_halo)
                except RuntimeError as err:
                    raise ValueError(
                        "either dataset must have tile dimension or MPI must be present"
                    ) from err
            else:
                X = append_halos(ds=X, n_halo=self._n_halo)
        X_stacked = stack(X, unstacked_dims=self._unstacked_dims)
        inputs = [X_stacked[name].values for name in self.input_variables]
        outputs = self.model.predict(inputs)
        if isinstance(outputs, np.ndarray):
            outputs = [outputs]
        return_ds = self._array_prediction_to_dataset(
            self.output_variables, outputs, X_stacked.coords[SAMPLE_DIM_NAME],
        )
        return match_prediction_to_input_coords(X, return_ds)

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self.model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            with open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
                f.write(
                    yaml.dump(
                        {
                            "input_variables": self.input_variables,
                            "output_variables": self.output_variables,
                            "unstacked_dims": self._unstacked_dims,
                            "n_halo": self._n_halo,
                        }
                    )
                )

    def get_dict_compatible_model(self) -> tf.keras.Model:
        """
        Rename keras model inputs/outputs to match input_variables and
        output_variables, and ensure model accepts and outputs dicts.
        """
        renamed_inputs: Mapping[str, str] = {
            str(layer_name): str(input_var)
            for layer_name, input_var in zip(
                self.model.input_names, self.input_variables
            )
        }
        renamed_outputs: Mapping[str, str] = {
            str(layer_name): str(output_var)
            for layer_name, output_var in zip(
                self.model.output_names, self.output_variables
            )
        }

        dict_compatible_model = ensure_dict_output(self.model)

        model_renamed_inputs = rename_dict_input(dict_compatible_model, renamed_inputs)
        return rename_dict_output(model_renamed_inputs, renamed_outputs)

    def input_sensitivity(self, stacked_sample: xr.Dataset) -> InputSensitivity:
        # Jacobian functions take in dict input
        data_dict = {}
        for var in stacked_sample:
            values = stacked_sample[var].values
            if len(stacked_sample[var].dims) == 1:
                values = values.reshape(-1, 1)
            data_dict[var] = values

        jacobians = compute_jacobians(
            self.get_dict_compatible_model(),  # type: ignore
            data_dict,
            self.input_variables,
        )
        # normalize factors so sensitivities are comparable but still
        # preserve level-relative magnitudes
        std_factors = {name: np.std(data, axis=0) for name, data in data_dict.items()}
        jacobians_std = nondimensionalize_jacobians(jacobians, std_factors)
        return InputSensitivity(jacobians=jacobians_std)


@io.register("all-keras-dict")
class PureKerasDictPredictor(Predictor):
    """Save a keras model encoding the input/output names into a saved_model
    artifact

    The output names of the keras model are assumed to be correct. To do this
    requires including the following logic when building the model::

        out_with_correct_name = tf.keras.layers.Lambda(
            lambda x: x, name="desired_name")

    Note:
        This class relies on PureKerasModel to implement Mhe Predictor methods.
        It might be worth refactoring the implementation here instead since the
        dict-in/out format has more structure.

    """

    _MODEL_FILENAME = "model.tf"
    _CONFIG_FILENAME = "config.yaml"

    def __init__(self, model: tf.keras.Model, passthrough: Any = None):
        """Initialize the predictor.

        Args:
            model: a keras model. Tuple in/out models are okay, but will be cast
                to a dict-output model before saving.
            passthrough: any data! Is NOT serialized, but can be
                used by other parts of the training workflow. Mostly exists for
                backwards compatibility and should be avoided in new code.
        """
        dict_model = ensure_dict_output(model)
        tuple_model = ensure_tuple_model(dict_model)
        super().__init__(tuple_model.input_names, tuple_model.output_names)
        self._model = tuple_model
        self._pure_keras_predictor = PureKerasModel(
            self.input_variables,
            self.output_variables,
            tuple_model,
            # unstacked dims is hardcoded most places in fv3fit.keras, but it
            # seems like useful metadata, but more of an aspect of the training
            # data pipeline since it is the piece of code that removes the
            # dimension names...would be nice to remove this concept...e.g. pass
            # unstacked_dims to the .dump rather than in the constructor
            # TODO consider this refactor
            unstacked_dims=["z"],
        )
        self.passthrough = passthrough

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        return self._pure_keras_predictor.predict(X)

    def dump(self, path: str) -> None:
        """Serialize to a directory."""
        with put_dir(path) as local_path:
            self._model.save(self._model_filename(local_path))
            path

    @classmethod
    def _model_filename(cls, path: str) -> str:
        return os.path.join(path, cls._MODEL_FILENAME)

    @classmethod
    def load(cls, path: str) -> "Predictor":
        """Load a serialized model from a directory."""
        model = tf.keras.models.load_model(cls._model_filename(path))
        return cls(model)

    def input_sensitivity(self, stacked_sample: xr.Dataset) -> InputSensitivity:
        """Calculate sensitivity to input features."""
        raise NotImplementedError(
            "input_sensitivity is not implemented...but could be pretty "
            "easily using jacobian routines."
        )
