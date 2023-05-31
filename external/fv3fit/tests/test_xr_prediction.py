from fv3fit._shared.xr_prediction import predict_on_dataset, DatasetPredictor
from fv3fit._shared.io import register
import contextlib
import numpy as np
import os
import pytest
import tensorflow as tf
import xarray


class ConstantArrayPredictor:
    """
    A simple predictor meant to be used for testing.

    Supports scalar and vector outputs, where the vector outputs are all
    of the same shape and assigned a dimension name of "z".
    """

    _MODEL_NAME = "model.tf"

    def __init__(self, model: tf.keras.Model):
        """Initialize the predictor

        Args:
            model: keras model returning identity

        """
        self.model = model

    def predict(self, X):
        return [
            self.model(X),
        ]

    def dump(self, path: str) -> None:
        self.model.save(os.path.join(path, self._MODEL_NAME))

    @classmethod
    def load(cls, path: str) -> "ConstantArrayPredictor":
        model = tf.keras.models.load_model(os.path.join(path, cls._MODEL_NAME))
        return cls(model)


@contextlib.contextmanager
def registration_context():
    try:
        yield
    finally:
        register._register_class(ConstantArrayPredictor, "constant-array")


@pytest.mark.parametrize(
    ["unstacked_dims"],
    [
        pytest.param((), id="no_unstacked_dims"),
        pytest.param(("z",), id="column_unstacked_dim"),
    ],
)
def test_predict_on_dataset(unstacked_dims):
    input_variable = xarray.DataArray(np.arange(100).reshape(5, 20), dims=["x", "z"],)
    input_shape = _get_input_shape(input_variable, unstacked_dims)

    prediction = predict_on_dataset(
        model=_get_dummy_model(input_shape),
        X=xarray.Dataset({"a": input_variable}),
        input_variables=["a"],
        output_variables=["b"],
        unstacked_dims=unstacked_dims,
        n_halo=0,
    )
    assert set(prediction["b"].dims) == set(["x", "z"])


def _get_input_shape(da, unstacked_dims):
    unstacked_shape = [da.sizes[dim] for dim in unstacked_dims]
    if len(unstacked_shape) == 0:
        unstacked_shape = [1]
    return tuple(unstacked_shape)


def _get_dummy_model(input_shape):
    in_ = tf.keras.Input(shape=input_shape)
    out_ = tf.keras.layers.Lambda(lambda x: x)(in_)
    model = tf.keras.Model(inputs=in_, outputs=out_)
    return model


def test_DatasetPredictor_predict():
    with registration_context():
        input_vars = ["in0"]
        output_vars = ["out0"]
        input_shape = 5
        n_samples = 3
        base_model = ConstantArrayPredictor(model=_get_dummy_model(input_shape))
        predictor = DatasetPredictor(
            input_variables=input_vars,
            output_variables=output_vars,
            model=base_model,
            unstacked_dims=["z"],
            n_halo=0,
        )
        in_xr = xarray.Dataset(
            {"in0": (["_fv3net_sample", "z"], np.random.rand(n_samples, input_shape))}
        )
        out = predictor.predict(in_xr)
        np.testing.assert_allclose(in_xr["in0"], out["out0"])


def test_DatasetPredictor_dump_load(tmpdir):
    with registration_context():

        input_vars = ["in0"]
        output_vars = ["out0"]
        input_shape = 5
        n_samples = 3
        base_model = ConstantArrayPredictor(model=_get_dummy_model(input_shape))
        predictor = DatasetPredictor(
            input_variables=input_vars,
            output_variables=output_vars,
            model=base_model,
            unstacked_dims=["z"],
            n_halo=0,
        )
        save_path = os.path.join(str(tmpdir), "dataset_predictor")
        predictor.dump(save_path)
        loaded_predictor = DatasetPredictor.load(save_path)

        in_xr = xarray.Dataset(
            {"in0": (["_fv3net_sample", "z"], np.random.rand(n_samples, input_shape))}
        )
        np.testing.assert_allclose(
            loaded_predictor.predict(in_xr)["out0"], in_xr["in0"]
        )
