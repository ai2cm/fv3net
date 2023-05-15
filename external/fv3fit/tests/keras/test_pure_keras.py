import pathlib
import fv3fit
from fv3fit.keras._models.shared.pure_keras import (
    PureKerasDictPredictor,
    PureKerasModel,
)
import tensorflow as tf
import pytest
import xarray
import numpy as np


def test_PureKerasDictPredictor_dump_load(tmp_path: pathlib.Path):
    in_ = tf.keras.Input(shape=[1], name="a")
    out = tf.keras.layers.Lambda(lambda x: x, name="out")(in_)
    dict_model = tf.keras.Model(inputs=in_, outputs=out)
    predictor = PureKerasDictPredictor(dict_model)
    fv3fit.dump(predictor, tmp_path.as_posix())
    loaded = fv3fit.load(tmp_path.as_posix())
    assert predictor.input_variables == loaded.input_variables
    assert predictor.output_variables == loaded.output_variables


def test_PureKerasDictPredictor_predict():
    # random number
    r = 0.3432134898
    in_ = tf.keras.Input(shape=[1], name="a")
    out = tf.keras.layers.Lambda(lambda x: x * r, name="out")(in_)
    dict_model = tf.keras.Model(inputs=in_, outputs=out)
    predictor = PureKerasDictPredictor(dict_model)
    in_xr = xarray.Dataset({"a": (["blah", "z"], [[1]])})
    out = predictor.predict(in_xr)
    assert float(out["out"][0]) == pytest.approx(r)


@pytest.mark.parametrize(
    ["unstacked_dims"],
    [
        pytest.param((), id="no_unstacked_dims"),
        pytest.param(("z",), id="column_unstacked_dim"),
    ],
)
def test_PureKerasModel_predict(unstacked_dims):
    input_variable = xarray.DataArray(np.arange(100).reshape(5, 20), dims=["x", "z"],)
    input_shape = _get_input_shape(input_variable, unstacked_dims)
    model = _get_dummy_model(input_shape)
    predictor = PureKerasModel(
        input_variables=["a"],
        output_variables=["b"],
        model=model,
        unstacked_dims=unstacked_dims,
    )
    prediction = predictor.predict(xarray.Dataset({"a": input_variable}))
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
