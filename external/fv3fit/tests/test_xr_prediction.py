from fv3fit._shared.xr_prediction import predict_on_dataset
import tensorflow as tf
import pytest
import xarray
import numpy as np


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
