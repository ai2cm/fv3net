from fv3fit import DenseHyperparameters
import xarray as xr
import numpy as np

import pytest

from fv3fit.keras._models._sequences import _ThreadedSequencePreLoader
from fv3fit.keras._models.models import DenseModel, _fill_default
from fv3fit._shared import PackerConfig, SliceConfig
import tensorflow.keras


def test__ThreadedSequencePreLoader():
    """ Check correctness of the pre-loaded sequence"""
    sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    loader = _ThreadedSequencePreLoader(sequence, num_workers=4)
    result = [item for item in loader]
    assert len(result) == len(sequence)
    for item in result:
        assert item in sequence


@pytest.mark.parametrize("base_state", ["manual", "default"])
def test_DenseModel_jacobian(base_state):
    class IdentityModel(DenseModel):
        def get_model(self, n, m):
            x = tensorflow.keras.Input(shape=[n])
            model = tensorflow.keras.Model(inputs=[x], outputs=[x])
            model.compile()
            return model

    batch = xr.Dataset(
        {
            "a": (["x", "z"], np.arange(10, dtype=np.float).reshape(2, 5)),
            "b": (["x", "z"], np.arange(10, dtype=np.float).reshape(2, 5)),
        }
    )
    model = IdentityModel(["a"], ["b"], DenseHyperparameters(["a"], ["b"]))
    model.fit([batch])
    if base_state == "manual":
        jacobian = model.jacobian(batch[["a"]].isel(x=0))
    elif base_state == "default":
        jacobian = model.jacobian()

    assert jacobian[("a", "b")].dims == ("b", "a")
    np.testing.assert_allclose(np.asarray(jacobian[("a", "b")]), np.eye(5))


@pytest.mark.parametrize(
    "kwargs, arg, key, default, expected",
    [
        ({}, None, "kwarg0", 0, {"kwarg0": 0}),
        ({"kwarg0": 0}, 0, "kwarg0", 0, {"kwarg0": 0}),
        ({"kwarg0": 1}, 0, "kwarg0", 0, None),
    ],
)
def test_fill_default(kwargs, arg, key, default, expected):
    if expected is None:
        with pytest.raises(ValueError):
            _fill_default(kwargs, arg, key, default)
    else:
        assert _fill_default(kwargs, arg, key, default) == expected


def test_nonnegative_model_outputs():
    hyperparameters = DenseHyperparameters(
        ["input"], ["output"], nonnegative_outputs=True
    )
    model = DenseModel(["input"], ["output"], hyperparameters,)
    batch = xr.Dataset(
        {
            "input": (["x"], np.arange(100)),
            # even with negative targets, trained model should be nonnegative
            "output": (["x"], np.full((100,), -1e4)),
        }
    )
    model.fit([batch])
    prediction = model.predict(batch)
    assert prediction.min() >= 0.0


def test_DenseModel_clipped_inputs():
    hyperparameters = DenseHyperparameters(
        ["a", "b"],
        ["c"],
        packer_config=PackerConfig({"a": {"z": SliceConfig(None, 3)}}),
    )
    model = DenseModel(["a", "b"], ["c"], hyperparameters)

    nz = 5
    dims = ["x", "y", "z"]
    shape = (2, 2, nz)
    arr = np.arange(np.prod(shape)).reshape(shape).astype(float)
    input_data = xr.Dataset({"a": (dims, arr), "b": (dims, arr), "c": (dims, arr + 1)})

    slice_filled_input = xr.Dataset(
        {"a": input_data["a"].where(input_data.z < 3).fillna(1.0), "b": input_data["b"]}
    )

    model.fit([input_data])
    prediction_clipped = model.predict(input_data)
    assert model.X_packer._n_features["a"] == 3
    assert model.X_packer._n_features["b"] == 5

    prediction_nan_filled = model.predict(slice_filled_input)

    xr.testing.assert_allclose(prediction_nan_filled, prediction_clipped, rtol=1e-3)


def test_DenseModel_raises_not_implemented_error_with_clipped_output_data():
    hyperparameters = DenseHyperparameters(
        ["a", "b"],
        ["c"],
        packer_config=PackerConfig({"c": {"z": SliceConfig(None, 3)}}),
    )

    with pytest.raises(NotImplementedError):
        DenseModel(
            ["a", "b"], ["c"], hyperparameters,
        )
