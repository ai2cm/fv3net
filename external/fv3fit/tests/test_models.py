from fv3fit._shared.config import DenseHyperparameters
import xarray as xr
import numpy as np

import pytest

from fv3fit.keras._models._sequences import _ThreadedSequencePreLoader
from fv3fit.keras._models.models import DenseModel, _fill_default
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
            "a": (["x", "z"], np.arange(10).reshape(2, 5)),
            "b": (["x", "z"], np.arange(10).reshape(2, 5)),
        }
    )
    model = IdentityModel("x", ["a"], ["b"], DenseHyperparameters())
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
    hyperparameters = DenseHyperparameters(nonnegative_outputs=True)
    model = DenseModel("sample", ["input"], ["output"], hyperparameters,)
    batch = xr.Dataset(
        {
            "input": (["sample"], np.arange(100)),
            # even with negative targets, trained model should be nonnegative
            "output": (["sample"], np.full((100,), -1e4)),
        }
    )
    model.fit([batch])
    prediction = model.predict(batch)
    assert prediction.min() >= 0.0
