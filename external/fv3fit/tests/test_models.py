import xarray as xr
import numpy as np

import pytest

from fv3fit.keras._models._sequences import _ThreadedSequencePreLoader
from fv3fit.keras._models.models import PackedKerasModel, _fill_defaults
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
def test_PackedKerasModel_jacobian(base_state):
    class IdentityModel(PackedKerasModel):
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
    model = IdentityModel("x", ["a"], ["b"])
    model.fit([batch])
    if base_state == "manual":
        jacobian = model.jacobian(batch[["a"]].isel(x=0))
    elif base_state == "default":
        jacobian = model.jacobian()

    assert jacobian[("a", "b")].dims == ("b", "a")
    np.testing.assert_allclose(np.asarray(jacobian[("a", "b")]), np.eye(5))


@pytest.mark.parametrize(
    "kwargs, inputs, expected",
    [
        ({}, {}, {"kwarg0": 0, "kwarg1": 1}),
        ({}, {"kwarg0": 1, "kwarg1": 2}, {"kwarg0": 1, "kwarg1": 2}),
        ({"kwarg0": 1}, {"kwarg0": 1, "kwarg1": 2}, {"kwarg0": 1, "kwarg1": 2}),
        ({"kwarg0": 1}, {"kwarg0": 0, "kwarg1": 2}, None),
    ],
)
def test_fill_defaults(kwargs, inputs, expected):
    defaults = {"kwarg0": 0, "kwarg1": 1}
    if expected is None:
        with pytest.raises(ValueError):
            _fill_defaults(kwargs, inputs, defaults)
    else:
        assert _fill_defaults(kwargs, inputs, defaults) == expected
