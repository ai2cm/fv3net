import xarray as xr
import numpy as np

import pytest

from fv3fit.keras._models._sequences import _ThreadedSequencePreLoader
from fv3fit.keras._models.models import PackedKerasModel
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
    "args, kwargs, expected",
    [
        (
            [],
            {},
            {
                "batch_size": None,
                "epochs": 1,
                "workers": 1,
                "max_queue_size": 8,
                "validation_samples": 13824,
                "use_last_batch_to_validate": False,
            },
        ),
        (
            [1, None, 3],
            {"validation_samples": 1000},
            {
                "epochs": 1,
                "batch_size": None,
                "workers": 3,
                "max_queue_size": 8,
                "validation_samples": 1000,
                "use_last_batch_to_validate": False,
            },
        ),
        ([1, None, 3], {"epochs": 2}, None),
    ],
)
def test_fill_fit_kwargs_default(args, kwargs, expected):
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
    model = IdentityModel("x", ["a"], ["b"], fit_kwargs=kwargs)
    if expected:
        model.fit([batch], None, *args)
        assert model._fit_kwargs == expected
    else:
        with pytest.raises(ValueError):
            model.fit([batch], None, *args)
