import xarray as xr
import numpy as np

import pytest

from fv3fit.keras._models._sequences import _ThreadedSequencePreLoader
from fv3fit.keras._models.models import PackedKerasModel


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
    class IdentityMock:
        def __call__(self, x):
            return x

        def fit(self, *args, **kwargs):
            pass

    class IdentityModel(PackedKerasModel):
        def get_model(self, n, m):
            return IdentityMock()

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
