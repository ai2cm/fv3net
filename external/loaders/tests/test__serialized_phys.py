import pytest
import xarray as xr
import numpy as np

from loaders.batches._serialized_phys import SerializedSequence


@pytest.fixture
def xr_data():

    data = np.random.randn(10, 15, 20, 25)
    return xr.DataArray(data=data, dims=["savepoint", "rank", "horiz", "vert"])


@pytest.mark.parametrize("item_dim", ["savepoint", "rank", "horiz", "vert"])
def test_SerializedSequence_int_item(xr_data, item_dim):

    seq = SerializedSequence(xr_data, item_dim=item_dim)
    dat = seq[0]
    xr.testing.assert_equal(xr_data.isel({item_dim: 0}), dat)


def test_SerializedSequence_len(xr_data):

    seq = SerializedSequence(xr_data, item_dim="savepoint")
    assert len(seq) == 10
