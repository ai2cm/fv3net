from fv3fit._shared import pack, count_features, unpack
import xarray as xr
from typing import Sequence, Mapping
import numpy as np
import pytest
from fv3fit._shared.packer import PackingInfo


def get_dataset(
    base_dims: Sequence[str],
    base_shape: Sequence[int],
    feature_dim: str,
    n_features: Mapping[str, int],
):
    data_vars = {}
    for name, n in n_features.items():
        shape = list(base_shape)
        dims = list(base_dims)
        if n > 1:
            shape += [n]
            dims += [name + "_" + feature_dim]
        data_vars[name] = xr.DataArray(np.random.randn(*shape), dims=dims)
    return xr.Dataset(data_vars=data_vars)


@pytest.mark.parametrize(
    "base_dims, base_shape, feature_dim, n_features",
    (
        pytest.param(["sample"], [5], "z", {"var_1": 9}, id="single_variable_2d"),
        pytest.param(["sample"], [5], "z", {"var_1": 1}, id="single_variable_1d"),
        pytest.param(["sample"], [5], "z", {"var_1": 1, "var_2": 9}, id="2d_and_1d"),
        pytest.param(["x", "y"], [5, 5], "z", {"var_1": 1, "var_2": 7}, id="3d_and_2d"),
        pytest.param(
            ["sample", "x", "y"],
            [10, 5, 5],
            "z",
            {"var_1": 1, "var_2": 7},
            id="4d_and_3d",
        ),
    ),
)
def test_roundtrip(
    base_dims: Sequence[str],
    base_shape: Sequence[int],
    feature_dim: str,
    n_features: Mapping[str, int],
):
    ds = get_dataset(
        base_dims=base_dims,
        base_shape=base_shape,
        feature_dim=feature_dim,
        n_features=n_features,
    )
    packed, feature_index = pack(data=ds, sample_dims=base_dims)
    unpacked = unpack(data=packed, sample_dims=base_dims, feature_index=feature_index)
    for name in ds.data_vars:
        print(name)
        np.testing.assert_array_equal(unpacked[name].values, ds[name].values)


def test_pack_2d_and_3d():
    ds = xr.Dataset(
        data_vars={
            "surface": xr.DataArray(np.zeros([100]), dims=["sample"]),
            "vertically_resolved": xr.DataArray(
                np.zeros([100, 10]), dims=["sample", "z"]
            ),
        }
    )
    packed, _ = pack(data=ds, sample_dims=["sample"])
    assert packed.shape == (100, 11)


@pytest.mark.parametrize(
    "base_dims, base_shape, feature_dim, n_features",
    (
        pytest.param(["sample"], [5], "z", {"var_1": 9}, id="single_variable_2d"),
        pytest.param(["sample"], [5], "z", {"var_1": 1}, id="single_variable_1d"),
        pytest.param(["sample"], [5], "z", {"var_1": 1, "var_2": 9}, id="2d_and_1d"),
        pytest.param(["x", "y"], [5, 5], "z", {"var_1": 1, "var_2": 7}, id="3d_and_2d"),
        pytest.param(
            ["sample", "x", "y"],
            [10, 5, 5],
            "z",
            {"var_1": 1, "var_2": 7},
            id="4d_and_3d",
        ),
    ),
)
def test_count_features(
    base_dims: Sequence[str],
    base_shape: Sequence[int],
    feature_dim: str,
    n_features: Mapping[str, int],
):
    ds = get_dataset(
        base_dims=base_dims,
        base_shape=base_shape,
        feature_dim=feature_dim,
        n_features=n_features,
    )
    _, feature_index = pack(data=ds, sample_dims=base_dims)
    counts = count_features(feature_index)
    assert counts == n_features


def test_packing_info_multi_index():
    packing_info = PackingInfo(names=["Q1", "Q2"], features=[3, 3])
    multi_index = packing_info.multi_index
    assert multi_index.to_list() == [
        ("Q1", 0),
        ("Q1", 1),
        ("Q1", 2),
        ("Q2", 0),
        ("Q2", 1),
        ("Q2", 2),
    ]
