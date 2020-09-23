import pytest
import string
import xarray as xr
import numpy as np

import loaders.batches._serialized_phys as sp

SAMPLE_DIM_NAME = "sample"


@pytest.fixture
def xr_data():
    data = np.random.randn(10, 15, 20, 8)
    dims = ["savepoint", "horiz", "feature", "tracer_num"]
    reg = xr.DataArray(data=data[:, :, :, 0], dims=dims[:3])
    reg = reg.to_dataset(name="field")
    reg["field_with_tracer"] = xr.DataArray(data=data, dims=dims)

    return reg


@pytest.mark.parametrize("idx", [0, -1])
def test_Serilized_int_item(xr_data, idx):
    seq = sp.SerializedSequence(xr_data=xr_data, item_dim="savepoint")
    dat = seq[idx]
    xr.testing.assert_equal(xr_data.isel(savepoint=idx), dat)


@pytest.mark.parametrize("item_dim,", ["savepoint", "horiz"])
def test_SerializedSequence_item_dim(xr_data, item_dim):

    seq = sp.SerializedSequence(xr_data, item_dim=item_dim)
    dat = seq[0]
    xr.testing.assert_equal(xr_data.isel({item_dim: 0}), dat)


@pytest.mark.parametrize("selection_slice", [slice(3), slice(2, 4), slice(0, 6, 2)])
def test_SerializedSequence_slice_item(xr_data, selection_slice):

    seq = sp.SerializedSequence(xr_data, item_dim="savepoint")
    dat = seq[selection_slice]
    xr.testing.assert_equal(xr_data.isel({"savepoint": selection_slice}), dat)


def test_SerializedSequence_len(xr_data):

    seq = sp.SerializedSequence(xr_data, item_dim="savepoint")
    assert len(seq) == 10


def test__find_tracer_dim():

    with pytest.raises(ValueError):
        sp._find_tracer_dim(["tracer1", "tracer2"])

    tracer_dim = sp._find_tracer_dim(["horiz", "tracer_num", "vert"])
    assert tracer_dim == "tracer_num"


@pytest.fixture()
def tracer_dataset(xr_data):
    return xr_data.stack({SAMPLE_DIM_NAME: ["savepoint", "horiz"]}).transpose()


def test__separate_by_extra_feature_dim_all2d(tracer_dataset):

    flattened = sp._separate_by_extra_feature_dim(tracer_dataset)
    for da in flattened.values():
        assert da.ndim <= 2


def test__separate_by_extra_feature_dim_separated_var(tracer_dataset):

    flattened = sp._separate_by_extra_feature_dim(tracer_dataset)

    for i in range(tracer_dataset.dims["tracer_num"]):
        separated_var = f"field_with_tracer_{i}"
        assert separated_var in flattened
        xr.testing.assert_equal(
            tracer_dataset["field_with_tracer"].isel(tracer_num=i),
            flattened[separated_var],
        )

    assert "field_with_tracer" not in flattened


def test__stack_extra_feature_dim(tracer_dataset):

    flattened = sp._stack_extra_features(tracer_dataset, SAMPLE_DIM_NAME)

    for da in flattened.values():
        assert da.ndim <= 2


def test__stack_extra_feature_dim_multi_feature():

    more_data = np.random.randn(15, 21, 8, 4)
    ds = xr.Dataset(
        {
            "three_features": (
                [SAMPLE_DIM_NAME, "feature_plus_one", "tracer_num", "zy_extra_feature"],
                more_data,
            ),
            "two_features": (
                [SAMPLE_DIM_NAME, "feature_plus_one", "tracer_num"],
                more_data[..., 0],
            ),
            "two_features_diff_dimsize": (
                [SAMPLE_DIM_NAME, "feature", "tracer_num"],
                more_data[:, :20, :, 1],
            ),
        }
    )
    flattened = sp._stack_extra_features(ds, SAMPLE_DIM_NAME)

    for var in ds:
        assert var in flattened
        assert flattened[var].sizes[SAMPLE_DIM_NAME] == 15

    # check contents of three features variable
    # Note: extra dimensions other than
    # sample are sorted for stack in _stack_extra_features, so in order to be
    # equivalent to numpy flatten, the dims are specified in alphabetical order
    np.testing.assert_equal(
        flattened["three_features"].values, more_data.reshape(15, 21 * 8 * 4)
    )


def test_FlatSerialSeq(xr_data):
    seq = sp.SerializedSequence(xr_data, item_dim="savepoint")
    flat_seq = sp.FlattenDims(seq, ["savepoint", "horiz"], dim_name=SAMPLE_DIM_NAME)

    assert len(flat_seq) == len(seq)

    # single item
    dat = flat_seq[0]
    assert SAMPLE_DIM_NAME in dat.dims
    assert dat.dims[SAMPLE_DIM_NAME] == 15  # single savepoint x 15 horizontal

    for da in dat.values():
        assert da.ndim <= 2

    # slice
    dat_slice = flat_seq[0:2]
    assert dat_slice.dims[SAMPLE_DIM_NAME] == 30  # two savepoints x 15 horizontal


def test__check_sample_first(tracer_dataset):
    field = tracer_dataset.field.transpose()
    # confirm sample not first
    assert field.dims[0] != SAMPLE_DIM_NAME
    tracer_dataset["field"] = field

    res = sp._check_sample_first(tracer_dataset, SAMPLE_DIM_NAME)
    for da in res.values():
        assert da.dims[0] == SAMPLE_DIM_NAME


def test__drop_const_vars():
    constant = np.ones((10,), dtype=np.float32)
    # diffs should be below threshold of 1.2e-7
    equiv_constant = constant + np.random.uniform(low=-5e-8, high=5e-8, size=10).astype(
        np.float32
    )
    not_constant = np.arange(0, 10)
    non_numeric = np.array(list(string.ascii_letters[:10]))

    ds = xr.Dataset(
        {
            "constant": (["x"], constant),
            "equiv_constant": (["x"], equiv_constant),
            "not_constant": (["x"], not_constant),
            "non_numeric": (["x"], non_numeric),
        }
    )

    dropped_ds = sp._drop_const_vars(ds)
    assert len(dropped_ds.data_vars) == 1
    assert "constant" not in dropped_ds
    assert "equiv_constant" not in dropped_ds
    assert "not_constant" in dropped_ds
    assert "non_numeric" not in dropped_ds
