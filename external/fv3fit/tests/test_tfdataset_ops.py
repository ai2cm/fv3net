import copy
from fv3fit._shared.config import SliceConfig
from fv3fit._shared.packer import (
    pack_tfdataset,
    clip_sample,
)
from fv3fit.tfdataset import tfdataset_from_batches
import tensorflow as tf
from typing import Mapping, Sequence
import numpy as np
import pytest
import xarray as xr
import fv3fit.tfdataset


def assert_datasets_equal(dataset1, dataset2, rtol=1e-6):
    sample = next(iter(dataset1))
    if isinstance(sample, dict):
        for name in sample:
            result = tf.data.Dataset.zip((dataset1, dataset2)).map(
                lambda x1, x2: tf.reduce_any(
                    tf.abs(x2[name] - x1[name]) / (0.5 * (x2[name] + x1[name])) > rtol
                )
            )
        assert not tf.reduce_any(any(result)), name
    else:
        result = tf.data.Dataset.zip((dataset1, dataset2)).map(
            lambda x1, x2: tf.reduce_any(tf.abs(x2 - x1) / (0.5 * (x2 + x1)) > rtol)
        )
        assert not tf.reduce_any(any(result))


def get_tfdataset(
    n_dims: int, n_samples: int, entry_features: Mapping[str, int],
):
    if n_dims < 2:
        raise ValueError(
            "n_dims must be at least 2 (requires sample, feature dimensions)"
        )
    data = {}
    for name, n_features in entry_features.items():
        shape = list(range(2, n_dims + 2))
        shape[-1] = n_features
        shape[0] = n_samples
        data[name] = tf.convert_to_tensor(
            np.random.uniform(size=shape).astype(np.float32)
        )
    return get_tfdataset_from_data(data)


def get_tfdataset_from_data(data):
    def gen():
        yield data

    sample = next(iter(gen()))
    if isinstance(sample, dict):
        output_signature = {
            key: tf.TensorSpec(val.shape, dtype=val.dtype)
            for key, val in sample.items()
        }
    else:
        output_signature = tf.TensorSpec(sample.shape, dtype=sample.dtype)
    return tf.data.Dataset.from_generator(
        gen, output_signature=output_signature
    ).unbatch()


def test_assert_datasets_equal_raises():
    dataset_1 = get_tfdataset_from_data(tf.convert_to_tensor(np.asarray([0.0])))
    dataset_2 = get_tfdataset_from_data(tf.convert_to_tensor(np.asarray([1.0])))
    with pytest.raises(AssertionError):
        assert_datasets_equal(dataset_1, dataset_2, rtol=0.1)


def test_tfdataset_from_batches_empty():
    with pytest.raises(NotImplementedError):
        tfdataset_from_batches([])


def test_tfdataset_from_batches_single():
    n_samples = 3
    xr_dataset = xr.Dataset(
        data_vars={
            "a": xr.DataArray(np.random.uniform(size=[n_samples, 3, 4])),
            "b": xr.DataArray(np.random.uniform(size=[n_samples, 3])),
        }
    )
    tfdataset = tfdataset_from_batches([xr_dataset])
    for i, result in enumerate(iter(tfdataset)):
        assert isinstance(result, dict)
        np.testing.assert_array_equal(result["a"], xr_dataset["a"].values[i, :])
        np.testing.assert_array_equal(result["b"], xr_dataset["b"].values[i, :])


def test_tfdataset_from_batches_multiple():
    n_samples = 3
    batches = [
        xr.Dataset(
            data_vars={
                "a": xr.DataArray(np.random.uniform(size=[n_samples, 3, 4])),
                "b": xr.DataArray(np.random.uniform(size=[n_samples, 3])),
            }
        )
        for _ in range(3)
    ]
    tfdataset = tfdataset_from_batches(batches)
    for i, result in enumerate(iter(tfdataset)):
        assert isinstance(result, dict)
        i_sample = i % n_samples
        i_batch = i // n_samples
        np.testing.assert_array_equal(
            result["a"], batches[i_batch]["a"].values[i_sample, :]
        )
        np.testing.assert_array_equal(
            result["b"], batches[i_batch]["b"].values[i_sample, :]
        )


@pytest.mark.parametrize("n_dims", [2, 3, 5])
@pytest.mark.parametrize(
    "variable_names",
    [
        pytest.param(["a", "b", "c", "d"], id="all"),
        pytest.param(["b"], id="one"),
        pytest.param(["a", "c"], id="half"),
    ],
)
def test_pack_tfdataset(n_dims: int, variable_names: Sequence[str]):
    entry_features = {
        "a": 5,
        "b": 7,
        "c": 1,
        "d": 1,
    }
    dataset = get_tfdataset(n_dims=n_dims, n_samples=10, entry_features=entry_features)
    packed, packing_info = pack_tfdataset(dataset, variable_names=variable_names)
    packed_sum = tf.reduce_sum(sum(packed))
    expected_sum = sum(
        tf.reduce_sum(sum(dataset.map(lambda x: x[name]))) for name in variable_names
    )
    np.testing.assert_allclose(packed_sum.numpy(), expected_sum.numpy(), rtol=1e-6)
    assert packing_info.names == variable_names
    assert packing_info.features == [entry_features[name] for name in variable_names]
    sample = next(iter(packed))
    assert sample.shape[-1] == sum(
        n for (name, n) in entry_features.items() if name in variable_names
    )


@pytest.mark.parametrize("n_dims", [2, 3, 5])
@pytest.mark.parametrize(
    "config, clipped_features",
    [
        pytest.param(
            {"a": SliceConfig(start=1)},
            {"a": 4, "b": 7, "c": 1, "d": 1},
            id="a_start_1",
        ),
        pytest.param(
            {"a": SliceConfig(stop=-1)},
            {"a": 4, "b": 7, "c": 1, "d": 1},
            id="a_stop_1",
        ),
        pytest.param(
            {"b": SliceConfig(start=2, stop=-1)},
            {"a": 5, "b": 4, "c": 1, "d": 1},
            id="b_start_stop_1",
        ),
    ],
)
def test_clip_tfdataset(
    n_dims, config: Mapping[str, SliceConfig], clipped_features: Mapping[str, int]
):
    entry_features = {
        "a": 5,
        "b": 7,
        "c": 1,
        "d": 1,
    }
    dataset = get_tfdataset(n_dims=n_dims, n_samples=10, entry_features=entry_features)
    clipped = dataset.map(clip_sample(config))
    sample_in = next(iter(dataset))
    sample_out = next(iter(clipped))
    assert sample_out.keys() == sample_in.keys()
    for name, value in sample_out.items():
        assert value.shape[-1] == clipped_features[name], name
        if name in config:
            np.testing.assert_array_equal(
                sample_in[name][..., config[name].slice], sample_out[name]
            )
        else:
            np.testing.assert_array_equal(sample_in[name], sample_out[name])


def test__seq_to_tfdataset():
    batches = [{"a": np.arange(30).reshape(10, 3)} for _ in range(3)]

    def transform(batch):
        out = copy.deepcopy(batch)
        out["a"] = out["a"] * 2
        return out

    tf_ds = fv3fit.tfdataset.seq_to_tfdataset(batches, transform)
    assert isinstance(tf_ds, tf.data.Dataset)

    result = next(tf_ds.batch(10).as_numpy_iterator())
    assert isinstance(result, dict)
    np.testing.assert_equal(result["a"], batches[0]["a"] * 2)
