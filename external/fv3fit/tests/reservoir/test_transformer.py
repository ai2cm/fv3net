import numpy as np
import tempfile
import pytest

from fv3fit.reservoir.transformers.transformer import (
    DoNothingAutoencoder,
    build_scale_spatial_concat_z_transformer,
)


@pytest.mark.parametrize("nz, nvars", [(2, 2), (2, 1), (1, 2), (1, 1)])
def test_DoNothingAutoencoder(nz, nvars):
    nx = 5
    transformer = DoNothingAutoencoder([nz for var in range(nvars)])
    data = [np.ones((nx, nz)) for var in range(nvars)]
    transformer.encode(data)
    assert transformer.original_feature_sizes == [nz for var in range(nvars)]
    encoded_data = np.ones(nz * nvars)
    assert len(transformer.decode(encoded_data)) == len(data)


@pytest.mark.parametrize(
    "nt, nx, ny, nz, nvars",
    [(20, 4, 4, 3, 2), (None, 2, 2, 1, 1), (None, 2, 2, None, 1)],
)
def test_base_encode_txyz(nt, nx, ny, nz, nvars):
    shape = tuple([y for y in [nt, nx, ny, nz] if y is not None])
    expected_shape = (*shape[:-1], nz * nvars) if nz is not None else shape
    transformer = DoNothingAutoencoder([nz for var in range(nvars)])
    data_arrs = [np.random.rand(*shape) for var in range(nvars)]

    encoded = transformer.encode_unstacked_xyz(data_arrs)

    assert encoded.shape == expected_shape


@pytest.mark.parametrize(
    "nx, ny, nz, nvars", [(4, 4, 3, 2), (2, 2, 1, 1), (2, 2, 1, 2)]
)
def test_base_decode_txyz(nx, ny, nz, nvars):
    expected_shapes = [(nx, ny, nz) for var in range(nvars)]

    transformer = DoNothingAutoencoder([nz for var in range(nvars)])

    # need to call encode before decode
    data_arrs = [np.random.rand(*shape) for shape in expected_shapes]
    transformer.encode_unstacked_xyz(data_arrs)

    encoded_input = np.random.rand(nx, ny, transformer.n_latent_dims)
    decoded = transformer.decode_unstacked_xyz(encoded_input)

    assert len(expected_shapes) == len(decoded)
    for expected_shape, decoded_output in zip(expected_shapes, decoded):
        assert expected_shape == decoded_output.shape


def _get_sample_xyz_data(nx, ny, nz):
    nfeatures = nx * ny * nz
    arr_shape = [1, nx, ny, nz]

    xyz_base = np.arange(1, nfeatures + 1).reshape(arr_shape).astype(float)
    # create arrays that will std to 1 everywhere across sample dim
    a = np.concatenate([-10 * xyz_base, 10 * xyz_base], axis=0)  # mean = 0, std = 10
    b = np.concatenate([-1 * xyz_base, xyz_base], axis=0)  # mean = 1, std = 1
    return [a, b]


@pytest.mark.parametrize("nz", [1, 2], ids=["single-level", "multi-level",])
def test_scale_per_feature_concat_z_transform(nz):
    nx, ny = 3, 4
    a, b = _get_sample_xyz_data(nx, ny, nz)

    xyz_like = np.ones((1, nx, ny, nz))
    normalized = np.concatenate([-1 * xyz_like, 1 * xyz_like], axis=0)
    normalized_and_stacked = np.concatenate([normalized, normalized], axis=-1)

    transformer = build_scale_spatial_concat_z_transformer([a, b])
    encoded = transformer.encode_unstacked_xyz([a, b])

    # test that it normalizes
    nsamples, nvars = 2, 2
    assert encoded.shape == (nsamples, nx, ny, nz * nvars)
    np.testing.assert_allclose(encoded, normalized_and_stacked, rtol=1e-6)

    # test round trip
    decoded = transformer.decode_unstacked_xyz(encoded)
    np.testing.assert_allclose(decoded, [a, b], rtol=1e-6)


def test_scale_per_feature_concat_z_transform_no_leading_dim():
    nx, ny, nz = 3, 4, 1
    a, b = _get_sample_xyz_data(nx, ny, nz)

    xyz_like = -1 * np.ones((nx, ny, nz))
    normalized_and_stacked = np.concatenate([xyz_like, xyz_like], axis=-1)

    transformer = build_scale_spatial_concat_z_transformer([a, b])
    encoded = transformer.encode_unstacked_xyz([a[0], b[0]])

    # test that it normalizes
    nvars = 2
    assert encoded.shape == (nx, ny, nz * nvars)
    np.testing.assert_allclose(encoded, normalized_and_stacked, rtol=1e-6)

    # test round trip
    decoded = transformer.decode_unstacked_xyz(encoded)
    np.testing.assert_allclose(decoded, [a[0], b[0]], rtol=1e-6)


@pytest.mark.parametrize("nz", [1, 2], ids=["single-level", "multi-level",])
def test_scale_per_feature_concat_z_transform_mask(nz):
    nx, ny = 3, 4
    a, _ = _get_sample_xyz_data(nx, ny, nz)

    xyz_like = np.ones((1, nx, ny, nz))
    normalized = np.concatenate([-1 * xyz_like, 1 * xyz_like], axis=0)

    mask = xyz_like.copy()
    mask[..., 0, :] = 0  # mask y fields of a

    transformer = build_scale_spatial_concat_z_transformer([a], mask=mask)
    a_adj = a.copy()
    a_adj[..., 0, :] = 25
    encoded = transformer.encode_unstacked_xyz([a_adj])
    np.testing.assert_allclose(encoded, normalized * mask, rtol=1e-6)

    encoded[..., 0, :] = 25  # adjust normalized a
    decoded = transformer.decode_unstacked_xyz(encoded)
    a_adj[..., 0, :] = 0
    np.testing.assert_allclose(decoded, [a_adj], rtol=1e-6)


# test dump load is equivalent
def test_scale_per_feature_concat_z_transfor_dumpload():
    nx, ny, nz = 3, 4, 1
    a, b = _get_sample_xyz_data(nx, ny, nz)

    transformer = build_scale_spatial_concat_z_transformer([a, b])
    encoded = transformer.encode_unstacked_xyz([a, b])

    # test dump load
    with tempfile.TemporaryDirectory() as tmpdir:
        transformer.dump(tmpdir)
        loaded_transformer = transformer.load(tmpdir)

    loaded_encoded = loaded_transformer.encode_unstacked_xyz([a, b])
    np.testing.assert_allclose(encoded, loaded_encoded, rtol=1e-6)

    loaded_decoded = loaded_transformer.decode_unstacked_xyz(loaded_encoded)
    np.testing.assert_allclose(loaded_decoded, [a, b], rtol=1e-6)


# test that dims < 4 throws error
def test_scale_per_feature_concat_z_transform_not_enough_dims():
    nx, ny, nz = 3, 4, 1
    a, b = _get_sample_xyz_data(nx, ny, nz)

    with pytest.raises(ValueError):
        build_scale_spatial_concat_z_transformer([a[0, :, :, :]])


# test inconsistent shapes
def test_scale_per_feature_concat_z_transform_inconsistent_dims():
    """Add these tests since we hard code the spatial handling"""
    nx, ny, nz = 3, 4, 1
    a, b = _get_sample_xyz_data(nx, ny, nz)

    # inconsistent dimensions for fit
    with pytest.raises(ValueError):
        build_scale_spatial_concat_z_transformer([a, b[:, :, 0:2]])

    transformer = build_scale_spatial_concat_z_transformer([a, b])

    # both inconsistent w/ fit data but with same dimensions
    with pytest.raises(ValueError):
        transformer.encode_unstacked_xyz([a[..., 0], b[..., 0]])

    # one inconsistent w/ fit data
    with pytest.raises(ValueError):
        transformer.encode_unstacked_xyz([a, b[:, :, :, 0]])
