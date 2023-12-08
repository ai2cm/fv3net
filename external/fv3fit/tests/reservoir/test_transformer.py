import numpy as np
import pytest

from fv3fit.reservoir.transformers.transformer import DoNothingAutoencoder


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

    encoded = transformer.encode_txyz(data_arrs)

    assert encoded.shape == expected_shape


@pytest.mark.parametrize(
    "nx, ny, nz, nvars", [(4, 4, 3, 2), (2, 2, 1, 1), (2, 2, 1, 2)]
)
def test_base_decode_txyz(nx, ny, nz, nvars):
    expected_shapes = [(nx, ny, nz) for var in range(nvars)]

    transformer = DoNothingAutoencoder([nz for var in range(nvars)])

    # need to call encode before decode
    data_arrs = [np.random.rand(*shape) for shape in expected_shapes]
    transformer.encode_txyz(data_arrs)

    encoded_input = np.random.rand(nx, ny, transformer.n_latent_dims)
    decoded = transformer.decode_txyz(encoded_input)

    assert len(expected_shapes) == len(decoded)
    for expected_shape, decoded_output in zip(expected_shapes, decoded):
        assert expected_shape == decoded_output.shape
