import numpy as np
import os
import pytest

from fv3fit.reservoir.transformers.transformer import (
    encode_columns,
    DoNothingAutoencoder,
    decode_columns,
    TransformerGroup,
)


@pytest.mark.parametrize("hybrid_transformer", [None, DoNothingAutoencoder([2])])
def test_TransformerGroup(tmpdir, hybrid_transformer):
    input_transformer = DoNothingAutoencoder([4])
    output_transformer = DoNothingAutoencoder([2])

    transformers = TransformerGroup(
        input=input_transformer, output=output_transformer, hybrid=hybrid_transformer,
    )
    n_saved_transformers = 3 if hybrid_transformer is not None else 2
    transformers.dump(str(tmpdir))
    assert len(os.listdir(str(tmpdir))) == n_saved_transformers
    loaded_transformers = TransformerGroup.load(str(tmpdir))

    x = np.random.rand(2)
    if hybrid_transformer is not None:
        np.testing.assert_array_equal(
            loaded_transformers.hybrid.encode([x]), hybrid_transformer.encode([x])
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
def test_encode_columns(nt, nx, ny, nz, nvars):
    shape = tuple([y for y in [nt, nx, ny, nz] if y is not None])
    expected_shape = (*shape[:-1], nz * nvars) if nz is not None else shape
    transformer = DoNothingAutoencoder([nz for var in range(nvars)])
    data_arrs = [np.random.rand(*shape) for var in range(nvars)]

    encoded = encode_columns(data_arrs, transformer=transformer)

    assert encoded.shape == expected_shape


@pytest.mark.parametrize(
    "nx, ny, nz, nvars", [(4, 4, 3, 2), (2, 2, 1, 1), (2, 2, 1, 2)]
)
def test_decode_columns(nx, ny, nz, nvars):
    expected_shapes = [
        tuple([y for y in [nx, ny, nz] if y is not None]) for var in range(nvars)
    ]

    encoded_input_shape = (
        (*expected_shapes[0][:-1], nz * nvars) if nz is not None else expected_shapes[0]
    )
    transformer = DoNothingAutoencoder([nz for var in range(nvars)])

    # need to call encode before decode
    data_arrs = [np.random.rand(*shape) for shape in expected_shapes]
    transformer.encode(data_arrs)

    encoded_input = np.random.rand(*encoded_input_shape)
    decoded = decode_columns(encoded_input, transformer=transformer)

    assert len(expected_shapes) == len(decoded)
    for expected_shape, decoded_output in zip(expected_shapes, decoded):
        assert expected_shape == decoded_output.shape
