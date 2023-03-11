import numpy as np
import pytest
from fv3fit.reservoir.domain import (
    slice_along_axis,
    RankDivider,
    DataReshaper,
    concat_variables_along_feature_dim,
    flatten_subdomains_to_columns,
)


arr = np.arange(3)


@pytest.mark.parametrize(
    "data, ax, sl, expected",
    [
        (arr, 0, slice(1, None), np.array([1, 2])),
        (np.array([arr, arr, arr]), 0, slice(0, 2), np.array([arr, arr])),
        (np.array([arr, arr]), 1, slice(0, 2), np.array([[0, 1], [0, 1]])),
    ],
)
def test_slice_along_axis(data, ax, sl, expected):
    np.testing.assert_array_equal(slice_along_axis(data, axis=ax, inds=sl), expected)


default_rank_divider_kwargs = {
    "layout": [2, 2],
    "overlap": 1,
    "rank_dims": ["time", "x", "y", "z"],
    "rank_extent": [5, 6, 6, 7],
}


def get_reshaper(variables, **rank_divider_kwargs):
    return DataReshaper(
        variables=variables, rank_divider=RankDivider(**rank_divider_kwargs)
    )


def test_concat_variables_along_feature_dim():
    nt, nx, ny, nz = 5, 4, 4, 6
    arr0 = np.zeros((nt, nx, ny, nz))
    arr1 = np.ones((nt, nx, ny, nz))
    data_mapping = {"var1": arr1, "var0": arr0}

    concat_data = concat_variables_along_feature_dim(["var0", "var1"], data_mapping)
    assert concat_data.shape == (nt, nx, ny, nz * 2)
    np.testing.assert_array_equal(concat_data[:, :, :, :nz], arr0)
    np.testing.assert_array_equal(concat_data[:, :, :, nz:], arr1)


def test_flatten_subdomains_to_columns():
    input = np.array(
        [[0, 1, 10, 11], [2, 3, 12, 13], [20, 21, 30, 31], [22, 23, 32, 33]]
    )
    input_with_halo = np.pad(input, pad_width=1)
    # add time dim of length 2
    input_with_halo = np.stack([input_with_halo, input_with_halo], axis=0)
    flatten_subdomains_to_columns()
