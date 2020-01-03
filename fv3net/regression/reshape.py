"""Convert netCDF dataset to a memory efficient layout for machine learning

The original dataset should have the dimensions ('time', 'z', 'y', 'x'). This
script combines the y and x dimensions into a sample dimension with a specified
chunk size. This sample dimension is optionally shufffled, and then saved to a
zarr archive.

The saved archive is suitable to use with uwnet.train
"""

import numpy as np

from vcm.cubedsphere.constants import COORD_X_CENTER, COORD_Y_CENTER
from vcm.convenience import open_data


def chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices


def shuffled_within_chunks(indices, random_seed):
    np.random.seed(random_seed)
    return np.concatenate([np.random.permutation(index) for index in indices])


def shuffled(dataset, dim, random_seed):
    indices = chunk_indices(dataset.chunks[dim])
    shuffled_inds = shuffled_within_chunks(indices, random_seed)
    return dataset.isel({dim: shuffled_inds})


if __name__ == "__main__":
    chunk_size = 500_000
    output_file = "data/processed/flattened.zarr"
    shuffle = True
    valid_latitudes = slice(-80, 80)

    ds = open_data(sources=True)

    variables = "u v w temp q1 q2 qv pres z fsdt lhflx shflx test_case".split()
    sample_dims = ["time", COORD_Y_CENTER, COORD_X_CENTER]

    # compute a simple test case which can be reproduced by ML
    test_case = (ds.qv + ds.temp).assign_attrs(
        formula="QV + TEMP", description="Test for machine learning"
    )
    ds["test_case"] = test_case

    # stack data
    variables = list(variables)  # needs to be a list for xarray
    stacked = (
        ds[variables]
        .sel({COORD_Y_CENTER: valid_latitudes})
        .stack(sample=sample_dims)
        .transpose("sample", "pfull")
        .drop("sample")
        .dropna("sample")
    )

    # Chunk the data
    chunked = stacked.chunk({"sample": chunk_size})

    # Shuffle the indices
    if shuffle:
        chunked = shuffled(chunked, "sample")

    chunked.to_zarr(output_file, mode="w")
