import xarray as xr
import numpy as np


def shuffled(
    dataset: xr.Dataset, dim: str, random: np.random.RandomState
) -> xr.Dataset:
    """
    Shuffles dataset along a dimension within chunks if chunking is present

    Args:
        dataset: input data to be shuffled
        dim: dimension to shuffle indices along
        random: Initialized random number generator state used for shuffling
    """
    chunks_default = (len(dataset[dim]),)
    chunks = dataset.chunks.get(dim, chunks_default)
    chunk_indices = _get_chunk_indices(chunks)
    shuffled_inds = np.concatenate(
        [random.permutation(indices) for indices in chunk_indices]
    )

    return dataset.isel({dim: shuffled_inds})


def _get_chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices
