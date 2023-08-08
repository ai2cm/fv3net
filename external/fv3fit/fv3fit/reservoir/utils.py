import numpy as np
import tensorflow as tf
from typing import Iterable, Mapping, Tuple
from fv3fit.reservoir.transformers import ReloadableTransfomer, encode_columns
from fv3fit.reservoir.domain import assure_txyz_dims
from fv3fit.reservoir.domain2 import RankXYDivider


def _square_evens(v: np.ndarray) -> np.ndarray:
    evens = v[::2]
    odds = v[1::2]
    c = np.empty((v.size,), dtype=v.dtype)
    c[0::2] = evens ** 2
    c[1::2] = odds
    return c


def square_even_terms(v: np.ndarray, axis=1) -> np.ndarray:
    return np.apply_along_axis(func1d=_square_evens, axis=axis, arr=v)


def get_ordered_X(X: Mapping[str, tf.Tensor], variables: Iterable[str]):
    ordered_tensors = [X[v] for v in variables]
    return assure_txyz_dims(ordered_tensors)


def process_batch_Xy_data(
    variables: Iterable[str],
    batch_data: Mapping[str, tf.Tensor],
    rank_divider: RankXYDivider,
    autoencoder: ReloadableTransfomer,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Convert physical state to corresponding reservoir hidden state,
    and reshape data into the format used in training.
    """
    batch_X = get_ordered_X(batch_data, variables)

    # Concatenate features, normalize and optionally convert data
    # to latent representation
    batch_data_encoded = encode_columns(batch_X, autoencoder)

    X_flat = rank_divider.get_all_subdomains_with_flat_feature(batch_data_encoded)
    Y_no_halo = rank_divider.trim_halo_from_rank_data(batch_data_encoded)

    no_overlap_divider = rank_divider.get_no_overlap_rank_divider()
    Y_flat = no_overlap_divider.get_all_subdomains_with_flat_feature(Y_no_halo)

    return X_flat, Y_flat
