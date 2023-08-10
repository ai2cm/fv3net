import numpy as np
import tensorflow as tf
from typing import Iterable, Mapping, Tuple
from fv3fit.reservoir.transformers import (
    # ReloadableTransformer,
    Transformer,
    encode_columns,
    build_concat_and_scale_only_autoencoder,
)
from fv3fit.reservoir.domain import assure_txyz_dims
from fv3fit.reservoir.domain2 import RankXYDivider
from ._reshaping import stack_array_preserving_last_dim


class SynchronziationTracker:
    """Counts the number of times a reservoir has been incremented,
    and excludes time series data from training set if the number of
    incrments is less than the specified synchronization length.
    """

    def __init__(self, n_synchronize: int):
        self.n_synchronize = n_synchronize
        self.n_steps_synchronized = 0

    @property
    def completed_synchronization(self):
        if self.n_steps_synchronized > self.n_synchronize:
            return True
        else:
            return False

    def count_synchronization_steps(self, n_samples: int):
        self.n_steps_synchronized += n_samples

    def trim_synchronization_samples_if_needed(self, arr: np.ndarray) -> np.ndarray:
        """ Removes samples from the input array if they fall within the
        synchronization range.
        """
        if self.completed_synchronization is True:
            steps_past_sync = self.n_steps_synchronized - self.n_synchronize
            if steps_past_sync > len(arr):
                return arr
            else:
                return arr[-steps_past_sync:]
        else:
            return np.array([])


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
    autoencoder: Transformer,
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


def get_standard_normalizing_transformer(variables, sample_batch):
    variable_data = get_ordered_X(sample_batch, variables)
    variable_data_stacked = [
        stack_array_preserving_last_dim(arr).numpy() for arr in variable_data
    ]
    return build_concat_and_scale_only_autoencoder(
        variables=variables, X=variable_data_stacked
    )
