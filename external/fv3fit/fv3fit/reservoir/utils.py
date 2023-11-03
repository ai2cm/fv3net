import logging
import numpy as np
import tensorflow as tf
from typing import Iterable, Mapping, Optional, Hashable

from fv3fit.reservoir.transformers import (
    # ReloadableTransformer,
    Transformer,
    encode_columns,
    build_concat_and_scale_only_autoencoder,
)
from fv3fit.reservoir.config import ClipZConfig
from fv3fit.reservoir.domain2 import RankXYDivider
from ._reshaping import stack_array_preserving_last_dim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def assure_txyz_dims(var_data: tf.Tensor) -> tf.Tensor:
    # Assumes dims 1, 2, 3 are t, x, y.
    # If variable data has 3 dims, adds a 4th feature dim of size 1.
    # reshaped_tensors = []
    # for var_data in variable_tensors:
    if len(var_data.shape) == 4:
        reshaped_tensor = var_data
    elif len(var_data.shape) == 3:
        orig_shape = var_data.shape
        reshaped_tensor = tf.reshape(var_data, shape=(*orig_shape, 1))
    else:
        raise ValueError(
            f"Tensor data has {len(var_data.shape)} dims, must either "
            "have either 4 dims (t, x, y, z) or 3 dims (t, x, y)."
        )
    return reshaped_tensor


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
        logger.info(
            "Number of steps synchronized: "
            f"{self.n_steps_synchronized}/{self.n_synchronize}"
        )

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
    reshaped_tensors = [assure_txyz_dims(var_tensor) for var_tensor in ordered_tensors]

    return reshaped_tensors


def process_batch_data(
    variables: Iterable[str],
    batch_data: Mapping[str, tf.Tensor],
    rank_divider: RankXYDivider,
    autoencoder: Optional[Transformer],
    trim_halo: bool,
):
    """ Converts physical state to latent state
    and reshape data into the format used in training.
    The rank divider provided includes the full overlap, since
    the data it is operating on includes all halo points.
    """
    data = get_ordered_X(batch_data, variables)

    # Concatenate features, normalize and optionally convert data
    # to latent representation
    if autoencoder is not None:
        data_encoded = encode_columns(data, autoencoder)

    if trim_halo:
        data_trimmed = rank_divider.trim_halo_from_rank_data(data_encoded)
        no_overlap_rank_divider = rank_divider.get_no_overlap_rank_divider()
        return no_overlap_rank_divider.get_all_subdomains_with_flat_feature(
            data_trimmed
        )
    else:
        data_trimmed = data_encoded
        return rank_divider.get_all_subdomains_with_flat_feature(data_trimmed)


def get_standard_normalizing_transformer(
    variables, sample_batch,
):
    variable_data = get_ordered_X(sample_batch, variables)
    variable_data_stacked = [
        stack_array_preserving_last_dim(arr).numpy() for arr in variable_data
    ]
    return build_concat_and_scale_only_autoencoder(
        variables=variables, X=variable_data_stacked
    )


def clip_batch_data(
    batch: Mapping[str, tf.Tensor],
    clip_config: Optional[Mapping[Hashable, ClipZConfig]],
):
    dim_ordered_batch = {k: assure_txyz_dims(v) for k, v in batch.items()}

    if clip_config is None:
        return dim_ordered_batch
    else:
        clipped_batch = {}
        for var, tensor in batch.items():
            if var in clip_config:
                clipped_batch[var] = tensor[
                    ..., clip_config[var].start : clip_config[var].stop
                ]
                print(f"Clipped {var} to shape {clipped_batch[var].shape}")
            else:
                clipped_batch[var] = tensor

        return clipped_batch
