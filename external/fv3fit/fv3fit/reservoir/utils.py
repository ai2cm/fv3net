import numpy as np
import tensorflow as tf
from typing import Iterable, Mapping, Tuple
from fv3fit.reservoir.transformers import ReloadableTransfomer, encode_columns
from fv3fit.reservoir.domain import RankDivider, assure_txyz_dims


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
    rank_divider: RankDivider,
    autoencoder: ReloadableTransfomer,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Convert physical state to corresponding reservoir hidden state,
    and reshape data into the format used in training.
    """
    batch_X = get_ordered_X(batch_data, variables)

    # Concatenate features, normalize and optionally convert data
    # to latent representation
    batch_data_encoded = encode_columns(batch_X, autoencoder)

    time_series_X_reshaped, time_series_Y_reshaped = [], []
    for timestep_data in batch_data_encoded:
        # Divide into subdomains and flatten each subdomain by stacking
        # x/y/encoded-feature dims into a single subdomain-feature dimension.
        # Dimensions of a single subdomain's data become [time, subdomain-feature]
        X_subdomains_as_columns, Y_subdomains_as_columns = [], []
        for s in range(rank_divider.n_subdomains):
            X_subdomain_data = rank_divider.get_subdomain_tensor_slice(
                timestep_data, subdomain_index=s, with_overlap=True,
            )
            X_subdomains_as_columns.append(np.reshape(X_subdomain_data, -1))

            # Prediction does not include overlap
            Y_subdomain_data = rank_divider.get_subdomain_tensor_slice(
                timestep_data, subdomain_index=s, with_overlap=False,
            )
            Y_subdomains_as_columns.append(np.reshape(Y_subdomain_data, -1))
        # Concatentate subdomain data arrays along a new subdomain axis.
        # Dimensions are now [time, subdomain-feature, subdomain]
        X_reshaped = np.stack(X_subdomains_as_columns, axis=-1)
        Y_reshaped = np.stack(Y_subdomains_as_columns, axis=-1)

        time_series_X_reshaped.append(X_reshaped)
        time_series_Y_reshaped.append(Y_reshaped)

    return (
        np.stack(time_series_X_reshaped, axis=0),
        np.stack(time_series_Y_reshaped, axis=0),
    )
