import numpy as np
import tensorflow as tf
from typing import Sequence


def flatten_2d_keeping_columns_contiguous(arr: np.ndarray):
    # ex. [[1,2],[3,4], [5,6]] -> [1,3,5,2,4,6]
    return np.reshape(arr, -1, "F")


def stack_array_preserving_last_dim(data):
    original_z_dim = data.shape[-1]
    reshaped = tf.reshape(data, shape=(-1, original_z_dim))
    return reshaped


def encode_columns(data: Sequence[tf.Tensor], encoder: tf.keras.Model,) -> np.ndarray:
    # reduce a sequnence of N x M x Vi dim data over i variables
    # to a single N x M x Z dim array, where Vi is original number of features
    # (usually vertical levels) of each variable and Z << V is a smaller number
    # of latent dimensions.
    original_sample_shape = data[0].shape[:-1]
    reshaped = [stack_array_preserving_last_dim(var) for var in data]
    encoded_reshaped = encoder.predict(reshaped)
    return encoded_reshaped.reshape(*original_sample_shape, -1)


def decode_columns(data: tf.Tensor, decoder: tf.keras.Model) -> Sequence[np.ndarray]:
    # Differs from encode_columns as the decoder expects a single input array
    # (not a list of one array per variable) and
    # can predict multiple outputs rather than a single latent vector.
    # Expand a sequnence of N x M x L dim data into i variables
    # to one or more N x M x Vi dim array, where Vi is number of features
    # (usually vertical levels) of each variable and L << V is a smaller number
    # of latent dimensions
    reshaped = stack_array_preserving_last_dim(data)
    decoded_reshaped = decoder.predict(reshaped)
    original_2d_shape = data.shape[:-1]
    if len(decoder.outputs) == 1:
        return decoded_reshaped.reshape(*original_2d_shape, -1)
    else:
        decoded_data = []
        for i, var_data in enumerate(decoded_reshaped):
            decoded_data.append(decoded_reshaped[i].reshape(*original_2d_shape, -1))
        return decoded_data


def split_1d_into_2d_rows(arr: np.ndarray, n_rows: int) -> np.ndarray:
    # Consecutive chunks of 1d array form rows of 2d array
    # ex. 1d to 2d reshaping (8,) -> (2,4)) for n_rows=2
    # [1,2,3,4,5,6,7,8] -> [[1,2,3,4], [5,6,7,8]]
    return np.reshape(arr, (n_rows, -1), order="C")


def split_1d_into_2d_columns(arr: np.ndarray, n_columns: int) -> np.ndarray:
    # Consecutive chunks of 1d array form columns of 2d array
    # ex. 1d to 2d reshaping (8,) -> (2, 4)
    # [1,2,3,4,5,6,7,8] -> [[1,3,5,7], [2,4,6,8]]Zx
    return np.reshape(arr, (-1, n_columns), "F")


def concat_inputs_along_subdomain_features(a, b):
    # [time, subdomain-feature, subdomain]
    # Concatenates two input arrays with same time and subdomain dim
    # sizes along the subdomain-feature axis (1)
    return np.concatenate([a, b], axis=1)


def stack_samples(tensor, keep_first_dim: bool):
    # Used to reshape a subdomains into a flat columns.
    # Option to keep first dim, used in the case where time is the first dimension
    if keep_first_dim is True:
        n_samples = tensor.shape[0]
        return np.reshape(tensor, (n_samples, -1))
    else:
        return np.reshape(tensor, -1)
