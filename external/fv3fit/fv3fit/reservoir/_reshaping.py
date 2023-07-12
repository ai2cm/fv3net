import numpy as np
import tensorflow as tf


def flatten_2d_keeping_columns_contiguous(arr: np.ndarray):
    # ex. [[1,2],[3,4], [5,6]] -> [1,3,5,2,4,6]
    return np.reshape(arr, -1, "F")


def concat_inputs_along_subdomain_features(a, b):
    # [time, subdomain-feature, subdomain]
    # Concatenates two input arrays with same time and subdomain dim
    # sizes along the subdomain-feature axis (1)
    return np.concatenate([a, b], axis=1)


def split_1d_samples_into_2d_rows(arr: np.ndarray, n_rows: int) -> np.ndarray:
    # Consecutive chunks of 1d array form rows of 2d array
    # ex. 1d to 2d reshaping (8,) -> (2,4) for n_rows=2
    # [1,2,3,4,5,6,7,8] -> [[1,2,3,4], [5,6,7,8]]
    return np.reshape(arr, (n_rows, -1), order="C")


def stack_array_preserving_last_dim(data):
    original_z_dim = data.shape[-1]
    reshaped = tf.reshape(data, shape=(-1, original_z_dim))
    return reshaped
