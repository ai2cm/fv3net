import numpy as np


def flatten_2d_keeping_columns_contiguous(arr: np.ndarray):
    # ex. [[1,2],[3,4], [5,6]] -> [1,3,5,2,4,6]
    return np.reshape(arr, -1, "F")


def concat_inputs_along_subdomain_features(a, b):
    # [time, subdomain-feature, subdomain]
    # Concatenates two input arrays with same time and subdomain dim
    # sizes along the subdomain-feature axis (1)
    return np.concatenate([a, b], axis=1)


def stack_data(tensor, keep_first_dim: bool):
    # Used to reshape a subdomains into a flat columns.
    # Option to keep first dim, used in the case where time is the first dimension
    if keep_first_dim is True:
        n_samples = tensor.shape[0]
        return np.reshape(tensor, (n_samples, -1))
    else:
        return np.reshape(tensor, -1)


def split_1d_samples_into_2d_rows(
    arr: np.ndarray, n_rows: int, keep_first_dim_shape: bool
) -> np.ndarray:
    # Consecutive chunks of 1d array form rows of 2d array
    # ex. 1d to 2d reshaping (8,) -> (2,4) for n_rows=2
    # [1,2,3,4,5,6,7,8] -> [[1,2,3,4], [5,6,7,8]]
    if keep_first_dim_shape is True:
        time_dim_size = arr.shape[0]
        return np.reshape(arr, (time_dim_size, n_rows, -1), order="C")
    else:
        return np.reshape(arr, (n_rows, -1), order="C")
