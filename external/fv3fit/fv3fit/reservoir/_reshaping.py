import numpy as np


def flatten_2d_keeping_columns_contiguous(arr: np.ndarray):
    # ex. [[1,2],[3,4], [5,6]] -> [1,3,5,2,4,6]
    return np.reshape(arr, -1, "F")


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
