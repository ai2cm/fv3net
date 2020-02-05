import fsspec
import joblib
import numpy as np
import xarray as xr

from ..dataset_handler import stack_and_drop_nan_samples
from vcm.cloud import gsutil
from vcm.fv3_restarts import _split_url
from vcm.cubedsphere.constants import INIT_TIME_DIM

SAMPLE_DIM = "sample"


def load_test_dataset(test_data_path, num_files_to_load=50):
    """

    Args:
        test_data_path: path to dir that contains test data, which assumed to be saved
        in multiple batches in zarrs
        num_files_to_load: number of files to concat into final test dataset

    Returns:
        xarray dataset created by concatenating test data zarrs
    """
    protocol, _ = _split_url(test_data_path)
    fs = fsspec.filesystem(protocol)
    zarrs_in_test_dir = [file for file in fs.ls(test_data_path) if ".zarr" in file]
    if len(zarrs_in_test_dir) < 1:
        raise ValueError(f"No .zarr files found in  {test_data_path}.")
    np.random.shuffle(zarrs_in_test_dir)

    ds_test = xr.concat(
            map(xr.open_zarr,
                [fs.get_mapper(file_path)
                 for file_path in zarrs_in_test_dir[:num_files_to_load]]),
        INIT_TIME_DIM)
    ds_stacked = stack_and_drop_nan_samples(ds_test)
    return ds_stacked


def predict_dataset(sk_wrapped_model_path, ds_stacked):
    """

    Args:
        sk_wrapped_model_path: location of trained model wrapped in the SklearnWrapper
        ds_stacked: dataset with features and targets data variables, stacked in a
        single sample dimension
        stack_dim: dimension to stack along

    Returns:

    """
    sk_wrapped_model = _load_model(sk_wrapped_model_path)
    ds_pred = sk_wrapped_model.predict(
        ds_stacked[sk_wrapped_model.input_vars_], SAMPLE_DIM)
    return ds_pred


def _load_model(model_path):
    protocol, _ = _split_url(model_path)
    if protocol == "gs":
        gsutil.copy(model_path, "temp_model.pkl")
        return joblib.load("temp_model.pkl")
    else:
        return joblib.load(model_path)




