import fsspec
import joblib
import xarray as xr

from ..dataset_handler import stack_and_drop_nan_samples
from vcm.cloud import gsutil
from vcm.convenience import round_time
from vcm.fv3_restarts import _split_url
from vcm.cubedsphere.constants import INIT_TIME_DIM

SAMPLE_DIM = "sample"
KEEP_VARS = ["delp", "slope", "precip_sfc"]


def load_test_dataset(test_data_path, num_files_to_load=50, downsample_time_factor=1):
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
    test_data_urls = load_downsampled_time_range(
        fs, test_data_path, downsample_time_factor)
    zarrs_in_test_dir = sorted([file for file in test_data_urls if ".zarr" in file])
    if len(zarrs_in_test_dir) < 1:
        raise ValueError(f"No .zarr files found in  {test_data_path}.")
    ds_test = xr.concat(
        [xr.open_zarr(fs.get_mapper(file_path), consolidated=True)
            for file_path in zarrs_in_test_dir[:num_files_to_load]],
        INIT_TIME_DIM,
    )
    ds_test = ds_test.assign_coords(
        {INIT_TIME_DIM: [round_time(t) for t in ds_test[INIT_TIME_DIM].values]}
    )
    ds_stacked = stack_and_drop_nan_samples(ds_test)
    return ds_stacked


def load_downsampled_time_range(
        fs,
        test_data_path,
        downsample_time_factor
):
    sorted_urls = sorted(fs.ls(test_data_path))
    downsampled_urls = [
        sorted_urls[i * downsample_time_factor]
        for i in range(int(len(sorted_urls) / downsample_time_factor))
    ]
    return downsampled_urls
    

def predict_dataset(sk_wrapped_model, ds_stacked):
    """

    Args:
        sk_wrapped_model_path: location of trained model wrapped in the SklearnWrapper
        ds_stacked: dataset with features and targets data variables, stacked in a
        single sample dimension
        stack_dim: dimension to stack along

    Returns:
        Unstacked prediction dataset
    """
    ds_keep_vars = ds_stacked[KEEP_VARS]
    ds_pred = sk_wrapped_model.predict(
        ds_stacked[sk_wrapped_model.input_vars_], SAMPLE_DIM
    )
    return xr.merge([ds_pred, ds_keep_vars]).unstack()


def load_model(model_path):
    protocol, _ = _split_url(model_path)
    if protocol == "gs":
        gsutil.copy(model_path, "temp_model.pkl")
        return joblib.load("temp_model.pkl")
    else:
        return joblib.load(model_path)
